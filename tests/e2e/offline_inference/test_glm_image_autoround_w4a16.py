# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for GLM-Image AutoRound W4A16 quantized inference.

These tests require:
  - 2 CUDA GPUs
  - The quantized model checkpoint (Intel/GLM-Image-int4-AutoRound)
  - The baseline model checkpoint (zai-org/GLM-Image)
"""

import gc
import math
import os

import pytest
from PIL import Image

from tests.helpers.env import DeviceMemoryMonitor
from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm import SamplingParams
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

QUANTIZED_MODEL = os.environ.get("GLM_IMAGE_AUTOROUND_MODEL", "Intel/GLM-Image-int4-AutoRound")
BASELINE_MODEL = os.environ.get("GLM_IMAGE_BASELINE_MODEL", "zai-org/GLM-Image")

# Small resolution to keep GPU memory & time manageable
HEIGHT = 256
WIDTH = 256
NUM_STEPS = 2  # minimal for smoke-test

# GLM-Image AR generation config (from generation_config.json)
GLM_IMAGE_EOS_TOKEN_ID = 16385
GLM_IMAGE_VISION_VOCAB_SIZE = 16512

_CI_DEPLOY = get_deploy_config_path("glm_image.yaml")


def _get_stage_config():
    """Build a CI-friendly stage config with eager mode for testing."""
    return modify_stage_config(
        _CI_DEPLOY,
        updates={
            "stages": {
                0: {"enforce_eager": True},
                1: {"enforce_eager": True},
            },
        },
    )


stage_config = _get_stage_config()


def compute_max_tokens(height: int, width: int, factor: int = 32) -> int:
    """Compute max_new_tokens for GLM-Image AR text-to-image generation."""
    token_h = height // factor
    token_w = width // factor
    large_tokens = token_h * token_w

    ratio = token_h / token_w if token_w > 0 else 1.0
    small_token_h = max(1, int(math.sqrt(ratio) * (factor // 2)))
    small_token_w = max(1, int(math.sqrt(1 / ratio) * (factor // 2)))
    small_tokens = small_token_h * small_token_w

    return small_tokens + large_tokens + 1


def _build_t2i_prompt(
    prompt: str,
    height: int = HEIGHT,
    width: int = WIDTH,
) -> dict:
    """Build prompt dict for text-to-image generation."""
    return {
        "prompt": prompt,
        "height": height,
        "width": width,
        "mm_processor_kwargs": {
            "target_h": height,
            "target_w": width,
        },
    }


def _ar_sampling_params(max_tokens: int, height: int, width: int, seed: int = 42) -> SamplingParams:
    """Build AR stage SamplingParams for GLM-Image."""
    return SamplingParams(
        temperature=0.9,
        top_p=0.75,
        top_k=GLM_IMAGE_VISION_VOCAB_SIZE,
        max_tokens=max_tokens,
        stop_token_ids=[GLM_IMAGE_EOS_TOKEN_ID],
        seed=seed,
        detokenize=False,
        extra_args={
            "target_h": height,
            "target_w": width,
        },
    )


def _diffusion_sampling_params(
    height: int = HEIGHT,
    width: int = WIDTH,
    num_steps: int = NUM_STEPS,
    seed: int = 42,
) -> OmniDiffusionSamplingParams:
    """Build Diffusion stage OmniDiffusionSamplingParams."""
    return OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=0.0,
        seed=seed,
    )


def _generate_image(
    model_name: str,
    *,
    prompt_dict: dict,
    height: int = HEIGHT,
    width: int = WIDTH,
    num_steps: int = NUM_STEPS,
    stage_config_path: str | None = None,
) -> tuple[list[Image.Image], float]:
    """Load GLM-Image, generate one image, return (images, peak_memory_mb)."""
    gc.collect()
    current_omni_platform.empty_cache()
    device_index = current_omni_platform.current_device()
    current_omni_platform.reset_peak_memory_stats()
    monitor = DeviceMemoryMonitor(device_index=device_index, interval=0.02)
    monitor.start()

    gen_kwargs = {}
    if stage_config_path:
        gen_kwargs["stage_configs_path"] = stage_config_path

    with OmniRunner(model_name, seed=42, **gen_kwargs) as runner:
        current_omni_platform.reset_peak_memory_stats()

        ar_params = _ar_sampling_params(
            max_tokens=compute_max_tokens(height, width),
            height=height,
            width=width,
            seed=42,
        )
        diffusion_params = _diffusion_sampling_params(
            height=height,
            width=width,
            num_steps=num_steps,
            seed=42,
        )

        outputs = runner.omni.generate(
            [prompt_dict],
            [ar_params, diffusion_params],
            py_generator=True,
        )
        outputs = list(outputs)

    peak = monitor.peak_used_mb
    monitor.stop()

    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    req_out = first_output.request_output
    assert isinstance(req_out, OmniRequestOutput) and hasattr(req_out, "images")
    images = req_out.images

    gc.collect()
    current_omni_platform.empty_cache()

    return images, peak


# ------------------------------------------------------------------
# Test: text-to-image generation produces a valid image (quantized)
# ------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_glm_image_autoround_w4a16_generates_image():
    """Load the W4A16 quantized GLM-Image model and verify it produces a valid image."""
    prompt_dict = _build_t2i_prompt("A photo of a cat sitting on a laptop keyboard")
    images, _ = _generate_image(
        QUANTIZED_MODEL,
        prompt_dict=prompt_dict,
        stage_config_path=stage_config,
    )

    assert len(images) >= 1, "Expected at least one generated image"
    img = images[0]
    assert isinstance(img, Image.Image)
    assert img.width == WIDTH, f"Expected width {WIDTH}, got {img.width}"
    assert img.height == HEIGHT, f"Expected height {HEIGHT}, got {img.height}"

    # Sanity: image should not be blank (all-zero)
    import numpy as np

    arr = np.array(img)
    assert arr.std() > 1.0, "Generated image appears blank (std ≈ 0)"


# ------------------------------------------------------------------
# Test: image-to-image generation (quantized)
# ------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_glm_image_autoround_w4a16_image_to_image():
    """Load the W4A16 quantized GLM-Image and verify image-to-image generation works."""
    ref_image_arr = generate_synthetic_image(WIDTH, HEIGHT)["np_array"]

    gc.collect()
    current_omni_platform.empty_cache()
    device_index = current_omni_platform.current_device()
    current_omni_platform.reset_peak_memory_stats()
    monitor = DeviceMemoryMonitor(device_index=device_index, interval=0.02)
    monitor.start()

    with OmniRunner(QUANTIZED_MODEL, seed=42, stage_configs_path=stage_config) as runner:
        current_omni_platform.reset_peak_memory_stats()

        ar_params = _ar_sampling_params(
            max_tokens=compute_max_tokens(HEIGHT, WIDTH),
            height=HEIGHT,
            width=WIDTH,
            seed=42,
        )
        diffusion_params = _diffusion_sampling_params(
            height=HEIGHT,
            width=WIDTH,
            num_steps=NUM_STEPS,
            seed=42,
        )

        prompt_dict = {
            "prompt": "Make it look like winter",
            "multi_modal_data": {"image": ref_image_arr},
            "height": HEIGHT,
            "width": WIDTH,
            "mm_processor_kwargs": {
                "target_h": HEIGHT,
                "target_w": WIDTH,
            },
        }

        outputs = runner.omni.generate(
            [prompt_dict],
            [ar_params, diffusion_params],
            py_generator=True,
        )
        outputs = list(outputs)

    peak = monitor.peak_used_mb
    monitor.stop()

    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    req_out = first_output.request_output
    assert isinstance(req_out, OmniRequestOutput) and hasattr(req_out, "images")
    images = req_out.images

    assert len(images) >= 1, "Expected at least one generated image"
    img = images[0]
    assert isinstance(img, Image.Image)
    assert img.width == WIDTH
    assert img.height == HEIGHT

    gc.collect()
    current_omni_platform.empty_cache()

