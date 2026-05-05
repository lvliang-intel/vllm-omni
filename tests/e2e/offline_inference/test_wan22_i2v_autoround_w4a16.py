# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for Wan2.2-I2V-A14B AutoRound W4A16 quantized inference.

These tests require:
  - A CUDA GPU with sufficient memory (~36 GiB for quantized model)
  - The quantized model checkpoint (Intel/Wan2.2-I2V-A14B-Diffusers-int4-AutoRound)
"""

import gc
import os
import os as _os

import numpy as np
import pytest
import torch
from PIL import Image
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from tests.helpers.env import DeviceMemoryMonitor
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

QUANTIZED_MODEL = "Intel/Wan2.2-I2V-A14B-Diffusers-int4-AutoRound"
BASELINE_MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

# Allow overriding via environment for local testing
QUANTIZED_MODEL = _os.environ.get("WAN22_I2V_AUTOROUND_MODEL", QUANTIZED_MODEL)
BASELINE_MODEL = _os.environ.get("WAN22_I2V_BASELINE_MODEL", BASELINE_MODEL)

# Small resolution to keep GPU memory & time manageable
HEIGHT = 480
WIDTH = 640
NUM_FRAMES = 5  # must satisfy num_frames % 4 == 1 for Wan2.2
NUM_STEPS = 2  # minimal for smoke-test


def _create_test_image(width: int = WIDTH, height: int = HEIGHT) -> Image.Image:
    """Create a deterministic test image for I2V tests."""
    rng = np.random.RandomState(42)
    arr = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _generate_video(model_name: str, **extra_kwargs) -> tuple[object, float]:
    """Load a Wan2.2 I2V model, generate one video, return (frames, peak_memory_mb)."""
    gc.collect()
    current_omni_platform.empty_cache()
    device_index = current_omni_platform.current_device()
    current_omni_platform.reset_peak_memory_stats()
    monitor = DeviceMemoryMonitor(device_index=device_index, interval=0.02)
    monitor.start()

    image = _create_test_image()

    with OmniRunner(
        model_name,
        enforce_eager=True,
        boundary_ratio=0.875,
        flow_shift=12.0,
        **extra_kwargs,
    ) as runner:
        current_omni_platform.reset_peak_memory_stats()
        outputs = runner.omni.generate(
            {
                "prompt": "A cat sitting on a table, smooth motion",
                "multi_modal_data": {"image": image},
            },
            sampling_params_list=OmniDiffusionSamplingParams(
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=NUM_STEPS,
                guidance_scale=5.0,
                guidance_scale_2=6.0,
                boundary_ratio=0.875,
                generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(42),
            ),
        )

    peak = monitor.peak_used_mb
    monitor.stop()

    first_output = outputs[0]
    assert first_output.final_output_type == "image"

    req_out = first_output.request_output
    if isinstance(req_out, list):
        req_out = req_out[0]
    assert isinstance(req_out, OmniRequestOutput) and hasattr(req_out, "images")
    frames = req_out.images[0]

    gc.collect()
    current_omni_platform.empty_cache()

    return frames, peak


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
def test_wan22_i2v_autoround_w4a16_generates_video():
    """Load the W4A16 quantized Wan2.2 I2V model and verify it produces a valid video."""
    frames, _ = _generate_video(QUANTIZED_MODEL)

    assert frames is not None, "Expected video frames output"
    assert hasattr(frames, "shape"), "Expected frames to have a shape attribute"

    # frames shape: (batch, num_frames, height, width, channels)
    assert frames.shape[1] == NUM_FRAMES, f"Expected {NUM_FRAMES} frames, got {frames.shape[1]}"
    assert frames.shape[2] == HEIGHT, f"Expected height {HEIGHT}, got {frames.shape[2]}"
    assert frames.shape[3] == WIDTH, f"Expected width {WIDTH}, got {frames.shape[3]}"

    # Sanity: video should not be blank (frames are [0, 1] floats)
    arr = np.asarray(frames)
    assert arr.std() > 0.01, "Generated video appears blank (std ≈ 0)"


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
def test_wan22_i2v_autoround_w4a16_memory_savings():
    """Compare peak GPU memory of quantized vs BF16 baseline.

    The W4A16 model should use meaningfully less memory than the
    BF16 baseline since weights are 4-bit instead of 16-bit.
    """
    _, quant_peak = _generate_video(QUANTIZED_MODEL)
    cleanup_dist_env_and_memory()
    _, baseline_peak = _generate_video(BASELINE_MODEL)

    print(f"Quantized (W4A16) peak memory: {quant_peak:.0f} MB")
    print(f"Baseline (BF16) peak memory:   {baseline_peak:.0f} MB")
    print(f"Savings:                        {baseline_peak - quant_peak:.0f} MB")

    # Wan2.2 I2V A14B transformer is ~28 GB in BF16; W4A16 should save ~20 GB.
    # Use a conservative threshold to account for activations and overhead.
    min_savings_mb = 5000
    assert quant_peak + min_savings_mb < baseline_peak, (
        f"Quantized model ({quant_peak:.0f} MB) should use at least "
        f"{min_savings_mb} MB less than baseline ({baseline_peak:.0f} MB)"
    )
