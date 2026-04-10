# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for Qwen3-Omni AutoRound W4A16 quantized inference.

These tests cover text, audio, image, video, and mixed-modality inputs
to verify multimodal understanding with quantized weights.

Requirements:
  - CUDA GPUs (2x H100-80G or equivalent)
  - The quantized model checkpoint (Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound)
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

from pathlib import Path

import pytest

from tests.conftest import (
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
    modify_stage_config,
)
from tests.utils import hardware_test

QUANTIZED_MODEL = "Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound"
BASELINE_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# Allow overriding via environment for local testing
QUANTIZED_MODEL = os.environ.get("QWEN3_OMNI_AUTOROUND_MODEL", QUANTIZED_MODEL)
BASELINE_MODEL = os.environ.get("QWEN3_OMNI_BASELINE_MODEL", BASELINE_MODEL)


def _get_stage_config():
    """Build a CI-friendly stage config with eager mode."""
    return modify_stage_config(
        str(Path(__file__).parent.parent / "stage_configs" / "qwen3_omni_ci.yaml"),
        updates={
            "stage_args": {
                0: {"engine_args.enforce_eager": "true"},
                1: {"engine_args.enforce_eager": "true"},
            },
        },
    )


stage_config = _get_stage_config()

# Parametrise: (model, stage_config)
quant_params = [(QUANTIZED_MODEL, stage_config)]


# ------------------------------------------------------------------
# Test: text-only input → text output
# ------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_text_to_text(omni_runner, omni_runner_handler):
    """Text input → text output with W4A16 quantized Qwen3-Omni."""
    request_config = {
        "prompts": "What is the capital of France?",
        "modalities": ["text"],
    }
    response = omni_runner_handler.send_request(request_config)
    assert response.success, f"Request failed: {response.error_message}"
    assert response.text_content and len(response.text_content.strip()) > 0


# ------------------------------------------------------------------
# Test: audio input → text output
# ------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_audio_to_text(omni_runner, omni_runner_handler):
    """Audio input → text output with W4A16 quantized Qwen3-Omni."""
    audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()

    request_config = {
        "prompts": "What is the content of this audio?",
        "audios": (audio, 16000),
        "modalities": ["text"],
    }
    response = omni_runner_handler.send_request(request_config)
    assert response.success, f"Request failed: {response.error_message}"
    assert response.text_content and len(response.text_content.strip()) > 0


# ------------------------------------------------------------------
# Test: image input → text output
# ------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_image_to_text(omni_runner, omni_runner_handler):
    """Image input → text output with W4A16 quantized Qwen3-Omni."""
    image = generate_synthetic_image(16, 16)["np_array"]

    request_config = {
        "prompts": "Describe what you see in this image.",
        "images": image,
        "modalities": ["text"],
    }
    response = omni_runner_handler.send_request(request_config)
    assert response.success, f"Request failed: {response.error_message}"
    assert response.text_content and len(response.text_content.strip()) > 0


# ------------------------------------------------------------------
# Test: video input → text output
# ------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_video_to_text(omni_runner, omni_runner_handler):
    """Video input → text output with W4A16 quantized Qwen3-Omni."""
    video = generate_synthetic_video(224, 224, 300)["np_array"]

    request_config = {
        "prompts": "Describe the video briefly.",
        "videos": video,
        "modalities": ["text"],
    }
    response = omni_runner_handler.send_request(request_config)
    assert response.success, f"Request failed: {response.error_message}"
    assert response.text_content and len(response.text_content.strip()) > 0


# ------------------------------------------------------------------
# Test: video input → audio output
# ------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_video_to_audio(omni_runner, omni_runner_handler):
    """Video input → audio output with W4A16 quantized Qwen3-Omni."""
    video = generate_synthetic_video(224, 224, 300)["np_array"]

    request_config = {
        "prompts": "Describe the video briefly.",
        "videos": video,
        "modalities": ["audio"],
    }
    response = omni_runner_handler.send_request(request_config)
    assert response.success, f"Request failed: {response.error_message}"


# ------------------------------------------------------------------
# Test: mixed modality (audio + image + video) → audio output
# ------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_mix_to_audio(omni_runner, omni_runner_handler):
    """Mixed-modality input → audio output with W4A16 quantized Qwen3-Omni."""
    video = generate_synthetic_video(224, 224, 300)["np_array"]
    image = generate_synthetic_image(16, 16)["np_array"]
    audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()

    request_config = {
        "prompts": "What is recited in the audio? What is in this image? Describe the video briefly.",
        "videos": video,
        "images": image,
        "audios": (audio, 16000),
        "modalities": ["audio"],
    }
    response = omni_runner_handler.send_request(request_config)
    assert response.success, f"Request failed: {response.error_message}"
