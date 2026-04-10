# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for Qwen2.5-Omni AutoRound W4A16 quantized inference.

These tests require:
  - CUDA GPUs (4x L4 / 24 GB or equivalent)
  - The quantized model checkpoint (Intel/Qwen2.5-Omni-7B-int4-AutoRound)
"""

import os
from pathlib import Path

import pytest

from tests.conftest import (
    generate_synthetic_audio,
    generate_synthetic_image,
    modify_stage_config,
)
from tests.utils import hardware_test

QUANTIZED_MODEL = "Intel/Qwen2.5-Omni-7B-int4-AutoRound"
BASELINE_MODEL = "Qwen/Qwen2.5-Omni-7B"

# Allow overriding via environment for local testing
QUANTIZED_MODEL = os.environ.get("QWEN2_5_OMNI_AUTOROUND_MODEL", QUANTIZED_MODEL)
BASELINE_MODEL = os.environ.get("QWEN2_5_OMNI_BASELINE_MODEL", BASELINE_MODEL)


def _get_stage_config():
    """Build a CI-friendly stage config with eager mode."""
    return modify_stage_config(
        str(Path(__file__).parent.parent / "stage_configs" / "qwen2_5_omni_ci.yaml"),
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
baseline_params = [(BASELINE_MODEL, stage_config)]


def _get_question(prompt_type="text_only"):
    prompts = {
        "mix": "What is recited in the audio? What is in this image? Describe the video briefly.",
        "text_only": "What is the capital of China?",
    }
    return prompts.get(prompt_type, prompts["text_only"])


# ------------------------------------------------------------------
# Test: quantized model produces valid text output
# ------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_qwen2_5_omni_autoround_text_output(omni_runner, omni_runner_handler):
    """Load the W4A16 quantized Qwen2.5-Omni model and verify it produces valid text."""
    request_config = {"prompts": _get_question("text_only"), "modalities": ["text"]}
    response = omni_runner_handler.send_request(request_config)
    assert response.success, f"Request failed: {response.error_message}"
    assert response.text_content, "Expected non-empty text output from quantized model"
    assert len(response.text_content.strip()) > 0, "Text output is blank"


# ------------------------------------------------------------------
# Test: quantized model produces valid audio output from mixed input
# ------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_qwen2_5_omni_autoround_audio_output(omni_runner, omni_runner_handler):
    """Load the W4A16 quantized Qwen2.5-Omni and verify it produces audio output."""
    image = generate_synthetic_image(16, 16)["np_array"]
    audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()

    request_config = {
        "prompts": _get_question("mix"),
        "images": image,
        "audios": (audio, 16000),
        "modalities": ["audio"],
    }
    response = omni_runner_handler.send_request(request_config)
    assert response.success, f"Request failed: {response.error_message}"
