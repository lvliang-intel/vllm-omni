# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for Qwen3-Omni AutoRound W4A16 quantized inference.

These tests require:
  - CUDA GPUs (2x H100-80G or equivalent)
  - The quantized model checkpoint (Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound)
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

from pathlib import Path

import pytest

from tests.conftest import (
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
baseline_params = [(BASELINE_MODEL, stage_config)]


def _get_question(prompt_type="video"):
    prompts = {
        "video": "Describe the video briefly.",
        "text_only": "What is the capital of France?",
    }
    return prompts.get(prompt_type, prompts["video"])


# ------------------------------------------------------------------
# Test: quantized model produces valid audio output from video input
# ------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_qwen3_omni_autoround_video_to_audio(omni_runner, omni_runner_handler):
    """Load the W4A16 quantized Qwen3-Omni and verify it produces audio from video input."""
    video = generate_synthetic_video(224, 224, 300)["np_array"]

    request_config = {
        "prompts": _get_question("video"),
        "videos": video,
        "modalities": ["audio"],
    }
    response = omni_runner_handler.send_request(request_config)
    assert response.success, f"Request failed: {response.error_message}"
