# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    _QWEN3_OMNI_TOKEN_ATTR_DEFAULTS,
    Qwen3OmniMoeProcessorCompat,
    _ensure_tokenizer_mm_tokens,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeTokenizer:
    """Tokenizer stub that mimics Qwen2TokenizerFast WITHOUT mm attrs."""

    def __init__(self):
        self._vocab = {
            "<|audio_pad|>": 11,
            "<|image_pad|>": 12,
            "<|video_pad|>": 13,
            "<|vision_start|>": 14,
            "<|vision_end|>": 15,
            "<|audio_start|>": 16,
            "<|audio_end|>": 17,
        }

    def get_vocab(self):
        return dict(self._vocab)


# -- _ensure_tokenizer_mm_tokens --


def test_ensure_tokenizer_mm_tokens_patches_all_seven_attrs():
    """All 7 multimodal token attributes should be set on a bare tokenizer."""
    tok = _FakeTokenizer()

    _ensure_tokenizer_mm_tokens(tok)

    for attr, expected in _QWEN3_OMNI_TOKEN_ATTR_DEFAULTS.items():
        assert getattr(tok, attr) == expected, f"{attr} mismatch"


def test_ensure_tokenizer_mm_tokens_preserves_existing():
    """Pre-existing attributes must not be overwritten."""
    tok = _FakeTokenizer()
    tok.image_token = "<|CUSTOM|>"

    _ensure_tokenizer_mm_tokens(tok)

    assert tok.image_token == "<|CUSTOM|>"
    assert tok.audio_token == "<|audio_pad|>"  # other attrs still patched


def test_ensure_tokenizer_mm_tokens_idempotent():
    """Calling twice should produce the same result."""
    tok = _FakeTokenizer()
    _ensure_tokenizer_mm_tokens(tok)
    _ensure_tokenizer_mm_tokens(tok)

    for attr, expected in _QWEN3_OMNI_TOKEN_ATTR_DEFAULTS.items():
        assert getattr(tok, attr) == expected


# -- _Qwen3OmniMoeProcessorCompat --


def test_compat_processor_patches_tokenizer_before_parent_init(monkeypatch):
    """Tokenizer should have all attrs by the time parent __init__ reads them."""
    captured_attrs = {}

    def fake_parent_init(
        self,
        image_processor=None,
        video_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        **kw,
    ):
        # Capture what the tokenizer looks like at parent __init__ time
        for attr in _QWEN3_OMNI_TOKEN_ATTR_DEFAULTS:
            captured_attrs[attr] = getattr(tokenizer, attr, "MISSING")

    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
        Qwen3OmniMoeProcessor,
    )

    monkeypatch.setattr(Qwen3OmniMoeProcessor, "__init__", fake_parent_init)

    tok = _FakeTokenizer()
    Qwen3OmniMoeProcessorCompat(tokenizer=tok)

    for attr, expected in _QWEN3_OMNI_TOKEN_ATTR_DEFAULTS.items():
        assert captured_attrs[attr] == expected, f"{attr} was {captured_attrs[attr]!r} at parent __init__ time"


def test_compat_processor_passes_all_args_through(monkeypatch):
    """All constructor args must reach the parent __init__."""
    received = {}

    def fake_parent_init(
        self,
        image_processor=None,
        video_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        **kw,
    ):
        received.update(
            dict(
                image_processor=image_processor,
                video_processor=video_processor,
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
                chat_template=chat_template,
            )
        )

    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
        Qwen3OmniMoeProcessor,
    )

    monkeypatch.setattr(Qwen3OmniMoeProcessor, "__init__", fake_parent_init)

    tok = _FakeTokenizer()
    sentinel_ip = object()
    sentinel_vp = object()
    sentinel_fe = object()

    Qwen3OmniMoeProcessorCompat(
        image_processor=sentinel_ip,
        video_processor=sentinel_vp,
        feature_extractor=sentinel_fe,
        tokenizer=tok,
        chat_template="tmpl",
    )

    assert received["image_processor"] is sentinel_ip
    assert received["video_processor"] is sentinel_vp
    assert received["feature_extractor"] is sentinel_fe
    assert received["tokenizer"] is tok
    assert received["chat_template"] == "tmpl"
