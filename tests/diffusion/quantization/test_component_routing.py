# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for component routing for quantization.

"""

import pytest
import torch
from unittest.mock import MagicMock

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.quantization.component_config import (
    ComponentQuantizationConfig,
    remap_quant_config_with_hf_mapper,
)

pytestmark = [pytest.mark.core_model]


# ---------------------------------------------------------------------------
# Helpers: lightweight mock quant configs
# ---------------------------------------------------------------------------

class _MockQuantConfig(QuantizationConfig):
    """Minimal mock that only implements get_name()."""

    def __init__(self, name: str, **attrs):
        self._name = name
        for k, v in attrs.items():
            setattr(self, k, v)

    def get_name(self) -> str:
        return self._name

    def get_quant_method(self, layer, prefix):
        return MagicMock()

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16, torch.float16]

    def get_min_capability(self):
        return 0

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError

    def get_config_filenames(self):
        return []


def _make_inc_config(block_names="thinker.model.layers,talker.model.layers",
                     extra_config=None):
    """Create a mock INC-like config with block_name_to_quantize."""
    return _MockQuantConfig(
        "inc",
        block_name_to_quantize=block_names,
        extra_config=extra_config or {},
        weight_bits=4,
        group_size=128,
        sym=True,
        packed_modules_mapping={},
    )


THINKER_MAPPER = WeightsMapper(orig_to_new_prefix={
    "thinker.lm_head.": "language_model.lm_head.",
    "thinker.model.": "language_model.model.",
    "thinker.": "",
})

TALKER_MAPPER = WeightsMapper(orig_to_new_prefix={
    "talker.codec_head.": "language_model.lm_head.",
    "talker.model.": "language_model.model.",
    "talker.thinker_to_talker_proj.": "thinker_to_talker_proj.",
    "talker.": "",
})


# ===================================================================
# 1. remap_quant_config_with_hf_mapper
# ===================================================================

class TestRemapQuant:

    def test_no_block_name_returns_same_object(self):
        """Configs without block_name_to_quantize are returned unchanged (identity)."""
        for name in ("modelopt", "fp8", "modelopt_fp4"):
            cfg = _MockQuantConfig(name)
            result = remap_quant_config_with_hf_mapper(
                cfg, hf_to_vllm_mapper=THINKER_MAPPER, stage_prefix="thinker"
            )
            assert result is cfg

    def test_inc_csv_string_normalized_to_list(self):
        """CSV string block_name_to_quantize is split into a list."""
        cfg = _make_inc_config("thinker.model.layers,talker.model.layers")
        result = remap_quant_config_with_hf_mapper(
            cfg, hf_to_vllm_mapper=THINKER_MAPPER, stage_prefix="thinker"
        )
        assert isinstance(result.block_name_to_quantize, list)

    def test_inc_deep_copied(self):
        """Remap creates a deep copy -- original config is not mutated."""
        cfg = _make_inc_config("thinker.model.layers")
        original_blocks = cfg.block_name_to_quantize
        result = remap_quant_config_with_hf_mapper(
            cfg, hf_to_vllm_mapper=THINKER_MAPPER, stage_prefix="thinker"
        )
        assert result is not cfg
        assert cfg.block_name_to_quantize == original_blocks  # original untouched

    def test_thinker_blocks_remapped(self):
        """thinker.model.layers -> thinker.language_model.model.layers after remap."""
        cfg = _make_inc_config("thinker.model.layers,talker.model.layers")
        result = remap_quant_config_with_hf_mapper(
            cfg, hf_to_vllm_mapper=THINKER_MAPPER, stage_prefix="thinker"
        )
        assert all(b.startswith("thinker.") for b in result.block_name_to_quantize)
        assert any("language_model.model.layers" in b for b in result.block_name_to_quantize)

    def test_cross_stage_blocks_filtered_out(self):
        """Blocks belonging to other stages are filtered out."""
        cfg = _make_inc_config("thinker.model.layers,talker.model.layers")
        result = remap_quant_config_with_hf_mapper(
            cfg, hf_to_vllm_mapper=THINKER_MAPPER, stage_prefix="thinker"
        )
        assert not any("talker" in b for b in result.block_name_to_quantize)

    def test_talker_remap(self):
        """Remap from talker side: only talker blocks survive."""
        cfg = _make_inc_config("thinker.model.layers,talker.model.layers")
        result = remap_quant_config_with_hf_mapper(
            cfg, hf_to_vllm_mapper=TALKER_MAPPER, stage_prefix="talker"
        )
        assert all(b.startswith("talker.") for b in result.block_name_to_quantize)
        assert not any("thinker" in b for b in result.block_name_to_quantize)

    def test_extra_config_keys_remapped(self):
        """Regex keys in extra_config get their escaped-dot prefixes remapped."""
        extra = {
            r".*thinker\.model\.layers\.0\.mlp\.gate.*": {"bits": 16, "data_type": "float"},
        }
        cfg = _make_inc_config("thinker.model.layers", extra_config=extra)
        result = remap_quant_config_with_hf_mapper(
            cfg, hf_to_vllm_mapper=THINKER_MAPPER, stage_prefix="thinker"
        )
        # The key should now reference the vLLM runtime path
        assert any("language_model" in k for k in result.extra_config)
        # Original thinker\.model prefix should be replaced
        assert not any(r"thinker\.model" in k for k in result.extra_config)

    def test_single_block_name(self):
        """Only one block name (not CSV) still works."""
        cfg = _make_inc_config("thinker.model.layers")
        result = remap_quant_config_with_hf_mapper(
            cfg, hf_to_vllm_mapper=THINKER_MAPPER, stage_prefix="thinker"
        )
        assert len(result.block_name_to_quantize) == 1
        assert result.block_name_to_quantize[0].startswith("thinker.")

    def test_already_list_block_names(self):
        """block_name_to_quantize already a list (not CSV string) works."""
        cfg = _make_inc_config(["thinker.model.layers", "talker.model.layers"])
        result = remap_quant_config_with_hf_mapper(
            cfg, hf_to_vllm_mapper=THINKER_MAPPER, stage_prefix="thinker"
        )
        assert isinstance(result.block_name_to_quantize, list)
        assert all(b.startswith("thinker.") for b in result.block_name_to_quantize)

    def test_empty_stage_prefix(self):
        """Empty stage_prefix doesn't add a dot prefix and doesn't filter."""
        cfg = _make_inc_config("thinker.model.layers")
        result = remap_quant_config_with_hf_mapper(
            cfg, hf_to_vllm_mapper=THINKER_MAPPER, stage_prefix=""
        )
        assert len(result.block_name_to_quantize) >= 1


# ===================================================================
# 2. Three-branch thinker routing (simulated)
# ===================================================================

def _simulate_thinker_routing(quant_config):
    """Simulate the three-branch routing in thinker __init__.

    Returns (visual_quant_config, language_quant_config, wrapped_vllm_quant).
    """
    _PRE_QUANTIZED_METHODS = {"modelopt", "modelopt_fp4", "modelopt_mxfp8"}

    if isinstance(quant_config, ComponentQuantizationConfig):
        visual_quant_config = quant_config.resolve("visual")
        language_quant_config = quant_config.resolve("language_model")
        return visual_quant_config, language_quant_config, quant_config
    elif quant_config is not None:
        if quant_config.get_name() in _PRE_QUANTIZED_METHODS:
            return quant_config, quant_config, quant_config
        else:
            language_quant_config = quant_config
            wrapped = ComponentQuantizationConfig(
                component_configs={"language_model": quant_config},
                default_config=None,
            )
            return None, language_quant_config, wrapped
    else:
        return None, None, None


class TestThinkerRouting:

    def test_none(self):
        vis, lang, wrapped = _simulate_thinker_routing(None)
        assert vis is None
        assert lang is None
        assert wrapped is None

    @pytest.mark.parametrize("method", ["modelopt", "modelopt_fp4", "modelopt_mxfp8"])
    def test_pre_quantized_all_components(self, method):
        """Pre-quantized methods pass config to all components."""
        cfg = _MockQuantConfig(method)
        vis, lang, wrapped = _simulate_thinker_routing(cfg)
        assert vis is cfg
        assert lang is cfg
        assert wrapped is cfg

    def test_fp8_dynamic_language_only(self):
        """fp8 dynamic: visual=None, language gets original config."""
        cfg = _MockQuantConfig("fp8")
        vis, lang, wrapped = _simulate_thinker_routing(cfg)
        assert vis is None
        assert lang is cfg
        assert isinstance(wrapped, ComponentQuantizationConfig)
        assert wrapped.resolve("language_model") is cfg
        assert wrapped.resolve("visual") is None

    def test_inc_autoround_language_only(self):
        """INC/AutoRound: not in _PRE_QUANTIZED_METHODS -> wrapped like fp8."""
        cfg = _MockQuantConfig("inc")
        vis, lang, wrapped = _simulate_thinker_routing(cfg)
        assert vis is None
        assert lang is cfg
        assert isinstance(wrapped, ComponentQuantizationConfig)

    def test_component_config_passthrough(self):
        """Explicit ComponentQuantizationConfig is used directly."""
        inner_fp8 = _MockQuantConfig("fp8")
        inner_modelopt = _MockQuantConfig("modelopt")
        cqc = ComponentQuantizationConfig(
            component_configs={
                "visual": inner_modelopt,
                "language_model": inner_fp8,
            }
        )
        vis, lang, wrapped = _simulate_thinker_routing(cqc)
        assert vis is inner_modelopt
        assert lang is inner_fp8
        assert wrapped is cqc


# ===================================================================
# 3. Talker visual routing (init_multi_modal guard)
# ===================================================================

def _simulate_talker_visual_routing(quant_config):
    """Simulate the talker init_multi_modal visual routing."""
    _PRE_QUANTIZED_METHODS = {"modelopt", "modelopt_fp4", "modelopt_mxfp8"}
    if quant_config is not None and quant_config.get_name() in _PRE_QUANTIZED_METHODS:
        return quant_config
    return None


class TestTalkerVisualRouting:

    def test_none(self):
        assert _simulate_talker_visual_routing(None) is None

    @pytest.mark.parametrize("method", ["modelopt", "modelopt_fp4", "modelopt_mxfp8"])
    def test_pre_quantized_passes_through(self, method):
        """Pre-quantized methods pass quant config to visual."""
        cfg = _MockQuantConfig(method)
        assert _simulate_talker_visual_routing(cfg) is cfg

    def test_fp8_blocked(self):
        """fp8 dynamic must NOT be passed to visual."""
        cfg = _MockQuantConfig("fp8")
        assert _simulate_talker_visual_routing(cfg) is None

    def test_inc_blocked(self):
        """INC/AutoRound must NOT be passed to visual (not in _PRE_QUANTIZED_METHODS)."""
        cfg = _MockQuantConfig("inc")
        assert _simulate_talker_visual_routing(cfg) is None


# ===================================================================
# 4. ComponentQuantizationConfig.resolve
# ===================================================================

class TestComponentResolve:

    def test_longest_prefix_match(self):
        a = _MockQuantConfig("a")
        b = _MockQuantConfig("b")
        cqc = ComponentQuantizationConfig(
            component_configs={"language_model": a, "language_model.model": b}
        )
        assert cqc.resolve("language_model.model.layers.0") is b
        assert cqc.resolve("language_model.lm_head") is a

    def test_no_match_returns_default(self):
        a = _MockQuantConfig("a")
        default = _MockQuantConfig("default")
        cqc = ComponentQuantizationConfig(
            component_configs={"language_model": a},
            default_config=default,
        )
        assert cqc.resolve("visual") is default

    def test_no_match_no_default_returns_none(self):
        a = _MockQuantConfig("a")
        cqc = ComponentQuantizationConfig(
            component_configs={"language_model": a},
        )
        assert cqc.resolve("visual") is None

    def test_get_name(self):
        cqc = ComponentQuantizationConfig(component_configs={})
        assert cqc.get_name() == "component"

    def test_get_quant_method_delegates(self):
        """get_quant_method dispatches to the resolved config."""
        inner = _MockQuantConfig("fp8")
        cqc = ComponentQuantizationConfig(
            component_configs={"language_model": inner},
        )
        layer = MagicMock()
        result = cqc.get_quant_method(layer, "language_model.model.layers.0.mlp")
        assert result is not None  # delegates to inner.get_quant_method

    def test_get_quant_method_returns_none_for_unmatched(self):
        """get_quant_method returns None when no config matches."""
        inner = _MockQuantConfig("fp8")
        cqc = ComponentQuantizationConfig(
            component_configs={"language_model": inner},
        )
        layer = MagicMock()
        result = cqc.get_quant_method(layer, "visual.blocks.0.mlp")
        assert result is None

    def test_min_capability(self):
        a = _MockQuantConfig("a")
        a.get_min_capability = lambda: 80
        b = _MockQuantConfig("b")
        b.get_min_capability = lambda: 70
        cqc = ComponentQuantizationConfig(component_configs={"x": a, "y": b})
        assert cqc.get_min_capability() == 70

    def test_min_capability_empty(self):
        cqc = ComponentQuantizationConfig(component_configs={})
        assert cqc.get_min_capability() == 0
