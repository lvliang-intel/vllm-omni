# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extended INC/AutoRound config for multi-stage omni models.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.models.utils import WeightsMapper

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )


class OmniINCConfig(INCConfig):
    """INCConfig extended with multi-stage prefix remapping.

    For omni models the HF checkpoint names in ``block_name_to_quantize``
    and ``extra_config`` contain stage prefixes (``thinker.``, ``talker.``)
    that must be translated to the vLLM runtime prefix namespace.

    This happens transparently through :meth:`apply_vllm_mapper` which
    vLLM calls before model construction — no model-code changes needed.
    """

    # ------------------------------------------------------------------
    # Core integration: called by vLLM's configure_quant_config()
    # ------------------------------------------------------------------

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        """Remap HF checkpoint names to vLLM runtime prefixes.

        Called by vLLM's ``configure_quant_config()`` before model
        ``__init__``.  The base :class:`INCConfig` version handles
        ``block_name_to_quantize`` (via ``apply_list``) correctly, but
        ``extra_config`` keys are regex patterns — prefix-based mapping
        won't match them.  This override builds an additional
        *substring* mapper with escaped-dot keys to remap those patterns.
        """
        # Normalize CSV string before mapping
        if isinstance(self.block_name_to_quantize, str):
            self.block_name_to_quantize = [
                b.strip() for b in self.block_name_to_quantize.split(",")
                if b.strip()
            ]

        # block_name_to_quantize — base class logic works fine
        if self.block_name_to_quantize is not None:
            self.block_name_to_quantize = hf_to_vllm_mapper.apply_list(
                self.block_name_to_quantize
            )

        # extra_config — need substring replacement for regex pattern keys
        if self.extra_config is not None:
            prefix_map = (
                getattr(hf_to_vllm_mapper, "orig_to_new_prefix", None) or {}
            )
            escaped_substr_map: dict[str, str | None] = {}
            for orig, new in prefix_map.items():
                if new is None:
                    escaped_substr_map[orig.replace(".", r"\.")] = None
                else:
                    escaped_substr_map[orig.replace(".", r"\.")] = new.replace(
                        ".", r"\."
                    )

            if escaped_substr_map:
                extra_mapper = WeightsMapper(
                    orig_to_new_substr=escaped_substr_map
                )
                self.extra_config = extra_mapper.apply_dict(self.extra_config)
            else:
                # Fallback to base-class dict mapping
                self.extra_config = hf_to_vllm_mapper.apply_dict(
                    self.extra_config
                )

    # ------------------------------------------------------------------
    # Upgrading a vanilla INCConfig created by vLLM
    # ------------------------------------------------------------------

    @classmethod
    def from_inc_config(cls, inc: INCConfig) -> OmniINCConfig:
        """Promote a vanilla :class:`INCConfig` to :class:`OmniINCConfig`.

        Copies all attributes so that the new instance is a drop-in
        replacement.
        """
        omni = object.__new__(cls)
        omni.__dict__.update(inc.__dict__)
        return omni

    @classmethod
    def maybe_upgrade(cls, quant_config: QuantizationConfig | None) -> QuantizationConfig | None:
        """Upgrade *quant_config* to :class:`OmniINCConfig` if applicable.

        Returns the original config unchanged when it is not an INC
        config or is already an :class:`OmniINCConfig`.
        """
        if quant_config is None:
            return None
        if isinstance(quant_config, cls):
            return quant_config
        if isinstance(quant_config, INCConfig):
            return cls.from_inc_config(quant_config)
        return quant_config
