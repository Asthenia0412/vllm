# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deprecated shim for FlashInfer quantization utilities.

All symbols are now consolidated in :mod:`vllm.utils.flashinfer`. This module
re-exports them for backward compatibility and emits a ``DeprecationWarning``
on import. Import directly from ``vllm.utils.flashinfer`` instead.
"""

from __future__ import annotations

import warnings

from vllm.utils.flashinfer import (
    NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE,
    activation_to_flashinfer_int,
    activation_to_flashinfer_type,
    align_fp4_moe_hidden_dim_for_fi,
    align_fp4_moe_weights_for_fi,
    align_moe_weights_for_fi,
    convert_moe_weights_to_flashinfer_trtllm_block_layout,
    has_flashinfer_situ_activation,
    prepare_fp8_moe_layer_for_fi,
    quantize_nvfp4_per_token_input,
    rotate_weights_for_fi_trtllm_fp8_per_tensor_moe,
    swap_w13_to_w31,
    trtllm_nvfp4_hidden_alignment,
)

warnings.warn(
    "vllm.model_executor.layers.quantization.utils.flashinfer_utils is "
    "deprecated. All symbols have moved to vllm.utils.flashinfer. Import "
    "directly from vllm.utils.flashinfer instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE",
    "activation_to_flashinfer_int",
    "activation_to_flashinfer_type",
    "align_fp4_moe_hidden_dim_for_fi",
    "align_fp4_moe_weights_for_fi",
    "align_moe_weights_for_fi",
    "convert_moe_weights_to_flashinfer_trtllm_block_layout",
    "has_flashinfer_situ_activation",
    "prepare_fp8_moe_layer_for_fi",
    "quantize_nvfp4_per_token_input",
    "rotate_weights_for_fi_trtllm_fp8_per_tensor_moe",
    "swap_w13_to_w31",
    "trtllm_nvfp4_hidden_alignment",
]
