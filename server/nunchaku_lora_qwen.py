"""
LoRA loader for NunchakuQwenImageTransformer2DModel.

Ported from ussoewwin/ComfyUI-QwenImageLoraLoader (nunchaku_code/lora_qwen.py).
Stripped of ComfyUI dependencies, Z-Image/NextDiT/LoKR support, and AWQ modulation handling.
Focuses on standard LoRA (rank-decomposed) applied to nunchaku quantized Qwen Image models.

Requires: nunchaku (for pack_lowrank_weight / unpack_lowrank_weight)

Usage:
    from lora_qwen import apply_lora, reset_lora
    apply_lora(transformer, "path/to/lora.safetensors", strength=1.0)
    reset_lora(transformer)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import warnings

import torch
import torch.nn as nn
from safetensors import safe_open

# Suppress PyTorch deprecation warning from nunchaku's pack/unpack_lowrank_weight
# (non-tuple sequence indexing — harmless until PyTorch 2.9)
warnings.filterwarnings("ignore", message="Using a non-tuple sequence for multidimensional indexing")

from nunchaku.lora.flux.nunchaku_converter import (
    pack_lowrank_weight,
    unpack_lowrank_weight,
)

logger = logging.getLogger(__name__)

# =======================
# Key classification
# =======================

# Regex for LoRA suffix extraction
_RE_LORA_SUFFIX = re.compile(r"\.(?P<tag>lora(?:[._](?:A|B|down|up)))(?:\.[^.]+)*\.weight$")
_RE_ALPHA_SUFFIX = re.compile(r"\.(?:alpha|lora_alpha)(?:\.[^.]+)*$")

# Centralized key mapping for Qwen Image architecture
KEY_MAPPING = [
    # Fused QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]qkv$"),
     r"\1.\2.attn.to_qkv", "qkv", None),
    # Decomposed QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._](q|k|v)$"),
     r"\1.\2.attn.to_qkv", "qkv", lambda m: m.group(3).upper()),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._](q|k|v)[._]proj$"),
     r"\1.\2.attn.to_qkv", "qkv", lambda m: m.group(3).upper()),
    # Fused Add_QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]add[._]qkv[._]proj$"),
     r"\1.\2.attn.add_qkv_proj", "add_qkv", None),
    # Decomposed Add_QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]add[._](q|k|v)[._]proj$"),
     r"\1.\2.attn.add_qkv_proj", "add_qkv", lambda m: m.group(3).upper()),
    # Fused QKV (Single Block)
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]attn[._]to[._]qkv$"),
     r"\1.\2.attn.to_qkv", "qkv", None),
    # Decomposed QKV (Single Block)
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]attn[._]to[._](q|k|v)$"),
     r"\1.\2.attn.to_qkv", "qkv", lambda m: m.group(3).upper()),
    # Output Projections
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]out[._]proj[._]context$"),
     r"\1.\2.attn.to_add_out", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]add[._]out$"),
     r"\1.\2.attn.to_add_out", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]out[._]proj$"),
     r"\1.\2.attn.to_out.0", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]out$"),
     r"\1.\2.attn.to_out.0", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]out[._]0$"),
     r"\1.\2.attn.to_out.0", "regular", None),
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]attn[._]to[._]out$"),
     r"\1.\2.attn.to_out", "regular", None),
    # Feed-Forward / MLP (Standard)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff[._]net[._]0(?:[._]proj)?$"),
     r"\1.\2.mlp_fc1", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff[._]net[._]2$"),
     r"\1.\2.mlp_fc2", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff_context[._]net[._]0(?:[._]proj)?$"),
     r"\1.\2.mlp_context_fc1", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff_context[._]net[._]2$"),
     r"\1.\2.mlp_context_fc2", "regular", None),
    # Feed-Forward / MLP (img/txt)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mlp)[._](net)[._](0)[._](proj)$"),
     r"\1.\2.\3.\4.\5.\6", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mlp)[._](net)[._](0)$"),
     r"\1.\2.\3.\4.\5", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mlp)[._](net)[._](2)$"),
     r"\1.\2.\3.\4.\5", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mlp)[._](net)[._](0)[._](proj)$"),
     r"\1.\2.\3.\4.\5.\6", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mlp)[._](net)[._](0)$"),
     r"\1.\2.\3.\4.\5", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mlp)[._](net)[._](2)$"),
     r"\1.\2.\3.\4.\5", "regular", None),
    # Mod Layers (img/txt) - sensitive with AWQ, but still map them
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mod)[._](1)$"),
     r"\1.\2.\3.\4", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mod)[._](1)$"),
     r"\1.\2.\3.\4", "regular", None),
    # Single Block Projections
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]proj[._]out$"),
     r"\1.\2.proj_out", "single_proj_out", None),
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]proj[._]mlp$"),
     r"\1.\2.mlp_fc1", "regular", None),
    # Normalization Layers
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]norm[._]linear$"),
     r"\1.\2.norm.linear", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]norm1[._]linear$"),
     r"\1.\2.norm1.linear", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]norm1_context[._]linear$"),
     r"\1.\2.norm1_context.linear", "regular", None),
    # Top-level modules
    (re.compile(r"^(img_in)$"), r"\1", "regular", None),
    (re.compile(r"^(txt_in)$"), r"\1", "regular", None),
    (re.compile(r"^(proj_out)$"), r"\1", "regular", None),
    (re.compile(r"^(norm_out)[._](linear)$"), r"\1.\2", "regular", None),
    (re.compile(r"^(time_text_embed)[._](timestep_embedder)[._](linear_1)$"),
     r"\1.\2.\3", "regular", None),
    (re.compile(r"^(time_text_embed)[._](timestep_embedder)[._](linear_2)$"),
     r"\1.\2.\3", "regular", None),
]


def _rename_underscore_key(old_name: str) -> str:
    """Convert Kohya underscore key names (lora_unet_...) to dot notation."""
    rules = [
        (r'_(\d+)_attn_to_out_(\d+)', r'.\1.attn.to_out.\2'),
        (r'_(\d+)_img_mlp_net_(\d+)_proj', r'.\1.img_mlp.net.\2.proj'),
        (r'_(\d+)_txt_mlp_net_(\d+)_proj', r'.\1.txt_mlp.net.\2.proj'),
        (r'_(\d+)_img_mlp_net_(\d+)', r'.\1.img_mlp.net.\2'),
        (r'_(\d+)_txt_mlp_net_(\d+)', r'.\1.txt_mlp.net.\2'),
        (r'_(\d+)_img_mod_(\d+)', r'.\1.img_mod.\2'),
        (r'_(\d+)_txt_mod_(\d+)', r'.\1.txt_mod.\2'),
        (r'_(\d+)_attn_', r'.\1.attn.'),
    ]
    name = old_name
    for pattern, replacement in rules:
        name = re.sub(pattern, replacement, name)
    return name


def _classify_and_map_key(key: str) -> Optional[Tuple[str, str, Optional[str], str]]:
    """Classify a LoRA key using KEY_MAPPING. Returns (group, base_key, component, ab) or None."""
    k = key
    for prefix in ("transformer.", "diffusion_model."):
        if k.startswith(prefix):
            k = k[len(prefix):]
    if k.startswith("lora_unet_"):
        k = k[len("lora_unet_"):]
        k = _rename_underscore_key(k)

    base = None
    ab = None

    m = _RE_LORA_SUFFIX.search(k)
    if m:
        tag = m.group("tag")
        base = k[:m.start()]
        if "lora_A" in tag or tag.endswith(".A") or "down" in tag:
            ab = "A"
        elif "lora_B" in tag or tag.endswith(".B") or "up" in tag:
            ab = "B"
    else:
        m = _RE_ALPHA_SUFFIX.search(k)
        if m:
            ab = "alpha"
            base = k[:m.start()]

    if base is None or ab is None:
        return None

    for pattern, template, group, comp_fn in KEY_MAPPING:
        match = pattern.match(base)
        if match:
            final_key = match.expand(template)
            component = comp_fn(match) if comp_fn else None
            return group, final_key, component, ab

    return None


# =======================
# Module helpers
# =======================

def _get_module(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Traverse dotted path to find a module."""
    if not name:
        return model
    module = model
    for part in name.split("."):
        if not part:
            continue
        if hasattr(module, part):
            module = getattr(module, part)
        elif part.isdigit() and isinstance(module, (nn.ModuleList, nn.Sequential)):
            try:
                module = module[int(part)]
            except (IndexError, TypeError):
                return None
        else:
            return None
    return module


def _resolve_module(model: nn.Module, name: str) -> Tuple[str, Optional[nn.Module]]:
    """Resolve name to module, with fallback paths."""
    m = _get_module(model, name)
    if m is not None:
        return name, m

    # to_out.0 <-> to_out fallback
    if name.endswith(".attn.to_out.0"):
        alt = name[:-2]
        m = _get_module(model, alt)
        if m is not None:
            return alt, m
    elif name.endswith(".attn.to_out"):
        alt = name + ".0"
        m = _get_module(model, alt)
        if m is not None:
            return alt, m

    # ff.net <-> mlp fallback
    mapping = {
        ".ff.net.0.proj": ".mlp_fc1", ".ff.net.2": ".mlp_fc2",
        ".ff_context.net.0.proj": ".mlp_context_fc1", ".ff_context.net.2": ".mlp_context_fc2",
    }
    for src, dst in mapping.items():
        if src in name:
            alt = name.replace(src, dst)
            m = _get_module(model, alt)
            if m is not None:
                return alt, m

    return name, None


# =======================
# LoRA state dict loading
# =======================

def _load_state_dict(path: Union[str, Path, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if isinstance(path, dict):
        return path
    p = Path(path)
    if p.suffix == ".safetensors":
        sd = {}
        with safe_open(p, framework="pt", device="cpu") as f:
            for key in f.keys():
                sd[key] = f.get_tensor(key)
        return sd
    return torch.load(p, map_location="cpu", weights_only=True)


# =======================
# QKV / GLU fusion
# =======================

def _fuse_qkv(qkv_weights: Dict[str, torch.Tensor],
              model: Optional[nn.Module] = None,
              base_key: Optional[str] = None
              ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fuse separate Q/K/V LoRA weights into a single QKV tensor."""
    required = ["Q_A", "Q_B", "K_A", "K_B", "V_A", "V_B"]
    if not all(k in qkv_weights for k in required):
        return None, None, None

    A_q, A_k, A_v = qkv_weights["Q_A"], qkv_weights["K_A"], qkv_weights["V_A"]
    B_q, B_k, B_v = qkv_weights["Q_B"], qkv_weights["K_B"], qkv_weights["V_B"]
    alpha_q = qkv_weights.get("Q_alpha")
    alpha_k = qkv_weights.get("K_alpha")
    alpha_v = qkv_weights.get("V_alpha")

    if not (A_q.shape == A_k.shape == A_v.shape):
        logger.warning(f"Q/K/V A shape mismatch: {A_q.shape}, {A_k.shape}, {A_v.shape}")
        return None, None, None
    if not (B_q.shape[1] == B_k.shape[1] == B_v.shape[1]):
        logger.warning(f"Q/K/V B rank mismatch: {B_q.shape[1]}, {B_k.shape[1]}, {B_v.shape[1]}")
        return None, None, None

    alpha_fused = None
    if (alpha_q is not None and alpha_k is not None and alpha_v is not None
            and alpha_q.item() == alpha_k.item() == alpha_v.item()):
        alpha_fused = alpha_q

    # A: concatenate along rank dimension -> [3*rank, in_features]
    A_fused = torch.cat([A_q, A_k, A_v], dim=0)

    # B: block-diagonal -> [out_q+out_k+out_v, 3*rank]
    r = B_q.shape[1]
    out_q, out_k, out_v = B_q.shape[0], B_k.shape[0], B_v.shape[0]

    # Use actual module out_features if available
    total_out = out_q + out_k + out_v
    if model is not None and base_key is not None:
        _, mod = _resolve_module(model, base_key)
        if mod is not None and hasattr(mod, "out_features"):
            total_out = mod.out_features

    B_fused = torch.zeros(total_out, 3 * r, dtype=B_q.dtype, device=B_q.device)
    B_fused[:out_q, :r] = B_q
    B_fused[out_q:out_q + out_k, r:2 * r] = B_k
    B_fused[out_q + out_k:out_q + out_k + out_v, 2 * r:] = B_v

    return A_fused, B_fused, alpha_fused


def _fuse_glu(glu_weights: Dict[str, torch.Tensor]
              ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fuse gate/up (w1/w3) LoRA weights for SwiGLU layers."""
    if "w1_A" not in glu_weights or "w3_A" not in glu_weights:
        return None, None, None

    A_w1, B_w1 = glu_weights["w1_A"], glu_weights["w1_B"]
    A_w3, B_w3 = glu_weights["w3_A"], glu_weights["w3_B"]
    alpha_w1 = glu_weights.get("w1_alpha")

    if A_w1.shape[0] != A_w3.shape[0]:
        logger.warning(f"GLU A in_features mismatch: {A_w1.shape} vs {A_w3.shape}")
        return None, None, None

    r1, r3 = B_w1.shape[1], B_w3.shape[1]
    out1, out3 = B_w1.shape[0], B_w3.shape[0]

    A_fused = torch.cat([A_w1, A_w3], dim=0)
    B_fused = torch.zeros(out1 + out3, r1 + r3, dtype=B_w1.dtype, device=B_w1.device)
    B_fused[:out1, :r1] = B_w1
    B_fused[out1:, r1:] = B_w3

    return A_fused, B_fused, alpha_w1


# =======================
# Single-block proj_out split
# =======================

def _handle_proj_out_split(lora_dict: Dict, base_key: str, model: nn.Module
                           ) -> Tuple[Dict[str, Tuple], List[str]]:
    """Split single-block proj_out LoRA into attn.to_out and mlp_fc2."""
    result, consumed = {}, []
    m = re.search(r"single_transformer_blocks\.(\d+)", base_key)
    if not m or base_key not in lora_dict:
        return result, consumed

    block_idx = m.group(1)
    block = _get_module(model, f"single_transformer_blocks.{block_idx}")
    if block is None:
        return result, consumed

    A_full = lora_dict[base_key].get("A")
    B_full = lora_dict[base_key].get("B")
    alpha = lora_dict[base_key].get("alpha")
    if A_full is None or B_full is None:
        return result, consumed

    attn_to_out = getattr(getattr(block, "attn", None), "to_out", None)
    mlp_fc2 = getattr(block, "mlp_fc2", None)
    if (attn_to_out is None or mlp_fc2 is None
            or not hasattr(attn_to_out, "in_features")
            or not hasattr(mlp_fc2, "in_features")):
        return result, consumed

    attn_in = attn_to_out.in_features
    mlp_in = mlp_fc2.in_features
    if A_full.shape[1] != attn_in + mlp_in:
        logger.warning(f"{base_key}: A shape {A_full.shape} vs expected {attn_in + mlp_in}")
        return result, consumed

    A_attn = A_full[:, :attn_in]
    A_mlp = A_full[:, attn_in:]
    result[f"single_transformer_blocks.{block_idx}.attn.to_out"] = (A_attn, B_full.clone(), alpha)
    result[f"single_transformer_blocks.{block_idx}.mlp_fc2"] = (A_mlp, B_full.clone(), alpha)
    consumed.append(base_key)
    return result, consumed


# =======================
# Apply LoRA to module
# =======================

def _apply_to_module(module: nn.Module, A: torch.Tensor, B: torch.Tensor,
                     module_name: str, model: nn.Module) -> None:
    """Apply combined LoRA weights to a single module."""
    if module is None:
        raise ValueError(f"{module_name}: module is None")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"{module_name}: A/B must be 2D, got {A.shape}, {B.shape}")
    if not hasattr(module, "in_features"):
        raise ValueError(f"{module_name}: no in_features attribute")
    if A.shape[1] != module.in_features:
        raise ValueError(f"{module_name}: A {A.shape} vs in_features={module.in_features}")
    if B.shape[0] != module.out_features:
        raise ValueError(f"{module_name}: B {B.shape} vs out_features={module.out_features}")

    # Nunchaku quantized modules (proj_down / proj_up)
    if hasattr(module, "proj_down") and hasattr(module, "proj_up"):
        pd = unpack_lowrank_weight(module.proj_down.data, down=True)
        pu = unpack_lowrank_weight(module.proj_up.data, down=False)

        base_rank = pd.shape[0] if pd.shape[1] == module.in_features else pd.shape[1]

        if pd.shape[1] == module.in_features:  # [rank, in]
            new_pd = torch.cat([pd, A], dim=0)
            axis_down = 0
        else:  # [in, rank]
            new_pd = torch.cat([pd, A.T], dim=1)
            axis_down = 1

        new_pu = torch.cat([pu, B], dim=1)

        module.proj_down.data = pack_lowrank_weight(new_pd, down=True)
        module.proj_up.data = pack_lowrank_weight(new_pu, down=False)
        module.rank = base_rank + A.shape[0]

        if not hasattr(model, "_lora_slots"):
            model._lora_slots = {}
        slot = model._lora_slots.setdefault(
            module_name, {"base_rank": base_rank, "appended": 0,
                          "axis_down": axis_down, "type": "nunchaku"})
        slot["appended"] += A.shape[0]

    # Standard nn.Linear fallback
    elif isinstance(module, nn.Linear):
        if not hasattr(model, "_lora_slots"):
            model._lora_slots = {}
        if module_name not in model._lora_slots:
            model._lora_slots[module_name] = {
                "type": "linear",
                "original_weight": module.weight.detach().cpu().clone(),
            }
        delta = B @ A
        if delta.shape != module.weight.shape:
            raise ValueError(f"{module_name}: delta {delta.shape} vs weight {module.weight.shape}")
        module.weight.data.add_(delta.to(dtype=module.weight.dtype, device=module.weight.device))

    else:
        raise ValueError(f"{module_name}: unsupported module type {type(module)}")


# =======================
# Public API
# =======================

def apply_lora(
    model: nn.Module,
    lora_path: Union[str, Path, Dict[str, torch.Tensor]],
    strength: float = 1.0,
) -> int:
    """Apply a single LoRA to a NunchakuQwenImageTransformer2DModel.

    Args:
        model: The transformer model.
        lora_path: Path to .safetensors/.pt file, or pre-loaded state dict.
        strength: LoRA strength multiplier.

    Returns:
        Number of modules that had LoRA applied.
    """
    return compose_loras(model, [(lora_path, strength)])


def compose_loras(
    model: nn.Module,
    lora_configs: List[Tuple[Union[str, Path, Dict[str, torch.Tensor]], float]],
) -> int:
    """Reset and compose multiple LoRAs into the model.

    Args:
        model: The transformer model.
        lora_configs: List of (path_or_dict, strength) tuples.

    Returns:
        Number of modules that had LoRA applied.
    """
    reset_lora(model)

    aggregated: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for idx, (lora_src, strength) in enumerate(lora_configs):
        if abs(strength) < 1e-5:
            continue

        sd = _load_state_dict(lora_src)
        if not sd:
            logger.warning(f"LoRA {idx}: empty state dict, skipping")
            continue

        # Check format: only standard LoRA supported
        has_standard = any(
            p in k for k in sd
            for p in (".lora_up.", ".lora_down.", ".lora_A.", ".lora_B.",
                      ".lora.up.", ".lora.down.", ".lora.A.", ".lora.B.")
        )
        if not has_standard:
            logger.warning(f"LoRA {idx}: no standard LoRA keys found, skipping "
                           "(LoKR/LoHa/IA3 not supported with nunchaku)")
            continue

        # Group keys by base module
        grouped: Dict[str, Dict] = defaultdict(dict)
        for key, value in sd.items():
            parsed = _classify_and_map_key(key)
            if parsed:
                group, base_key, comp, ab = parsed
                if comp and ab:
                    grouped[base_key][f"{comp}_{ab}"] = value
                elif ab:
                    grouped[base_key][ab] = value

        # Process groups -> (A, B, alpha)
        processed: Dict[str, Tuple] = {}
        special_handled: set = set()

        for base_key, lw in grouped.items():
            if base_key in special_handled:
                continue

            A, B, alpha = None, None, None

            if "qkv" in base_key:
                if "A" in lw:
                    A, B, alpha = lw.get("A"), lw.get("B"), lw.get("alpha")
                else:
                    A, B, alpha = _fuse_qkv(lw, model=model, base_key=base_key)
            elif "w1_A" in lw or "w3_A" in lw:
                A, B, alpha = _fuse_glu(lw)
            elif ".proj_out" in base_key and "single_transformer_blocks" in base_key:
                split_map, consumed = _handle_proj_out_split(grouped, base_key, model)
                processed.update(split_map)
                special_handled.update(consumed)
                continue
            else:
                A, B, alpha = lw.get("A"), lw.get("B"), lw.get("alpha")

            if A is not None and B is not None:
                processed[base_key] = (A, B, alpha)

        for module_key, (A, B, alpha) in processed.items():
            aggregated[module_key].append({
                "A": A, "B": B, "alpha": alpha, "strength": strength,
                "lora_idx": idx,
            })

    # Apply aggregated weights
    applied = 0
    for module_key, weight_list in aggregated.items():
        resolved_name, module = _resolve_module(model, module_key)
        if module is None:
            logger.warning(f"[MISS] Module not found: {module_key} (resolved: {resolved_name})")
            continue

        # Skip AWQ modulation layers (img_mod/txt_mod) - too sensitive
        is_awq = module.__class__.__name__ == "AWQW4A16Linear" and hasattr(module, "qweight")
        is_mod = "img_mod" in resolved_name or "txt_mod" in resolved_name
        if is_awq and is_mod:
            logger.warning(f"[SKIP] AWQ modulation layer: {resolved_name} (sensitive to LoRA)")
            continue

        # Determine target dtype/device
        if hasattr(module, "proj_down"):
            target_dtype = module.proj_down.dtype
            target_device = module.proj_down.device
        elif hasattr(module, "weight"):
            target_dtype = module.weight.dtype
            target_device = module.weight.device
        else:
            target_dtype = torch.float16
            target_device = torch.device("cuda")

        all_A, all_B = [], []
        per_lora_ranks = []  # [(lora_idx, rank), ...]
        for w in weight_list:
            A, B, alpha = w["A"], w["B"], w["alpha"]
            rank = A.shape[0]
            if B.shape[1] != rank:
                continue
            scale = w["strength"]
            if alpha is not None:
                scale *= (alpha / rank)
            all_A.append(A.to(dtype=target_dtype, device=target_device))
            all_B.append((B * scale).to(dtype=target_dtype, device=target_device))
            per_lora_ranks.append((w.get("lora_idx", 0), rank))

        if not all_A:
            continue

        final_A = torch.cat(all_A, dim=0)
        final_B = torch.cat(all_B, dim=1)

        try:
            _apply_to_module(module, final_A, final_B, resolved_name, model)
            applied += 1

            # Track per-LoRA column boundaries within appended columns
            if not hasattr(model, "_lora_boundaries"):
                model._lora_boundaries = {}
            boundaries = []
            offset = 0
            for lora_idx, lora_rank in per_lora_ranks:
                boundaries.append({"lora_idx": lora_idx, "offset": offset, "rank": lora_rank})
                offset += lora_rank
            model._lora_boundaries[resolved_name] = boundaries
        except Exception as ex:
            logger.warning(f"[FAIL] {resolved_name}: {ex}")

    logger.info(f"LoRA composition complete. Applied to {applied} modules.")
    return applied


def set_lora_strength(model: nn.Module, strength: float) -> None:
    """Adjust global LoRA strength multiplier (nunchaku modules only)."""
    if not hasattr(model, "_lora_slots") or not model._lora_slots:
        logger.warning("No LoRA slots found.")
        return

    old = getattr(model, "_lora_strength", 1.0)
    factor = strength / old if old != 0 else 0

    for name, info in model._lora_slots.items():
        if info.get("type") != "nunchaku":
            continue
        module = _get_module(model, name)
        if module is None or info.get("appended", 0) <= 0:
            continue
        base_rank = info["base_rank"]
        appended = info["appended"]
        with torch.no_grad():
            module.proj_up.data[:, base_rank:base_rank + appended] *= factor

    model._lora_strength = strength


def reset_lora(model: nn.Module) -> None:
    """Remove all LoRA weights from the model."""
    if not hasattr(model, "_lora_slots") or not model._lora_slots:
        return

    for name, info in model._lora_slots.items():
        module = _get_module(model, name)
        if module is None:
            continue

        mtype = info.get("type", "nunchaku")

        if mtype == "nunchaku":
            base_rank = info["base_rank"]
            with torch.no_grad():
                pd = unpack_lowrank_weight(module.proj_down.data, down=True)
                pu = unpack_lowrank_weight(module.proj_up.data, down=False)
                if info.get("axis_down", 0) == 0:
                    pd_reset = pd[:base_rank, :].clone()
                else:
                    pd_reset = pd[:, :base_rank].clone()
                pu_reset = pu[:, :base_rank].clone()
                module.proj_down.data = pack_lowrank_weight(pd_reset, down=True)
                module.proj_up.data = pack_lowrank_weight(pu_reset, down=False)
                module.rank = base_rank

        elif mtype == "linear":
            if "original_weight" in info:
                with torch.no_grad():
                    module.weight.data.copy_(info["original_weight"].to(module.weight.device))

    model._lora_slots.clear()
    model._lora_strength = 1.0
    logger.info("All LoRA weights reset.")
