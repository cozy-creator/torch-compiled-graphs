"""A tiny SD1.5-class UNet: real diffusers author code, generated weights.

Nothing is downloaded and nothing is a stub -- this is a stock
``UNet2DConditionModel`` at toy widths, so the trace exercises the real
cross-attention/resnet/timestep-embedding structure that every banked fact was
measured against, at a size this box can compile.
"""

from __future__ import annotations

from typing import Any


def tiny_unet(**overrides: Any) -> Any:
    import torch
    from diffusers import UNet2DConditionModel

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        **{
            "sample_size": 8,
            "in_channels": 4,
            "out_channels": 4,
            "layers_per_block": 1,
            "block_out_channels": (8, 16),
            "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D"),
            "up_block_types": ("CrossAttnUpBlock2D", "UpBlock2D"),
            "cross_attention_dim": 16,
            "attention_head_dim": 2,
            "norm_num_groups": 4,
            **overrides,
        }
    )
    # A real `from_pretrained` load lands in eval, and the train flag is a trace
    # input: dropout bakes itself into the graph.
    unet.eval()
    return unet


def unet_call(
    batch: int = 2, *, timestep_dtype: str = "float32", device: str = "cpu"
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """The author's own call shape: (sample, timestep, encoder_hidden_states)."""

    import torch

    return (
        torch.zeros(batch, 4, 8, 8, device=device),
        torch.zeros((), dtype=getattr(torch, timestep_dtype), device=device),
        torch.zeros(batch, 77, 16, device=device),
    ), {"return_dict": False}


UNET_PARAMS = ("sample", "timestep", "encoder_hidden_states", "return_dict")


def export_unet(batch: int = 2, *, device: str = "cpu", **kwargs: Any) -> tuple[Any, Any]:
    """Export the tiny UNet and build its ingress. Returns (program, ingress).

    The module is moved to `device` BEFORE the trace: a graph is traced onto one
    device and cannot be re-homed afterwards.
    """

    import torch

    from torchcg.identity import build_call_ingress

    unet = tiny_unet().to(device)
    args, call_kwargs = unet_call(batch, device=device, **kwargs)
    program = torch.export.export(unet, args, call_kwargs, strict=False)
    ingress = build_call_ingress(program, UNET_PARAMS, args, call_kwargs)
    return program, ingress
