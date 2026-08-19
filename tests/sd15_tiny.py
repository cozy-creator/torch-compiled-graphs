"""A tiny, config-only SD1.5-class pipeline: real author code, generated weights.

Nothing is downloaded -- the tokenizer's vocabulary is written locally and
every module is instantiated from config with seeded random weights (a few MB,
box-generated, per the micro-rig carve-out). This is the ship-code-as-is
acceptance subject: stock diffusers ``StableDiffusionPipeline``, no wrapper,
no catalog.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from stackfixture import local_stack

from torchcg.discovery import discover_lane
from torchcg.document import GraphSetDocument

# The contract file's spelling (sdxl main_v2.py): a lane IS a tensorfs
# contract reference; compile targets are endpoint-level attribute paths on
# the pipeline's own components -- the module named is the one actually
# CALLED.
LANE_CONTRACT = "sd15.diffusers-fp32@1"
COMPILE_TARGETS = ("unet", "vae.decoder")


def build_pipe() -> Any:
    import torch
    from diffusers import (
        AutoencoderKL,
        DDIMScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=16,
        attention_head_dim=2,
        norm_num_groups=4,
    )
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=32,
    )
    text_encoder = CLIPTextModel(
        CLIPTextConfig(
            hidden_size=16,
            intermediate_size=16,
            num_attention_heads=2,
            num_hidden_layers=2,
            vocab_size=1000,
            max_position_embeddings=77,
            projection_dim=16,
        )
    )
    vocab_dir = Path(tempfile.mkdtemp(prefix="sd15-tiny-vocab-"))
    letters = [chr(code) for code in range(ord("a"), ord("z") + 1)]
    vocab = {
        token: index
        for index, token in enumerate(
            ["<|startoftext|>", "<|endoftext|>"]
            + [f"{letter}</w>" for letter in letters]
            + letters
        )
    }
    (vocab_dir / "vocab.json").write_text(json.dumps(vocab))
    (vocab_dir / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(str(vocab_dir / "vocab.json"), str(vocab_dir / "merges.txt"))
    tokenizer.model_max_length = 77
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    # Serve-faithful mode: a real `from_pretrained` load lands in eval, and
    # the train-flag is a trace input (dropout bakes it into the graph).
    unet.eval()
    vae.eval()
    text_encoder.eval()
    return StableDiffusionPipeline(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )


def drive(pipe: Any) -> None:
    """The author's own sample invocation: the pipeline loop runs as-is."""

    pipe(
        prompt="a cat",
        num_inference_steps=2,
        guidance_scale=7.5,
        height=32,
        width=32,
        output_type="pt",
    )


def discover_document() -> GraphSetDocument:
    pipe = build_pipe()
    lane = discover_lane(
        LANE_CONTRACT, COMPILE_TARGETS, pipe.components, lambda: drive(pipe)
    )
    return GraphSetDocument(stack=local_stack(), lanes=(lane,))


if __name__ == "__main__":
    import hashlib

    print(hashlib.sha256(discover_document().encode()).hexdigest())
