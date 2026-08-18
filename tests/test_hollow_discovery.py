"""pgw#1370 acceptance seed: the weights-free derive states the SAME identity.

The publish pipeline holds a CONFIG-ONLY checkpoint subset -- never weights --
and the author's code still says ``from_pretrained(...).to("cuda")``. Under
``hollow_session`` that code runs as-is on a cardless CPU box: parameters are
fake, buffers real, ``"cuda"`` is remapped, fake egress answers zeros. The
proof that observation is structure-only is byte-equality: the hollow drive's
document equals the real-weight drive's document, byte for byte.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")

import sd15_tiny  # noqa: E402
import torch  # noqa: E402

from torchcg.discovery import discover_lane  # noqa: E402
from torchcg.document import GraphSetDocument  # noqa: E402
from torchcg.graph_identity import closure_hash, installed_closure  # noqa: E402
from torchcg.hollow import (  # noqa: E402
    HollowError,
    hollow_session,
    virtualize_parameters,
)

_WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.ckpt")


@pytest.fixture(scope="module")
def config_only_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny SD15 checkpoint with every weight file DELETED."""

    tree = tmp_path_factory.mktemp("sd15-config-only")
    pipe = sd15_tiny.build_pipe()
    pipe.save_pretrained(tree)
    removed = 0
    for pattern in _WEIGHT_PATTERNS:
        for weight_file in tree.rglob(pattern):
            weight_file.unlink()
            removed += 1
    assert removed > 0, "the saved tree carried no weight files to delete"
    return tree


def _hollow_document(tree: Path, *, torch_dtype: object = None) -> GraphSetDocument:
    from diffusers import StableDiffusionPipeline

    with hollow_session():
        pipe = StableDiffusionPipeline.from_pretrained(tree, torch_dtype=torch_dtype)
        pipe.to("cuda")  # author code as-is; the session remaps to cpu
        lane = discover_lane(
            sd15_tiny.LANE_CONTRACT,
            sd15_tiny.COMPILE_TARGETS,
            pipe.components,
            lambda: pipe(
                prompt="a cat",
                num_inference_steps=2,
                guidance_scale=7.5,
                height=32,
                width=32,
                # "pil" exercises fake egress: postprocess calls .numpy() on
                # a fake image and must receive zeros, not a refusal.
                output_type="pil",
            ),
        )
    return GraphSetDocument(closure=closure_hash(installed_closure()), lanes=(lane,))


def test_hollow_drive_states_the_real_drive_identity(config_only_tree: Path) -> None:
    hollow = _hollow_document(config_only_tree)
    real = sd15_tiny.discover_document()
    assert hollow.encode() == real.encode()


def test_hollow_parameters_allocated_nothing(config_only_tree: Path) -> None:
    from diffusers import StableDiffusionPipeline
    from torch._subclasses.fake_tensor import FakeTensor

    with hollow_session():
        pipe = StableDiffusionPipeline.from_pretrained(config_only_tree)
        parameters = list(pipe.unet.parameters()) + list(pipe.vae.parameters()) + list(
            pipe.text_encoder.parameters()
        )
        assert parameters and all(isinstance(p, FakeTensor) for p in parameters)
        # Buffers stay REAL: they are functions of config and feed the
        # literal digest.
        for buffer in pipe.text_encoder.buffers():
            assert not isinstance(buffer, FakeTensor)


def test_lane_dtype_is_a_derivation_input(config_only_tree: Path) -> None:
    """A bf16 hollow load derives DIFFERENT graphs than the fp32 one."""

    fp32 = _hollow_document(config_only_tree)
    bf16 = _hollow_document(config_only_tree, torch_dtype=torch.bfloat16)
    fp32_hashes = {record.graph for lane in fp32.lanes for record in lane.graphs}
    bf16_hashes = {record.graph for lane in bf16.lanes for record in lane.graphs}
    assert fp32_hashes.isdisjoint(bf16_hashes)


def test_construction_allocates_parameters_on_meta() -> None:
    """The build itself allocates nothing: parameters are born on meta.

    ``virtualize_parameters`` fakes them afterwards either way; this pins the
    property that a 10 GiB denoiser never costs 10 GiB to CONSTRUCT.
    """

    from torchcg.hollow import _empty_parameters

    with _empty_parameters():
        layer = torch.nn.Linear(4, 4)
    assert layer.weight.device.type == "meta"
    assert layer.bias.device.type == "meta"


def test_a_meta_buffer_is_refused_not_zeroed() -> None:
    """A module that moves buffers to meta cannot state a literal digest."""

    class BadHost(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2))
            self.register_buffer("table", torch.zeros(2, device="meta"))

    with hollow_session() as session:
        with pytest.raises(HollowError, match="table"):
            virtualize_parameters(BadHost(), session)


def test_hollow_refuses_when_the_tree_lacks_configs(tmp_path: Path) -> None:
    from diffusers import UNet2DConditionModel

    with hollow_session():
        with pytest.raises(HollowError, match="UNet2DConditionModel"):
            UNet2DConditionModel.from_pretrained(tmp_path, subfolder="unet")
