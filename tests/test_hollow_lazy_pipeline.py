"""A LAZY pipeline loader hands back specs; a hollow load must hand back a pipeline.

``DiffusionPipeline.from_pretrained`` returns a pipeline CARRYING every
component, so intercepting the component loaders is enough to make that load
hollow. ``ModularPipeline.from_pretrained`` does not: it builds the component
SPECS and leaves every ``from_pretrained`` attribute ``None``, deferring the
build to whoever holds the object. At serve that holder exists (the streaming
engine passes the components in). A derive has none -- so it marked an empty
pipeline, found no denoiser, and armed nothing (pgw#1450: eleven declared
names, eleven ``None`` attributes, *"no DiT resolves on this pipeline"*).

The subject is the SHAPE, not a family: a tiny SD15-class tree re-indexed as a
modular repo, loaded through a pipeline class that takes its components at
construction -- the same adapter every modular endpoint writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")

import diffusers  # noqa: E402
import sd15_tiny  # noqa: E402
import torch  # noqa: E402
from diffusers.modular_pipelines.modular_pipeline import (  # noqa: E402
    ModularPipeline,
    ModularPipelineBlocks,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec  # noqa: E402

from torchcg.hollow import HollowError, hollow_session  # noqa: E402

_WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.ckpt")

#: The tiny tree's components, in the modular index's own spelling.
_COMPONENTS = {
    "unet": ("diffusers", "UNet2DConditionModel"),
    "vae": ("diffusers", "AutoencoderKL"),
    "text_encoder": ("transformers", "CLIPTextModel"),
    "tokenizer": ("transformers", "CLIPTokenizer"),
    "scheduler": ("diffusers", "DDIMScheduler"),
}

#: The nn.Module half -- what a lane's compile marks and a derive must find.
_MODULES = ("unet", "vae", "text_encoder")


class TinyStep(ModularPipelineBlocks):  # type: ignore[misc]
    model_name = "tiny-lazy"

    @property
    def description(self) -> str:
        return "the block whose expected_components ARE the pipeline's"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        return [
            ComponentSpec("unet", UNet2DConditionModel),
            ComponentSpec("vae", AutoencoderKL),
            ComponentSpec("text_encoder", CLIPTextModel),
            ComponentSpec("tokenizer", CLIPTokenizer),
            ComponentSpec("scheduler", DDIMScheduler),
        ]

    @property
    def inputs(self) -> list[Any]:
        return []

    @property
    def intermediate_outputs(self) -> list[Any]:
        return []

    def __call__(self, components: Any, state: Any) -> Any:
        return components, state


class TinyBlocks(SequentialPipelineBlocks):  # type: ignore[misc]
    model_name = "tiny-lazy"
    block_classes = [TinyStep]
    block_names = ["tiny"]


class TinyStreamingPipeline(ModularPipeline):  # type: ignore[misc]
    """The adapter every modular endpoint writes (minimax-h3's, verbatim shape).

    ``ModularPipeline.__init__`` routes ``**kwargs`` to ``load_config`` and
    registers every component as ``None``, so an author who wants a pipeline
    that KEEPS its constructor arguments attaches them afterwards.
    """

    default_blocks_name = "TinyBlocks"

    _OWN = (
        "blocks",
        "pretrained_model_name_or_path",
        "components_manager",
        "collection",
        "modular_config_dict",
        "config_dict",
    )

    def __init__(self, **kwargs: Any) -> None:
        own = {name: kwargs.pop(name) for name in self._OWN if name in kwargs}
        super().__init__(**own)
        components = {
            name: value
            for name, value in kwargs.items()
            if name in self._component_specs and value is not None
        }
        if components:
            self.update_components(**components)
        left = sorted(name for name in kwargs if name not in self._component_specs)
        if left:
            raise TypeError(f"{type(self).__name__}: {left} are not components")


@pytest.fixture(scope="module", autouse=True)
def _resolvable_by_name() -> Any:
    """Modular diffusers resolves blocks and pipeline classes off the package.

    The test's classes are registered there for the duration, which is how a
    real endpoint's classes are reachable too (they ship inside the repo the
    index names). Nothing about the seam under test knows either name.
    """

    for cls in (TinyBlocks, TinyStreamingPipeline):
        setattr(diffusers, cls.__name__, cls)
    yield
    for cls in (TinyBlocks, TinyStreamingPipeline):
        delattr(diffusers, cls.__name__)


def _write_modular_index(tree: Path, *, repo: str) -> None:
    index = {
        "_class_name": TinyStreamingPipeline.__name__,
        "_blocks_class_name": TinyBlocks.__name__,
        "_diffusers_version": diffusers.__version__,
    }
    for name, (library, class_name) in _COMPONENTS.items():
        index[name] = [
            library,
            class_name,
            {
                "repo": repo,
                "subfolder": name,
                "type_hint": [library, class_name],
                "variant": None,
                "revision": None,
            },
        ]
    (tree / "modular_model_index.json").write_text(json.dumps(index, indent=2))


@pytest.fixture(scope="module")
def modular_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny SD15 tree, weights deleted, indexed as a MODULAR repo."""

    tree = tmp_path_factory.mktemp("tiny-modular-config-only")
    sd15_tiny.build_pipe().save_pretrained(tree)
    removed = 0
    for pattern in _WEIGHT_PATTERNS:
        for weight_file in tree.rglob(pattern):
            weight_file.unlink()
            removed += 1
    assert removed > 0, "the saved tree carried no weight files to delete"
    _write_modular_index(tree, repo=str(tree))
    return tree


def test_the_lazy_loader_itself_attaches_nothing(modular_tree: Path) -> None:
    """The defect's source, OUTSIDE any session: this is diffusers' design.

    Pinned here because it is what makes the seam torchcg's: the loader is
    behaving as documented, so no amount of component-loader interception can
    produce an attached pipeline. Something has to finish the build.
    """

    pipeline = TinyStreamingPipeline.from_pretrained(str(modular_tree))

    assert sorted(pipeline.components) == sorted(_COMPONENTS)
    assert sorted(name for name, value in pipeline.components.items() if value is None) == sorted(
        _COMPONENTS
    )


def test_a_hollow_load_returns_a_pipeline_that_carries_its_components(
    modular_tree: Path,
) -> None:
    """The fix: every declared component attached, every module hollow."""

    with hollow_session("cuda") as session:
        pipeline = TinyStreamingPipeline.from_pretrained(
            str(modular_tree), torch_dtype=torch.bfloat16
        )

        unattached = [name for name, value in pipeline.components.items() if value is None]
        assert unattached == []

        modules = sorted(
            name
            for name, value in pipeline.components.items()
            if isinstance(value, torch.nn.Module)
        )
        assert modules == sorted(_MODULES)

        for name in _MODULES:
            parameter = next(getattr(pipeline, name).parameters())
            # Hollow, on the DRIVE device, at the dtype the caller asked for:
            # the lazy build is where a modular pipeline's torch_dtype belongs,
            # and the pipeline constructor never accepted it.
            assert isinstance(parameter, torch.Tensor)
            assert parameter.device.type == session.drive_device_type
            assert parameter.dtype is torch.bfloat16
            assert type(parameter).__name__ == "FakeTensor"


def test_the_pipeline_is_marked_by_the_same_components_read_a_derive_uses(
    modular_tree: Path,
) -> None:
    """``ctx.compile(pipe)`` marks ``pipe.components``; it must find modules.

    The refusal pgw#1450 reported is downstream of this read returning zero
    nn.Modules, so the read is asserted rather than the attribute walk.
    """

    with hollow_session("cuda"):
        pipeline = TinyStreamingPipeline.from_pretrained(str(modular_tree))
        marked = [
            value
            for value in pipeline.components.values()
            if isinstance(value, torch.nn.Module)
        ]

    assert len(marked) == len(_MODULES)


def test_a_component_that_cannot_be_built_is_named_and_refused(
    tmp_path: Path, modular_tree: Path
) -> None:
    """tcg#59's doctrine: intercept-or-REFUSE, never silently pass through.

    The lazy build logs per-component failures and returns a pipeline anyway,
    which is exactly the silence this issue is about. A component the index
    says is buildable and that is still ``None`` afterwards is a refusal.
    """

    broken = tmp_path / "broken-tree"
    broken.mkdir()
    for member in modular_tree.iterdir():
        if member.name != "modular_model_index.json":
            (broken / member.name).symlink_to(member)
    _write_modular_index(broken, repo=str(tmp_path / "not-a-tree"))

    with hollow_session("cuda"):
        with pytest.raises(HollowError) as refusal:
            TinyStreamingPipeline.from_pretrained(str(broken))

    message = str(refusal.value)
    assert "TinyStreamingPipeline" in message
    for name in _MODULES:
        assert name in message


def test_the_loader_is_restored_when_the_session_ends(modular_tree: Path) -> None:
    """The patch is the session's, not the process's."""

    before = ModularPipeline.__dict__["from_pretrained"]
    with hollow_session("cuda"):
        assert ModularPipeline.__dict__["from_pretrained"] is not before
    assert ModularPipeline.__dict__["from_pretrained"] is before
