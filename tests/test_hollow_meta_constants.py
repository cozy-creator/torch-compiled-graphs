"""A lifted constant DERIVED FROM PARAMETERS is meta, and cannot be copied.

tcg#66. ``_empty_parameters`` registers every parameter on meta so a 10 GiB
denoiser costs nothing to build. A module that computes a plain tensor
ATTRIBUTE from those parameters during ``__init__`` therefore ends up with a
meta attribute -- and ``_rehome`` moved lifted constants with ``.to(device)``,
which is undefined for a tensor that has no storage:

    NotImplementedError: Cannot copy out of meta tensor; no data!

The canonical producer is ``torch.nn.utils.weight_norm``, which deletes the
``weight`` Parameter and installs ``weight = _weight_norm(weight_v, weight_g,
dim)`` as an attribute. Every DAC/BigVGAN-derived audio VAE applies it to
every conv -- which is why minimax-h3's ``audio_vae`` died where its other
eight components did not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")

import torch  # noqa: E402

from torchcg.hollow import (  # noqa: E402
    HollowError,
    _empty_parameters,
    _rehome,
    _tensor_attributes,
    hollow_session,
    virtualize_parameters,
)


class WeightNormed(torch.nn.Module):
    """The shape, with nothing else in it: one weight_norm'd conv."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.utils.weight_norm(torch.nn.Conv1d(4, 4, 3))

    def forward(self, x: torch.Tensor) -> Any:
        return self.conv(x)


def _attributes(module: Any) -> dict[str, Any]:
    return {
        f"{prefix}.{name}": value
        for prefix, submodule in module.named_modules()
        for name, value in _tensor_attributes(submodule)
    }


def test_a_parameter_derived_attribute_lands_on_meta(recwarn: Any) -> None:
    """WHY it happens, pinned WITHOUT the session -- this is torch, not us.

    The filing left this unknown and it decides where the fix belongs: the
    attribute is not meta by construction and no low-memory load path put it
    there. It is meta because its PARENTS are, which is exactly what a hollow
    build does on purpose.
    """

    with _empty_parameters():
        module = WeightNormed()

    parameters = {name: p for name, p in module.named_parameters()}
    assert sorted(parameters) == ["conv.bias", "conv.weight_g", "conv.weight_v"]
    assert {p.device.type for p in parameters.values()} == {"meta"}

    # `weight` is no longer a Parameter at all -- weight_norm made it a plain
    # attribute computed from the two above, so it inherits their meta-ness.
    attributes = _attributes(module)
    assert list(attributes) == ["conv.weight"]
    assert attributes["conv.weight"].device.type == "meta"


def test_rehome_refuses_a_meta_tensor_by_name() -> None:
    """The typed refusal, so no future caller re-introduces the raw copy.

    Never a broad try/except at the call site: that is how a loud, correct,
    by-name refusal becomes a silently dropped component (tcg#65's defect).
    """

    with pytest.raises(HollowError) as refusal:
        _rehome(torch.zeros(2, 2, device="meta"), "cpu")

    assert "no storage to move" in str(refusal.value)
    assert "virtualized, not copied" in str(refusal.value)


def test_a_meta_lifted_constant_is_virtualized_like_the_parameters_it_came_from() -> None:
    with hollow_session("cuda") as session:
        with _empty_parameters():
            module = WeightNormed()
        virtualize_parameters(module, session, dtype=torch.bfloat16)

        weight = _attributes(module)["conv.weight"]
        assert type(weight).__name__ == "FakeTensor"
        assert weight.device.type == session.drive_device_type
        assert weight.dtype is torch.bfloat16
        assert tuple(weight.shape) == (4, 4, 3)


def test_a_meta_BUFFER_still_refuses_and_the_asymmetry_is_deliberate() -> None:
    """A buffer's value is a function of CONFIG and feeds the literal digest.

    An attribute derived from emptied parameters has no value to lose. Fenced
    because "meta is fine now" is the wrong lesson to take from the fix.
    """

    class MetaBuffered(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("sigmas", torch.zeros(4, device="meta"))

    with hollow_session("cuda") as session:
        with pytest.raises(HollowError) as refusal:
            virtualize_parameters(MetaBuffered(), session)

    assert "sigmas" in str(refusal.value)
    assert "function of config" in str(refusal.value)


@pytest.fixture(scope="module")
def oobleck_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A REAL diffusers audio VAE, config-only.

    ``AutoencoderOobleck`` is DAC-derived and weight_norms every conv -- the
    same family as the ``AutoencoderKLMiniMaxH3Audio`` that found this, and
    stock, so the subject is not a hand-built stand-in.
    """

    diffusers = pytest.importorskip("diffusers")
    autoencoder = getattr(diffusers, "AutoencoderOobleck", None)
    if autoencoder is None:
        pytest.skip("this diffusers has no AutoencoderOobleck")

    tree = tmp_path_factory.mktemp("oobleck-config-only")
    autoencoder(
        encoder_hidden_size=16,
        downsampling_ratios=[2, 2],
        channel_multiples=[1, 2],
        decoder_channels=16,
        decoder_input_channels=4,
        audio_channels=2,
        sampling_rate=16000,
    ).save_pretrained(tree)
    removed = 0
    for pattern in ("*.safetensors", "*.bin"):
        for weight_file in tree.rglob(pattern):
            weight_file.unlink()
            removed += 1
    assert removed > 0, "the saved tree carried no weight files to delete"
    return tree


def test_a_real_weight_normed_audio_vae_loads_hollow_and_exports(
    oobleck_tree: Path,
) -> None:
    """End to end through the intercepted loader, then drive, then export."""

    import diffusers

    with hollow_session("cuda") as session:
        model = diffusers.AutoencoderOobleck.from_pretrained(
            oobleck_tree, torch_dtype=torch.bfloat16
        )

        constants = _attributes(model)
        assert len(constants) > 0, "the fixture must actually carry lifted constants"
        assert {type(value).__name__ for value in constants.values()} == {"FakeTensor"}
        assert {value.device.type for value in constants.values()} == {
            session.drive_device_type
        }
        assert {value.dtype for value in constants.values()} == {torch.bfloat16}

        sample = torch.zeros(1, 2, 64, device=session.drive_device, dtype=torch.bfloat16)
        with torch.no_grad():
            driven = model.encoder(sample)
        assert type(driven).__name__ == "FakeTensor"

        program = torch.export.export(model.encoder, (sample,), strict=False)

    # Device uniformity must be TOTAL: a lifted constant left behind is the
    # failure that exports cleanly and only AOTI rejects.
    placements = {str(tensor.device) for tensor in program.state_dict.values()}
    placements |= {
        str(tensor.device)
        for tensor in getattr(program, "constants", {}).values()
        if isinstance(tensor, torch.Tensor)
    }
    assert placements == {session.drive_device_type}
