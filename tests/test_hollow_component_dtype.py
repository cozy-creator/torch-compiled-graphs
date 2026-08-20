"""Precision is a PER-COMPONENT fact, so the session asks per component.

tcg#68 / pgw#1512. A checkpoint may carry a bf16 denoiser beside an fp32 VAE.
Casting the whole tree to one dtype at load time is the conversion the serving
loader explicitly refuses to make ("bytes land verbatim in the container's own
dtype … any conversion is the STORE's job, never load time"), and doing it at
trace made a bf16 bias meet an fp32 activation inside a decode block that is
fine at serve.

The session therefore carries a POLICY (`dtype_for`) rather than a dtype, and
only the caller that knows which component a layout contract describes can
supply it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")

import torch  # noqa: E402

from torchcg.hollow import hollow_session  # noqa: E402


@pytest.fixture(scope="module")
def two_precision_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A config-only tree whose two components declare DIFFERENT dtypes."""

    from diffusers import AutoencoderKL, UNet2DConditionModel

    tree = tmp_path_factory.mktemp("two-precision")
    UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=16, attention_head_dim=2, norm_num_groups=4,
    ).save_pretrained(tree / "unet")
    AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,), layers_per_block=1, latent_channels=4,
        norm_num_groups=4, sample_size=32,
    ).save_pretrained(tree / "vae")
    for pattern in ("*.safetensors", "*.bin"):
        for weight in tree.rglob(pattern):
            weight.unlink()
    return tree


def _first_param_dtype(module: Any) -> Any:
    return next(module.parameters()).dtype


def test_each_component_takes_the_dtype_the_POLICY_gives_IT(
    two_precision_tree: Path,
) -> None:
    """The whole point: two components, two precisions, one session."""

    from diffusers import AutoencoderKL, UNet2DConditionModel

    asked: list[tuple[str, str | None]] = []

    def policy(path: Any, subfolder: Any) -> Any:
        asked.append((Path(str(path)).name, subfolder))
        return torch.bfloat16 if subfolder == "unet" else torch.float32

    with hollow_session("cuda", dtype_for=policy):
        unet = UNet2DConditionModel.from_pretrained(
            two_precision_tree, subfolder="unet"
        )
        vae = AutoencoderKL.from_pretrained(two_precision_tree, subfolder="vae")

    assert _first_param_dtype(unet) is torch.bfloat16
    assert _first_param_dtype(vae) is torch.float32
    # The policy was consulted per component, with the subfolder that
    # identifies it — not once for the tree.
    assert [subfolder for _tree, subfolder in asked] == ["unet", "vae"]


def test_an_EXPLICIT_torch_dtype_is_the_author_speaking_and_wins(
    two_precision_tree: Path,
) -> None:
    """A policy must never override a dtype the caller asked for by name."""

    from diffusers import UNet2DConditionModel

    def policy(_path: Any, _subfolder: Any) -> Any:
        return torch.float32

    with hollow_session("cuda", dtype_for=policy):
        unet = UNet2DConditionModel.from_pretrained(
            two_precision_tree, subfolder="unet", torch_dtype=torch.bfloat16
        )

    assert _first_param_dtype(unet) is torch.bfloat16


def test_no_policy_means_no_cast_which_is_this_package_standing_alone(
    two_precision_tree: Path,
) -> None:
    """torchcg states no precision policy of its own — the caller does.

    Pinned so the default is a deliberate absence rather than an accident: a
    silent default here would be exactly the fp32 fallback pgw#1448 deleted,
    wearing a library's name.
    """

    from diffusers import UNet2DConditionModel

    with hollow_session("cuda") as session:
        assert session.dtype_for is None
        unet = UNet2DConditionModel.from_pretrained(
            two_precision_tree, subfolder="unet"
        )

    assert _first_param_dtype(unet) is torch.float32


def test_the_policy_may_REFUSE_and_the_refusal_reaches_the_caller(
    two_precision_tree: Path,
) -> None:
    """"No answer" must be expressible as an error, not as a default.

    The derive's policy raises when neither the contract nor the container can
    say what a component's precision is; nothing here may swallow that into a
    quiet fp32.
    """

    from diffusers import UNet2DConditionModel

    class Undeclared(RuntimeError):
        pass

    def policy(_path: Any, subfolder: Any) -> Any:
        raise Undeclared(f"nothing declares a dtype for {subfolder!r}")

    with hollow_session("cuda", dtype_for=policy):
        with pytest.raises(Undeclared) as refusal:
            UNet2DConditionModel.from_pretrained(two_precision_tree, subfolder="unet")

    assert "'unet'" in str(refusal.value)
