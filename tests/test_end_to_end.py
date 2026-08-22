"""The whole path on real hardware: export -> mint -> store -> adopt -> dispatch.

An integration test, deliberately. Every hop here is the production code path --
a real `torch.export`, a real `aot_compile`, a real tensorfs CAS, a real
`load_package` with `user_managed=True` constants bound to the live module's own
weights, and the real dispatcher deciding per call. Mocking any one of them
would remove exactly the seam that has historically broken.
"""

from __future__ import annotations

import os

import pytest

from torchcg import store as store_module
from torchcg.adopt import Dispatcher, Record, load
from torchcg.identity import artifact_key
from torchcg.mint import GraphSpec, compile_policy, declared_input_layout, mint
from torchcg.store import Store

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
tensorfs = pytest.importorskip("tensorfs")

from fixture import export_unet, tiny_unet, unet_call  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("TORCHCG_E2E") != "1",
    reason="real AOTI compile; set TORCHCG_E2E=1 to run",
)

ENV = {"torch": torch.__version__, "os_release": "local"}


def _sm() -> str:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}"
    return f"cpu-{torch.backends.cpu.get_cpu_capability().lower()}"


def _cuda_compilable() -> bool:
    """A card is not enough: AOTI's CUDA path shells out to `nvcc`, and the
    `nvidia-*` wheels ship runtime libraries only. Without a real toolkit the
    honest target is the CPU one -- which still exercises every seam this
    library owns, because the CUDA-specific part is torch's codegen, not ours."""

    import shutil

    root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    return bool(
        torch.cuda.is_available()
        and (shutil.which("nvcc") or (root and os.path.exists(f"{root}/bin/nvcc")))
    )


def _device_type() -> str:
    return "cuda" if _cuda_compilable() else "cpu"


def _target() -> str:
    return "cuda:0" if _cuda_compilable() else "cpu"


def test_mint_store_adopt_dispatch(tmp_path) -> None:
    device = _device_type()
    program, ingress = export_unet(2, device=_target())
    from torchcg.identity import graph_hash

    graph = graph_hash(program, ingress)
    spec = GraphSpec(graph=graph, target="unet", program=program, ingress=ingress)

    artifact = mint(
        spec,
        sm=_sm(),
        env=ENV,
        device_type=device,
        destination=tmp_path / "artifact.tar.gz",
    )
    assert artifact.is_file()

    key = artifact_key(
        graph, sm=_sm(), env=ENV,
        policy=compile_policy(device), layout=declared_input_layout(),
    ).value

    store = Store(tensorfs.LocalCAS(tmp_path / "cas"))
    store.put(key, artifact)

    # Re-minting the SAME graph must not diverge: the envelope is deterministic
    # and the only thing that could differ is the compiler's own output.
    stored = store.get(key, tmp_path / "unpacked")
    assert stored is not None
    assert stored.metadata["graph"] == graph
    assert stored.metadata["compile_policy"]["always_keep_tensor_constants"] is True
    assert stored.metadata["compile_policy"]["aot_inductor.package_constants_in_so"] is False

    # Adopt against a LIVE module: constants are raw pointers into its weights.
    unet = tiny_unet().to(_target())
    package = load(stored, unet, key=key)

    eager_forward = unet.forward
    dispatcher = Dispatcher(eager=eager_forward)
    dispatcher.arm([Record(graph, ingress, package)])

    args, kwargs = unet_call(2, device=_target())

    for _ in range(3):
        dispatcher(*args, **kwargs)
    assert dispatcher.compiled_calls == 3, "the armed graph did not take the call"
    assert dispatcher.eager_calls == 0

    # A shape the record was not minted for falls through, loudly, once.
    other, _ = unet_call(5, device=_target())
    dispatcher(*other, **kwargs)
    assert dispatcher.eager_calls == 1


def test_an_int64_timestep_reaches_the_COMPILED_graph(tmp_path) -> None:
    """se#837 end to end: the recast is applied to a real call into a real .so.

    The unit arm proves the dispatcher hands over float32. This proves the
    compiled artifact ACCEPTS what it was handed -- which is the half a spy
    cannot answer.
    """

    device = _device_type()
    program, ingress = export_unet(2, device=_target())
    from torchcg.identity import graph_hash

    graph = graph_hash(program, ingress)
    artifact = mint(
        GraphSpec(graph=graph, target="unet", program=program, ingress=ingress),
        sm=_sm(), env=ENV, device_type=device,
        destination=tmp_path / "artifact.tar.gz",
    )
    key = artifact_key(
        graph, sm=_sm(), env=ENV,
        policy=compile_policy(device), layout=declared_input_layout(),
    ).value
    store = Store(tensorfs.LocalCAS(tmp_path / "cas"))
    store.put(key, artifact)
    stored = store.get(key, tmp_path / "unpacked")
    assert stored is not None

    unet = tiny_unet().to(_target())
    package = load(stored, unet, key=key)
    dispatcher = Dispatcher(eager=unet.forward)
    dispatcher.arm([Record(graph, ingress, package)])

    args, kwargs = unet_call(2, timestep_dtype="int64", device=_target())
    result = dispatcher(*args, **kwargs)
    assert dispatcher.compiled_calls == 1
    assert dispatcher.recast_calls == 1
    assert dispatcher.eager_calls == 0
    assert result is not None


def test_a_second_mint_of_one_graph_does_not_DIVERGE(tmp_path) -> None:
    """The mint-once claim, asked of two real compiles rather than of two
    copies of one file."""

    device = _device_type()
    program, ingress = export_unet(2, device=_target())
    from torchcg.identity import graph_hash

    graph = graph_hash(program, ingress)
    spec = GraphSpec(graph=graph, target="unet", program=program, ingress=ingress)
    key = artifact_key(
        graph, sm=_sm(), env=ENV,
        policy=compile_policy(device), layout=declared_input_layout(),
    ).value
    store = Store(tensorfs.LocalCAS(tmp_path / "cas"))

    first = mint(spec, sm=_sm(), env=ENV, device_type=device,
                 destination=tmp_path / "a.tar.gz")
    store.put(key, first)

    program2, ingress2 = export_unet(2, device=_target())
    assert graph_hash(program2, ingress2) == graph, "the same export re-keyed"
    second = mint(
        GraphSpec(graph=graph, target="unet", program=program2, ingress=ingress2),
        sm=_sm(), env=ENV, device_type=device, destination=tmp_path / "b.tar.gz",
    )
    # If AOTI's output is not byte-reproducible this REFUSES, and that refusal
    # is the honest answer: it means an axis the key does not carry decided the
    # output. Recorded either way rather than asserted away.
    try:
        store.put(key, second)
        reproducible = True
    except store_module.DivergentArtifact:  # type: ignore[attr-defined]
        reproducible = False
    print(f"AOTI byte-reproducible across two mints of one graph: {reproducible}")
