"""tcg#72: warm reuse without torch, and warm fetch without re-hashing.

Two independent costs, one cause (pgw#1546): a store that already holds the
bytes still made every warm pass pay per-artifact bookkeeping.

* ``Engine.reuse_index`` -- the (graph name -> exact key) probe. Before it, the
  only way to learn "this mint already happened" was to rebuild the plan, which
  needs the exported program loaded, which needs torch: ~5 s of import and
  deserialize per graph, measured, against 0.2 s of actual store work.
* ``LocalGraphStore`` fetch stamps -- a verified copy-out records what it
  verified; a later fetch whose destination is untouched serves it without
  re-hashing or re-copying. Every red arm here is a way the stamp must refuse.

No torch anywhere in this file, which is itself part of the contract.
"""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path
from typing import Any

import pytest
from tensorfs import LocalCAS

from torchcg import CallIngress, CallInput
from torchcg.artifact import build_metadata, pack_artifact, read_metadata
from torchcg.declaration import GraphSpecializationDeclaration
from torchcg.engine import Engine
from torchcg.graph_identity import EnvIdentity
from torchcg.host_isa import _host_requirement
from torchcg.requirements import RequirementsManifest
from torchcg.store import LocalGraphStore, _stamp_path

STACK = (("torch", "2.13.0"),)
ENV = EnvIdentity(stack=STACK, sm="sm_89")
TOOLCHAIN = {"torch": "record-digest", "triton": "compiler-digest"}


def _aoti_package(path: Path, name: str) -> Path:
    """``test_artifact.aoti_package``, with the package entry parameterized so
    the entry name can match an arbitrary graph-specialization name (the pgw
    ``tcg_artifacts`` shape)."""

    names = b"\0.shstrtab\0.lrodata\0"
    section_offset = 64
    section_size = 64
    section_count = 3
    string_offset = section_offset + section_size * section_count
    payload_offset = string_offset + len(names)
    shared_object = bytearray(payload_offset)
    shared_object[:4] = b"\x7fELF"
    shared_object[4:7] = bytes((2, 1, 1))
    struct.pack_into("<Q", shared_object, 0x28, section_offset)
    struct.pack_into("<HHH", shared_object, 0x3A, section_size, section_count, 1)
    struct.pack_into("<II", shared_object, section_offset + section_size, 1, 3)
    struct.pack_into(
        "<QQ", shared_object, section_offset + section_size + 0x18, string_offset, len(names)
    )
    struct.pack_into("<II", shared_object, section_offset + 2 * section_size, 11, 1)
    struct.pack_into(
        "<QQ", shared_object, section_offset + 2 * section_size + 0x18, payload_offset, 0
    )
    shared_object[string_offset:payload_offset] = names
    wrapper = "AOTInductorModelBase(1, 1, 0, device_str, std::move(cubin_dir), false)"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        root = f"data/aotinductor/{name}"
        archive.writestr(f"{root}/model.wrapper.cpp", wrapper)
        archive.writestr(f"{root}/model.so", bytes(shared_object))
    return path


def _metadata(name: str, *, sm: str = "sm_89") -> dict[str, object]:
    ingress = CallIngress(
        parameters=("value",),
        flat_arity=1,
        inputs=(CallInput("value", 0, "value", 0, (), "value", "float32", (2,)),),
    )
    graph: dict[str, object] = {
        "v": 4,
        "constant_fqns": [],
        "ingress": ingress.as_dict(),
    }
    declaration = GraphSpecializationDeclaration(
        name=name,
        target="unet",
        graph=graph,
        graph_witness="fedcba9876543210",
        range_digest=ingress.digest(),
        literal_values="",
    )
    return build_metadata(
        graph_specialization={
            "name": declaration.name,
            "target": declaration.target,
            "specialization_hash": declaration.specialization_hash,
            "graph": dict(declaration.graph),
            "graph_witness": declaration.graph_witness,
            "range_digest": declaration.range_digest,
            "fork": [],
            "specialization_dims": [],
            "strict": True,
            "lora_bucket": 0,
            "literal_values": "",
            "literal_payload_values": "",
            "placement": list(declaration.placement),
            "constants": [],
        },
        sm=sm,
        toolchain=dict(TOOLCHAIN),
        # This machine's OWN facts (the tcg_artifacts pattern): a hard-coded
        # x86-64-v3 would make admissibility depend on which runner ran this.
        host_isa=_host_requirement().facts(),
    )


def _artifact(tmp_path: Path, name: str, *, sm: str = "sm_89") -> tuple[Path, str]:
    """One real envelope stamped with ``name`` and the exact key it states."""

    package = _aoti_package(tmp_path / f".{name.replace('/', '_')}.pt2", name)
    built = pack_artifact(
        package, tmp_path / f"{name.replace('/', '_')}.tar.gz", _metadata(name, sm=sm)
    )
    key = read_metadata(built)["compiled_graph_key"]
    assert isinstance(key, str)
    return built, key


# --------------------------------------------------------------------------
# Engine.reuse_index
# --------------------------------------------------------------------------


def test_reuse_index_names_exactly_the_minted_axis(tmp_path: Path) -> None:
    engine = Engine(LocalCAS(tmp_path / "cas"))
    first, first_key = _artifact(tmp_path, "cg-graph-v1-" + "a" * 56)
    other_sm, _ = _artifact(tmp_path, "cg-graph-v1-" + "b" * 56, sm="sm_86")
    engine.import_artifact(first_key, first)
    engine.import_artifact(read_metadata(other_sm)["compiled_graph_key"], other_sm)

    index = engine.reuse_index("sm_89", TOOLCHAIN)
    assert index == {"cg-graph-v1-" + "a" * 56: first_key}

    # The key the index answers really materializes through the verified gate.
    resolved = engine.resolve(first_key, tmp_path / "out")
    assert resolved is not None and resolved.package.is_file()

    # A different sm sees only its own mint; a different toolchain sees nothing.
    assert list(engine.reuse_index("sm_86", TOOLCHAIN)) == ["cg-graph-v1-" + "b" * 56]
    assert engine.reuse_index("sm_89", {"torch": "other"}) == {}


def test_reuse_index_over_an_empty_store_is_empty(tmp_path: Path) -> None:
    assert Engine(LocalCAS(tmp_path / "cas")).reuse_index("sm_89", TOOLCHAIN) == {}


def test_a_quarantined_key_is_not_offered_for_reuse(tmp_path: Path) -> None:
    engine = Engine(LocalCAS(tmp_path / "cas"))
    built, key = _artifact(tmp_path, "cg-graph-v1-" + "a" * 56)
    result = engine.import_artifact(key, built)
    engine._store.quarantine(key, result.artifact)
    assert engine.reuse_index("sm_89", TOOLCHAIN) == {}


# --------------------------------------------------------------------------
# LocalGraphStore fetch stamps
# --------------------------------------------------------------------------


def _published(tmp_path: Path) -> tuple[LocalGraphStore, str, Path]:
    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    graph = "cg-graph-v1-" + "a" * 56
    artifact = tmp_path / "minted.bin"
    artifact.write_bytes(b"minted-bytes")
    manifest = RequirementsManifest(
        include_set=(("torch", ">=2.13.0"),), sm_compiled=ENV.sm
    )
    store.publish_artifact(graph, ENV, artifact, manifest)
    return store, graph, tmp_path / "out" / "artifact.so"


def test_a_second_fetch_of_untouched_bytes_reads_no_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The warm path: verified once, stat-checked thereafter."""

    store, graph, destination = _published(tmp_path)
    first = store.fetch_artifact(graph, ENV, destination)
    assert first is not None and first.read_bytes() == b"minted-bytes"
    assert _stamp_path(destination).is_file()

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a warm fetch re-read the store's object")

    monkeypatch.setattr(LocalCAS, "verify_object", refuse)
    before = destination.stat()
    second = store.fetch_artifact(graph, ENV, destination)
    assert second == destination
    after = destination.stat()
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns), (
        "the warm path must not re-copy either"
    )


def test_a_rewritten_destination_is_re_fetched_not_trusted(tmp_path: Path) -> None:
    """RED ARM: any rewrite changes the file identity and voids the stamp."""

    store, graph, destination = _published(tmp_path)
    store.fetch_artifact(graph, ENV, destination)
    destination.write_bytes(b"tampered")

    refetched = store.fetch_artifact(graph, ENV, destination)
    assert refetched is not None
    assert refetched.read_bytes() == b"minted-bytes"


def test_a_stamp_for_a_superseded_ref_is_ignored(tmp_path: Path) -> None:
    """RED ARM: the stamp binds destination bytes to ONE ref; a replaced
    position (programs are last-write-wins) must serve the new bytes."""

    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    graph = "cg-graph-v1-" + "a" * 56
    first = tmp_path / "first.pt2"
    first.write_bytes(b"first-program")
    second = tmp_path / "second.pt2"
    second.write_bytes(b"second-program")
    destination = tmp_path / "out" / "program.pt2"

    store.put_program(graph, first)
    fetched = store.fetch_program(graph, destination)
    assert fetched is not None and fetched.read_bytes() == b"first-program"
    store.put_program(graph, second)
    refetched = store.fetch_program(graph, destination)
    assert refetched is not None and refetched.read_bytes() == b"second-program"


def test_a_corrupt_stamp_falls_back_to_the_full_fetch(tmp_path: Path) -> None:
    store, graph, destination = _published(tmp_path)
    store.fetch_artifact(graph, ENV, destination)
    _stamp_path(destination).write_text("not json", encoding="ascii")
    refetched = store.fetch_artifact(graph, ENV, destination)
    assert refetched is not None and refetched.read_bytes() == b"minted-bytes"
    # And the fall-back re-stamps, so the path is warm again.
    assert isinstance(json.loads(_stamp_path(destination).read_bytes()), dict)
