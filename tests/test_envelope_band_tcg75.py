"""tcg#75 / pgw#1561: the store's artifact band holds ENVELOPES, and a bare
package is a NAMED skew, not "corruption".

Field shape this encodes (va#3 arm 2, 2026-08-20): every v2 artifact position
held the bare AOTI ``.pt2`` ZIP a pre-envelope publisher banked, the loader
gzip-decoded a ZIP, and 14 adoptions holed as ``cannot decompress`` — a
message whose only reading was bytes rotting at rest. The remedy for a skew
is re-publish; for corruption it is re-mint; so the skew carries its own type
and the publish path can REPLACE a skewed incumbent with a real envelope.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from tensorfs import LocalCAS
from test_artifact import aoti_package, metadata

from torchcg import ArtifactError, ArtifactFormatSkew
from torchcg.artifact import pack_artifact, read_metadata
from torchcg.graph_identity import EnvIdentity
from torchcg.requirements import RequirementsManifest
from torchcg.serve import materialize
from torchcg.store import LocalGraphStore, PublishOutcome, StoreError

STACK = (("torch", "2.13.0"),)
ENV = EnvIdentity(stack=STACK, sm="sm_89")
GRAPH = "cg-graph-v1-" + "a" * 56


def _manifest() -> RequirementsManifest:
    return RequirementsManifest(
        include_set=(("torch", ">=2.13.0"),), sm_compiled=ENV.sm
    )


def _envelope(tmp_path: Path, name: str = "envelope") -> Path:
    package = aoti_package(tmp_path / f".{name}.pt2")
    return pack_artifact(package, tmp_path / f"{name}.tar.gz", metadata())


def _bare_package(tmp_path: Path, name: str = "bare") -> Path:
    """What the pre-#1561 publisher banked: the AOTI ZIP, no envelope."""
    return aoti_package(tmp_path / f"{name}.pt2")


# --------------------------------------------------------------------------
# The skew is NAMED — reader side
# --------------------------------------------------------------------------


def test_a_bare_package_is_a_format_skew_not_corruption(tmp_path: Path) -> None:
    bare = _bare_package(tmp_path)
    with pytest.raises(ArtifactFormatSkew) as refusal:
        read_metadata(bare)
    message = str(refusal.value)
    assert "bare AOTI .pt2 package" in message
    assert "Re-publish" in message, "the skew's remedy is stated, not implied"
    # And it still IS an ArtifactError, so every existing catch stays narrow
    # and every existing hole path keeps working — with a sharper type name.
    assert isinstance(refusal.value, ArtifactError)


def test_materialize_names_the_skew_for_the_adopt_path(tmp_path: Path) -> None:
    """``AdoptSession._arm`` formats holes as ``{type name}: {message}``, so
    the type raised here IS the distinguishable hole reason."""
    bare = _bare_package(tmp_path)
    with pytest.raises(ArtifactFormatSkew):
        materialize(bare, tmp_path / "workspace")


def test_truncated_gzip_is_still_ordinary_corruption(tmp_path: Path) -> None:
    envelope = _envelope(tmp_path)
    clipped = tmp_path / "clipped.tar.gz"
    clipped.write_bytes(envelope.read_bytes()[:100])
    with pytest.raises(ArtifactError) as refusal:
        read_metadata(clipped)
    assert not isinstance(refusal.value, ArtifactFormatSkew)


# --------------------------------------------------------------------------
# The publish migrates — writer side
# --------------------------------------------------------------------------


def test_an_envelope_replaces_a_skewed_incumbent(tmp_path: Path) -> None:
    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    bare = _bare_package(tmp_path)
    assert store.publish_artifact(GRAPH, ENV, bare, _manifest()) is (
        PublishOutcome.PUBLISHED
    )

    envelope = _envelope(tmp_path)
    outcome = store.publish_artifact(GRAPH, ENV, envelope, _manifest())
    assert outcome is PublishOutcome.PUBLISHED, "a skewed incumbent is replaced"

    fetched = store.fetch_artifact(GRAPH, ENV, tmp_path / "out" / "artifact")
    assert fetched is not None
    assert fetched.read_bytes() == envelope.read_bytes()
    # The replaced position round-trips through the REAL reader.
    materialized = materialize(fetched, tmp_path / "materialized")
    assert materialized.package.is_file()

    # And the repair is idempotent: the same envelope again is PRESENT.
    assert store.publish_artifact(GRAPH, ENV, envelope, _manifest()) is (
        PublishOutcome.PRESENT
    )


def test_envelope_divergence_still_refuses(tmp_path: Path) -> None:
    """RED ARM for the migration: it must not have widened first-wins."""
    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    first = _envelope(tmp_path, "first")
    package = aoti_package(tmp_path / ".second.pt2")
    with zipfile.ZipFile(package, "a") as archive:
        # A ZIP comment changes the bytes without changing one derivable fact,
        # which is exactly TCG's DIVERGENT case.
        archive.comment = b"diverged"
    second = pack_artifact(package, tmp_path / "second.tar.gz", metadata())
    assert first.read_bytes() != second.read_bytes()

    store.publish_artifact(GRAPH, ENV, first, _manifest())
    with pytest.raises(StoreError, match="diverged"):
        store.publish_artifact(GRAPH, ENV, second, _manifest())


def test_a_bare_package_never_replaces_an_envelope(tmp_path: Path) -> None:
    """RED ARM: the migration is one-directional."""
    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    envelope = _envelope(tmp_path)
    store.publish_artifact(GRAPH, ENV, envelope, _manifest())
    with pytest.raises(StoreError, match="diverged"):
        store.publish_artifact(GRAPH, ENV, _bare_package(tmp_path), _manifest())


def test_pack_artifact_is_deterministic(tmp_path: Path) -> None:
    """The repack-on-publish design (pgw#1561) rests on this: two packs of the
    same members are byte-identical, so a re-publish is PRESENT, never a
    divergence, and the v2 envelope dedups with the engine cache's object."""
    first = _envelope(tmp_path, "one")
    second = _envelope(tmp_path, "two")
    assert first.read_bytes() == second.read_bytes()


def test_artifact_skew_probe_reads_two_bytes_worth_of_truth(tmp_path: Path) -> None:
    """The census-grade probe: names a skewed position, stays silent on a
    healthy or absent one, and never decodes anything."""
    store = LocalGraphStore(LocalCAS(tmp_path / "cas"))
    assert store.artifact_skew(GRAPH, ENV) is None, "absent is has_artifact's question"

    store.publish_artifact(GRAPH, ENV, _bare_package(tmp_path), _manifest())
    skew = store.artifact_skew(GRAPH, ENV)
    assert skew is not None and "bare AOTI .pt2 package" in skew

    envelope = _envelope(tmp_path)
    store.publish_artifact(GRAPH, ENV, envelope, _manifest())
    assert store.artifact_skew(GRAPH, ENV) is None, "the repair silences the probe"


def test_a_zip_that_is_not_even_aoti_is_still_the_skew_type(tmp_path: Path) -> None:
    """The sniff is on the container format, not on AOTI internals: any ZIP in
    the envelope band is a publisher wiring defect."""
    stray = tmp_path / "stray.zip"
    with zipfile.ZipFile(stray, "w") as archive:
        archive.writestr("whatever.txt", "not an artifact")
    with pytest.raises(ArtifactFormatSkew):
        read_metadata(stray)
