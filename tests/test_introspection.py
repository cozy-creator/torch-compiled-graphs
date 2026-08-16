from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import pytest

from torchcg.introspection import (
    DeclaredConstant,
    PackageIntrospectionError,
    _constants_in_so,
    _elf_section_sizes_from_bytes,
    _package_entry_names,
    code_only_violations,
    declared_constants,
)


def elf(*, lrodata: int = 0, duplicate_lrodata: int | None = None) -> bytes:
    names = b"\0.shstrtab\0.lrodata\0"
    section_offset = 64
    section_size = 64
    lrodata_sizes = (lrodata,) if duplicate_lrodata is None else (lrodata, duplicate_lrodata)
    section_count = 2 + len(lrodata_sizes)
    string_offset = section_offset + section_size * section_count
    payload_offset = string_offset + len(names)
    blob = bytearray(payload_offset + sum(lrodata_sizes))
    blob[:4] = b"\x7fELF"
    blob[4:7] = bytes((2, 1, 1))
    struct.pack_into("<Q", blob, 0x28, section_offset)
    struct.pack_into("<HHH", blob, 0x3A, section_size, section_count, 1)
    struct.pack_into("<II", blob, section_offset + section_size, 1, 3)
    struct.pack_into("<QQ", blob, section_offset + section_size + 0x18, string_offset, len(names))
    next_payload = payload_offset
    for index, size in enumerate(lrodata_sizes, start=2):
        header = section_offset + index * section_size
        struct.pack_into("<II", blob, header, 11, 1)
        struct.pack_into("<QQ", blob, header + 0x18, next_payload, size)
        next_payload += size
    blob[string_offset : string_offset + len(names)] = names
    return bytes(blob)


def wrapper(*, baked: bool, name: str = "weight", size: int = 16) -> str:
    flag = str(baked).lower()
    return f"""
constants_info_[0].name = "{name.replace(".", "_")}";
constants_info_[0].original_fqn = "{name}";
constants_info_[0].data_size = {size};
constants_info_[0].from_folded = false;
constants_info_[0].type = static_cast<int32_t>(
    torch::aot_inductor::ConstantType::Parameter);
constants_info_[0].dtype = cached_torch_dtype_float32;
constants_info_[0].shape = {{2, 2}};
AOTInductorModelBase(1, 1, 1, device_str, std::move(cubin_dir), {flag})
"""


def empty_wrapper(*, baked: bool = False) -> str:
    return f"AOTInductorModelBase(1, 1, 0, device_str, std::move(cubin_dir), {str(baked).lower()})"


def package(tmp_path: Path, entries: dict[str, tuple[str, bytes]]) -> Path:
    target = tmp_path / "model.pt2"
    with zipfile.ZipFile(target, "w") as archive:
        for entry, (source, shared_object) in entries.items():
            root = f"data/aotinductor/{entry}"
            archive.writestr(f"{root}/model.wrapper.cpp", source)
            archive.writestr(f"{root}/model.so", shared_object)
    return target


def test_declared_constants_parse_every_constant_field(tmp_path: Path) -> None:
    target = package(tmp_path, {"denoiser": (wrapper(baked=False, name="block.weight"), elf())})

    constants = declared_constants(target)
    assert constants == (
        DeclaredConstant(
            index=0,
            name="block_weight",
            fqn="block.weight",
            data_size=16,
            kind="Parameter",
            dtype="float32",
            shape=(2, 2),
            from_folded=False,
        ),
    )
    assert constants[0].source == "state_dict"
    assert constants[0].as_manifest_row() == {
        "fqn": "block.weight",
        "source": "state_dict",
        "dtype": "float32",
        "shape": [2, 2],
    }
    assert _constants_in_so(target) is False


def test_named_entries_are_discovered_and_scoped_as_whole_paths(tmp_path: Path) -> None:
    first_so = elf(lrodata=8)
    second_so = elf(lrodata=32)
    target = package(
        tmp_path,
        {
            "decoder": (wrapper(baked=False, name="decoder.weight", size=64), first_so),
            "denoiser/h=64,w=64": (
                wrapper(baked=True, name="denoiser.weight", size=16),
                second_so,
            ),
        },
    )

    assert _package_entry_names(target) == ("decoder", "denoiser/h=64,w=64")
    assert declared_constants(target, "decoder")[0].fqn == "decoder.weight"
    assert _constants_in_so(target, "denoiser/h=64,w=64") is True
    with pytest.raises(PackageIntrospectionError, match="exactly one"):
        declared_constants(target)


def test_entry_scope_requires_the_exact_canonical_full_path(tmp_path: Path) -> None:
    target = package(
        tmp_path,
        {"denoiser/variant": (wrapper(baked=False), elf())},
    )

    assert _package_entry_names(target) == ("denoiser/variant",)
    with pytest.raises(PackageIntrospectionError, match="exactly one"):
        declared_constants(target, "denoiser")

    torch_prefixed = tmp_path / "prefixed.pt2"
    with zipfile.ZipFile(torch_prefixed, "w") as archive:
        archive.writestr("model/data/aotinductor/decoder/model.wrapper.cpp", wrapper(baked=False))
    assert _package_entry_names(torch_prefixed) == ("decoder",)

    nested_prefix = tmp_path / "nested-prefix.pt2"
    with zipfile.ZipFile(nested_prefix, "w") as archive:
        archive.writestr(
            "outer/inner/data/aotinductor/model/model.wrapper.cpp", wrapper(baked=False)
        )
    assert _package_entry_names(nested_prefix) == ()


def test_constant_table_must_match_count_and_carry_every_required_field(tmp_path: Path) -> None:
    missing_row = package(
        tmp_path,
        {"model": (empty_wrapper().replace(", 0,", ", 1,"), elf())},
    )
    with pytest.raises(PackageIntrospectionError, match="declared count 1"):
        declared_constants(missing_row)

    incomplete = package(
        tmp_path,
        {"model": (wrapper(baked=False).replace("constants_info_[0].data_size = 16;", ""), elf())},
    )
    with pytest.raises(PackageIntrospectionError, match="missing required fields.*data_size"):
        declared_constants(incomplete)


def test_genuine_zero_constant_package_is_not_parser_drift(tmp_path: Path) -> None:
    target = package(tmp_path, {"model": (empty_wrapper(), elf())})

    assert declared_constants(target) == ()
    assert _constants_in_so(target) is False


def test_wrapper_decode_and_crc_errors_are_normalized(tmp_path: Path) -> None:
    invalid_utf8 = tmp_path / "invalid-utf8.pt2"
    with zipfile.ZipFile(invalid_utf8, "w") as archive:
        archive.writestr("data/aotinductor/model/model.wrapper.cpp", b"\xff")
    with pytest.raises(PackageIntrospectionError, match="not UTF-8"):
        declared_constants(invalid_utf8)

    corrupt = package(tmp_path, {"model": (wrapper(baked=False), elf())})
    raw = bytearray(corrupt.read_bytes())
    payload = raw.find(b"constants_info_")
    assert payload >= 0
    raw[payload] ^= 1
    corrupt.write_bytes(raw)
    with pytest.raises(PackageIntrospectionError, match="cannot read package member"):
        declared_constants(corrupt)


def test_elf_section_sizes_read_little_endian_elf_without_binutils() -> None:
    assert _elf_section_sizes_from_bytes(elf(lrodata=123))[".lrodata"] == 123


def test_elf_section_sizes_aggregate_duplicate_section_names() -> None:
    assert _elf_section_sizes_from_bytes(elf(lrodata=12, duplicate_lrodata=30))[".lrodata"] == 42


@pytest.mark.parametrize(
    "blob, message",
    [
        (b"not an elf", "not an ELF"),
        (b"\x7fELF" + bytes((1, 1)) + bytes(58), "not 64-bit"),
        (b"\x7fELF" + bytes((2, 1, 1)) + bytes(57), "section table"),
    ],
)
def test_elf_section_sizes_refuse_malformed_images(blob: bytes, message: str) -> None:
    with pytest.raises(PackageIntrospectionError, match=message):
        _elf_section_sizes_from_bytes(blob)


def test_elf_section_sizes_validate_string_and_file_backed_sections() -> None:
    wrong_string_type = bytearray(elf())
    struct.pack_into("<I", wrong_string_type, 64 + 64 + 4, 1)
    with pytest.raises(PackageIntrospectionError, match="section-name table"):
        _elf_section_sizes_from_bytes(bytes(wrong_string_type))

    out_of_range = bytearray(elf(lrodata=8))
    struct.pack_into("<Q", out_of_range, 64 + 2 * 64 + 0x18, len(out_of_range) + 1)
    with pytest.raises(PackageIntrospectionError, match="file range"):
        _elf_section_sizes_from_bytes(bytes(out_of_range))

    non_file_backed_lrodata = bytearray(elf(lrodata=8))
    struct.pack_into("<I", non_file_backed_lrodata, 64 + 2 * 64 + 4, 8)
    with pytest.raises(PackageIntrospectionError, match="unexpected ELF section type"):
        _elf_section_sizes_from_bytes(bytes(non_file_backed_lrodata))


def test_code_only_gate_accepts_external_constants(tmp_path: Path) -> None:
    target = package(tmp_path, {"model": (wrapper(baked=False, size=16), elf(lrodata=8))})

    assert code_only_violations(target) == []


def test_code_only_gate_reports_baked_flag_and_suspicious_lrodata(tmp_path: Path) -> None:
    target = package(
        tmp_path,
        {"model": (wrapper(baked=True, name="large.weight", size=16), elf(lrodata=16))},
    )

    violations = code_only_violations(target)
    assert len(violations) == 2
    assert "load_constants_from_blob=true" in violations[0]
    assert "large.weight (16B)" in violations[0]
    assert ".lrodata is 16B" in violations[1]
