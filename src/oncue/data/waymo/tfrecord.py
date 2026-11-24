"""Read uncompressed TFRecord shards without TensorFlow, verifying their CRC32C checksums."""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import BinaryIO

MAX_RECORD_BYTES = 512 * 1024**2
# Record layout: length (uint64), length's masked CRC32C (uint32), payload, payload CRC.
LENGTH_BYTES = 8
CRC_BYTES = 4
HEADER_BYTES = 12
RECORD_OVERHEAD_BYTES = 16
# TensorFlow's crc32c::Mask: rotate the CRC right by 15 bits and add this delta, mod 2**32.
CRC_MASK_DELTA = 0xA282EAD8
UINT32_MASK = 0xFFFFFFFF


def index_records(path: str | Path, limit: int | None = None) -> list[RecordRef]:
    """Return record locations from a shard, verifying each length CRC and skipping payloads."""
    path = str(Path(path).resolve())
    size = Path(path).stat().st_size
    refs: list[RecordRef] = []
    with open(path, "rb") as f:
        while limit is None or len(refs) < limit:
            offset = f.tell()
            header = f.read(HEADER_BYTES)
            if not header:
                break
            if len(header) != HEADER_BYTES:
                raise ValueError(f"{path}: truncated header at record {len(refs)}")
            length, crc = struct.unpack("<QI", header)
            if (
                crc != _masked_crc(header[:LENGTH_BYTES])
                or offset + RECORD_OVERHEAD_BYTES + length > size
            ):
                raise ValueError(f"{path}: corrupt or truncated record {len(refs)}")
            refs.append(RecordRef(path, len(refs), offset, length))
            f.seek(length + CRC_BYTES, 1)
    return refs


@dataclass(frozen=True)
class RecordRef:
    path: str
    index: int
    offset: int
    length: int


def _masked_crc(data: bytes) -> int:
    try:
        import google_crc32c
    except ImportError as exc:
        raise RuntimeError("Install oncue[data] to read TFRecords") from exc
    value = google_crc32c.value(data)
    return (((value >> 15) | (value << 17)) + CRC_MASK_DELTA) & UINT32_MASK


def read_record(ref: RecordRef) -> bytes:
    """Read the payload referenced by ``ref``; raise if the record changed since indexing."""
    with open(ref.path, "rb") as f:
        f.seek(ref.offset)
        item = _read_one(f, ref.index, ref.path)
    if item is None or item[0].length != ref.length:
        raise ValueError(f"{ref.path}: record {ref.index} changed since it was indexed")
    return item[1]


def _read_one(f: BinaryIO, index: int, path: str) -> tuple[RecordRef, bytes] | None:
    """Read one record, verifying both CRCs; return ``None`` at EOF."""
    offset = f.tell()
    header = f.read(HEADER_BYTES)
    if not header:
        return None
    if len(header) != HEADER_BYTES:
        raise ValueError(f"{path}: truncated header at record {index}")
    length, crc = struct.unpack("<QI", header)
    if crc != _masked_crc(header[:LENGTH_BYTES]) or length > MAX_RECORD_BYTES:
        raise ValueError(f"{path}: corrupt length at record {index}")
    payload, footer = f.read(length), f.read(CRC_BYTES)
    if len(payload) != length or len(footer) != CRC_BYTES:
        raise ValueError(f"{path}: truncated record {index}")
    if struct.unpack("<I", footer)[0] != _masked_crc(payload):
        raise ValueError(f"{path}: payload CRC mismatch at record {index}")
    return RecordRef(path, index, offset, length), payload


def iter_records(path: str | Path, limit: int | None = None) -> Iterator[tuple[RecordRef, bytes]]:
    """Yield ``(ref, payload)`` for each record in ``path``, up to ``limit`` records."""
    path = str(Path(path).resolve())
    with open(path, "rb") as f:
        index = 0
        while limit is None or index < limit:
            item = _read_one(f, index, path)
            if item is None:
                return
            yield item
            index += 1
