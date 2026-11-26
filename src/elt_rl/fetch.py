"""Bounded HTTP Range access to public upstream ZIPs; never downloads full archives."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import httpx

MAX_ASSET_BYTES = 256 * 1024 * 1024


class RangeReader(io.RawIOBase):
    def __init__(self, url: str, max_read: int = 32 * 1024 * 1024):
        self.url = url
        self.max_read = max_read
        self.client = httpx.Client(timeout=60, follow_redirects=True)
        self.position = 0
        self.size = 0
        try:
            tail, start, size = self._request("bytes=-65536")
        except Exception:
            self.client.close()
            raise
        self.size = size
        self.tail_start, self.tail = start, tail

    def _request(self, byte_range: str):
        with self.client.stream("GET", self.url, headers={"Range": byte_range}) as response:
            response.raise_for_status()
            match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("content-range", "")
            )
            if response.status_code != 206 or not match:
                raise ValueError("Server did not honor HTTP Range; refusing full archive download")
            start, end, size = map(int, match.groups())
            if end - start + 1 > self.max_read:
                raise ValueError("ZIP member exceeds download budget")
            chunks, length = [], 0
            for chunk in response.iter_bytes():
                length += len(chunk)
                if length > self.max_read:
                    raise ValueError("Range response exceeds byte budget")
                chunks.append(chunk)
            data = b"".join(chunks)
            if len(data) != end - start + 1:
                raise ValueError("Truncated range response")
            return data, start, size

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_CUR:
            offset += self.position
        elif whence == io.SEEK_END:
            offset += self.size
        if offset < 0:
            raise ValueError("Negative seek")
        self.position = offset
        return offset

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.position + size, self.size)
        if end <= self.position:
            return b""
        if end - self.position > self.max_read:
            raise ValueError("ZIP member exceeds download budget")
        if self.position >= self.tail_start:
            data = self.tail[self.position - self.tail_start : end - self.tail_start]
        else:
            data, start, total = self._request(f"bytes={self.position}-{end - 1}")
            if start != self.position or total != self.size or len(data) != end - self.position:
                raise ValueError("Server returned the wrong byte range")
        self.position = end
        return data

    def close(self):
        self.client.close()
        super().close()


def public_archive(file_id: str):
    return RangeReader(
        f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    )


def fetch_assets(repo: Path, task: str, output: Path) -> Path:
    from elt_rl.benchmark import ORIGINAL_REVISION, task_material

    files, lock = task_material(repo, task)
    from elt_rl.registry import UnsupportedCapability
    from elt_rl.task_adapters import pack_definitions

    definition = pack_definitions().get(task)
    if definition is None:
        raise UnsupportedCapability("Register the original asset members for this task first")
    script = files["setup/elt_setup.sh"].decode()
    archives = definition["archives"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for archive_id, members in archives.items():
        if archive_id not in script:
            raise ValueError("Asset archive is not linked by the pinned original setup")
        with public_archive(archive_id) as reader, zipfile.ZipFile(reader) as archive:
            for member, destination in members.items():
                info = archive.getinfo(member)
                if info.file_size > MAX_ASSET_BYTES:
                    raise ValueError("Inflated ZIP member exceeds the asset budget")
                with archive.open(info) as stream:
                    data = stream.read(MAX_ASSET_BYTES + 1)
                if len(data) > MAX_ASSET_BYTES or len(data) != info.file_size:
                    raise ValueError("Inflated ZIP member size is invalid")
                path = output / destination
                path.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256(data).hexdigest()
                if path.exists() and path.read_bytes() != data:
                    raise ValueError(
                        "Existing asset bytes disagree with the original-linked archive"
                    )
                path.write_bytes(data)
                records.append(
                    {
                        "archive_id": archive_id,
                        "member": member,
                        "path": destination,
                        "sha256": digest,
                        "bytes": len(data),
                    }
                )
    base = {
        "source": "official original setup/elt_setup.sh",
        "revision": ORIGINAL_REVISION,
        "retrieval": "bounded HTTP ranges from original linked mutable Drive archives; exact retrieved bytes frozen",
    }
    for name, prefix in [
        ("source-provenance.json", "sources/"),
        ("reference-provenance.json", "references/"),
        ("healthcheck-provenance.json", "healthcheck/"),
    ]:
        (output / name).write_text(
            json.dumps(
                {**base, "files": [r for r in records if r["path"].startswith(prefix)]}, indent=2
            )
            + "\n"
        )
    return output
