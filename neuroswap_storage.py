import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable


@dataclass(frozen=True)
class S3ObjectRef:
    key: str
    last_modified_ts: float
    size: int


def is_s3_enabled() -> bool:
    return (os.environ.get("NEUROSWAP_STORAGE") or "").strip().lower() == "s3"


def _require_s3_enabled():
    if not is_s3_enabled():
        raise RuntimeError("S3 storage is not enabled (set NEUROSWAP_STORAGE=s3)")


def _bucket() -> str:
    b = (os.environ.get("NEUROSWAP_S3_BUCKET") or "").strip()
    if not b:
        raise RuntimeError("Missing NEUROSWAP_S3_BUCKET")
    return b


def _prefix() -> str:
    p = (os.environ.get("NEUROSWAP_S3_PREFIX") or "neuroswap").strip()
    p = p.strip("/")
    return p


def _kind_prefix(kind: str) -> str:
    k = (kind or "").strip().lower()
    if k not in {"uploads", "videos", "outputs"}:
        raise RuntimeError(f"Invalid S3 kind: {kind}")
    return k


def make_key(kind: str, name: str) -> str:
    name = (name or "").strip()
    if not name or "/" in name or "\\" in name:
        raise RuntimeError("Invalid object name")
    return f"{_prefix()}/{_kind_prefix(kind)}/{name}"


@lru_cache(maxsize=1)
def _s3_client():
    _require_s3_enabled()
    try:
        import boto3
    except Exception as e:
        raise RuntimeError(f"boto3 missing or broken: {e}")
    return boto3.client("s3")


def put_bytes(kind: str, name: str, data: bytes, content_type: str | None = None) -> str:
    _require_s3_enabled()
    key = make_key(kind, name)
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    _s3_client().put_object(Bucket=_bucket(), Key=key, Body=data, **extra)
    return key


def put_file(kind: str, name: str, file_path: str, content_type: str | None = None) -> str:
    _require_s3_enabled()
    key = make_key(kind, name)
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    _s3_client().upload_file(file_path, _bucket(), key, ExtraArgs=extra or None)
    return key


def download_file(kind: str, name: str, dst_path: str) -> str:
    _require_s3_enabled()
    key = make_key(kind, name)
    _s3_client().download_file(_bucket(), key, dst_path)
    return dst_path


def get_stream(kind: str, name: str):
    _require_s3_enabled()
    key = make_key(kind, name)
    return _s3_client().get_object(Bucket=_bucket(), Key=key)


def delete_object(kind: str, name: str) -> str:
    _require_s3_enabled()
    key = make_key(kind, name)
    _s3_client().delete_object(Bucket=_bucket(), Key=key)
    return key


def list_objects(kind: str, limit: int) -> list[S3ObjectRef]:
    _require_s3_enabled()
    limit = int(limit or 0)
    if limit <= 0:
        return []
    client = _s3_client()
    prefix = f"{_prefix()}/{_kind_prefix(kind)}/"
    out: list[S3ObjectRef] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=_bucket(), Prefix=prefix):
        for it in (page.get("Contents") or []):
            key = str(it.get("Key") or "")
            if not key or not key.startswith(prefix):
                continue
            lm = it.get("LastModified")
            try:
                ts = float(lm.timestamp()) if lm is not None else 0.0
            except Exception:
                ts = 0.0
            try:
                sz = int(it.get("Size") or 0)
            except Exception:
                sz = 0
            out.append(S3ObjectRef(key=key, last_modified_ts=ts, size=sz))
            if len(out) >= limit * 3:
                break
        if len(out) >= limit * 3:
            break
    out.sort(key=lambda x: x.last_modified_ts, reverse=True)
    return out[:limit]


def key_to_name(kind: str, key: str) -> str | None:
    _require_s3_enabled()
    prefix = f"{_prefix()}/{_kind_prefix(kind)}/"
    if not key.startswith(prefix):
        return None
    name = key[len(prefix):]
    if not name or "/" in name or "\\" in name:
        return None
    return name
