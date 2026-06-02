#!/usr/bin/env python3
"""
Upload local dataset files to MinIO (S3-compatible store used by MLflow).

Reads credentials from mlflow/.env (same as docker compose). Talks to MinIO on
localhost — no docker exec into mc, no AWS CLI exports.

Examples:
    python scripts/upload_datasets_s3.py
    python scripts/upload_datasets_s3.py --tag v2
    python scripts/upload_datasets_s3.py --src outputs/datasets --dry-run
    python scripts/upload_datasets_s3.py --list
    python scripts/upload_datasets_s3.py --list --prefix chronos/datasets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / "mlflow" / ".env"
DEFAULT_SRC = REPO_ROOT / "outputs" / "datasets"
DEFAULT_PREFIX = "chronos/datasets"


def load_config(env_file: Path) -> dict[str, str]:
    if not env_file.is_file():
        raise FileNotFoundError(
            f"Missing {env_file}. Run: cp mlflow/.env.example mlflow/.env"
        )
    raw = dotenv_values(env_file)
    cfg = {k: v for k, v in raw.items() if v is not None}
    for key in ("MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD", "DEFAULT_BUCKET_NAME"):
        if key not in cfg:
            raise KeyError(f"{key} is not set in {env_file}")
    return cfg


def s3_endpoint(cfg: dict[str, str]) -> str:
    port = cfg.get("S3_API_PORT", "9000")
    return cfg.get("MLFLOW_S3_ENDPOINT_URL") or f"http://localhost:{port}"


def make_client(cfg: dict[str, str]):
    return boto3.client(
        "s3",
        endpoint_url=s3_endpoint(cfg),
        aws_access_key_id=cfg["MINIO_ROOT_USER"],
        aws_secret_access_key=cfg["MINIO_ROOT_PASSWORD"],
        region_name="us-east-1",
    )


def ensure_bucket(s3, bucket: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code not in ("404", "NoSuchBucket", "NotFound"):
            raise
        s3.create_bucket(Bucket=bucket)
        print(f"Created bucket: {bucket}")


def upload_tree(
    s3,
    bucket: str,
    s3_prefix: str,
    local_dir: Path,
    *,
    dry_run: bool = False,
) -> int:
    if not local_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {local_dir}")

    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    if not files:
        print(f"No files under {local_dir}")
        return 0

    s3_prefix = s3_prefix.strip("/")
    n = 0
    for path in files:
        rel = path.relative_to(local_dir).as_posix()
        key = f"{s3_prefix}/{rel}"
        if dry_run:
            print(f"  [dry-run] s3://{bucket}/{key}  <=  {path}")
        else:
            s3.upload_file(str(path), bucket, key)
            print(f"  s3://{bucket}/{key}")
        n += 1
    return n


def list_prefix(s3, bucket: str, prefix: str) -> int:
    prefix = prefix.strip("/")
    if prefix:
        prefix = prefix + "/"

    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            print(f"  s3://{bucket}/{obj['Key']}  ({obj['Size']} bytes)")
            n += 1
    if n == 0:
        print(f"(no objects under s3://{bucket}/{prefix})")
    return n


def cmd_upload(args: argparse.Namespace) -> int:
    cfg = load_config(args.env_file)
    bucket = cfg["DEFAULT_BUCKET_NAME"]
    src = args.src.resolve()
    tag = args.tag.strip("/")
    prefix = f"{args.prefix.strip('/')}/{tag}"

    print(f"Endpoint : {s3_endpoint(cfg)}")
    print(f"Source   : {src}/")
    print(f"Dest     : s3://{bucket}/{prefix}/")

    s3 = make_client(cfg)
    if not args.dry_run:
        ensure_bucket(s3, bucket)

    n = upload_tree(s3, bucket, prefix, src, dry_run=args.dry_run)
    print(f"{'Would upload' if args.dry_run else 'Uploaded'} {n} file(s).")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    cfg = load_config(args.env_file)
    bucket = cfg["DEFAULT_BUCKET_NAME"]
    prefix = args.prefix.strip("/")

    print(f"Endpoint : {s3_endpoint(cfg)}")
    print(f"Listing  : s3://{bucket}/{prefix}/")
    s3 = make_client(cfg)
    list_prefix(s3, bucket, prefix)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help=f"Path to mlflow .env (default: {DEFAULT_ENV_FILE.relative_to(REPO_ROOT)})",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List objects under --prefix instead of uploading",
    )
    p.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Local directory to upload")
    p.add_argument(
        "--tag",
        default=None,
        help="Version folder under prefix (default: today's UTC date YYYYMMDD)",
    )
    p.add_argument("--prefix", default=DEFAULT_PREFIX, help="S3 key prefix")
    p.add_argument("--dry-run", action="store_true", help="Print keys only, do not upload")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        return cmd_list(args)

    if args.tag is None:
        from datetime import datetime, timezone

        args.tag = datetime.now(timezone.utc).strftime("%Y%m%d")

    return cmd_upload(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, NotADirectoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
