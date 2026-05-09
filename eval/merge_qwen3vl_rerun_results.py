#!/usr/bin/env python
"""Merge rerun jsonl records into an existing Qwen3-VL eval jsonl."""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_BASE_PATH = "eval/results/qwen3vl_vllm_online_agent/qwen3vl_vllm_online_agent_no_lvb.jsonl"
DEFAULT_RERUN_GLOB = "eval/results/qwen3vl_vllm_online_agent/rerun_no_lvb_splits/ours_rerun_*.jsonl"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed jsonl at {path}:{line_no}") from exc


def record_key(record: Dict[str, Any]) -> Tuple[str, str]:
    return str(record.get("dataset", "")), str(record.get("question_id", ""))


def load_replacements(paths: List[Path]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    replacements: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for path in paths:
        for record in iter_jsonl(path):
            key = record_key(record)
            if not all(key):
                continue
            replacements[key] = record
    return replacements


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    parser.add_argument("--rerun-glob", default=DEFAULT_RERUN_GLOB)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_path)
    rerun_paths = sorted(Path().glob(args.rerun_glob))
    if not base_path.exists():
        raise FileNotFoundError(f"Base result not found: {base_path}")
    if not rerun_paths:
        raise FileNotFoundError(f"No rerun files matched: {args.rerun_glob}")

    base_records = list(iter_jsonl(base_path))
    replacements = load_replacements(rerun_paths)
    merged_records = []
    replaced = 0
    for record in base_records:
        key = record_key(record)
        if key in replacements:
            merged_records.append(replacements[key])
            replaced += 1
        else:
            merged_records.append(record)

    if args.in_place:
        output_path = base_path
        if not args.no_backup:
            backup_path = base_path.with_suffix(base_path.suffix + f".bak_{time.strftime('%Y%m%d_%H%M%S')}")
            shutil.copy2(base_path, backup_path)
            print(f"Backup written to {backup_path}")
    elif args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = base_path.with_suffix(base_path.suffix + ".merged")

    write_jsonl(output_path, merged_records)
    unused = len(set(replacements) - {record_key(record) for record in base_records})
    print(
        json.dumps(
            {
                "base_path": str(base_path),
                "output_path": str(output_path),
                "rerun_files": [str(path) for path in rerun_paths],
                "base_records": len(base_records),
                "replacement_records": len(replacements),
                "replaced_records": replaced,
                "unused_replacements": unused,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
