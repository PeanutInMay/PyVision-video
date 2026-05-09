#!/usr/bin/env python
"""Summarize Qwen3-VL online agent and baseline jsonl results."""

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_OUR_METHOD_PATH = "eval/results/qwen3vl_vllm_online_agent/qwen3vl_vllm_online_agent_no_lvb.jsonl"
DEFAULT_BASELINE_DIR = "eval/results/qwen3vl_vllm_online_baselines"
DEFAULT_METHOD_PATHS = {
    "our_method": DEFAULT_OUR_METHOD_PATH,
    "direct": f"{DEFAULT_BASELINE_DIR}/baseline_direct.jsonl",
    "perception_tool": f"{DEFAULT_BASELINE_DIR}/baseline_perception_tool.jsonl",
    "perception_inline": f"{DEFAULT_BASELINE_DIR}/baseline_perception_inline.jsonl",
}


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skip malformed line {line_no} in {path}")


def extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text or "")
    return matches[-1].strip() if matches else (text or "").strip()


def normalize_answer(value: Any) -> str:
    text = str(value).strip()
    text = extract_boxed(text)
    text = text.strip().strip("'\"").strip()
    return text


def parse_float(text: str) -> float | None:
    cleaned = text.replace(",", "").strip()
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def is_correct(prediction: Any, gold: Any) -> bool:
    pred = normalize_answer(prediction)
    gt = normalize_answer(gold)
    if not pred or not gt:
        return False

    if re.fullmatch(r"[A-E]", gt, flags=re.IGNORECASE):
        return pred[:1].upper() == gt.upper()

    pred_num = parse_float(pred)
    gt_num = parse_float(gt)
    if pred_num is not None and gt_num is not None:
        return math.isclose(pred_num, gt_num, rel_tol=1e-4, abs_tol=1e-4)

    return pred.lower().strip() == gt.lower().strip()


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: List[float]) -> float:
    return statistics.median(values) if values else 0.0


def summarize_records(method: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        dataset = str(record.get("dataset", "unknown"))
        modality = str(record.get("modality", "unknown"))
        groups[(dataset, modality)].append(record)

    rows: List[Dict[str, Any]] = []
    for (dataset, modality), group in sorted(groups.items()):
        correct = sum(1 for item in group if is_correct(item.get("prediction", ""), item.get("gold_answer", "")))
        total = len(group)
        elapsed = [float(item.get("elapsed_seconds") or 0.0) for item in group]
        model_time = [float(item.get("model_request_seconds") or 0.0) for item in group]
        turns = [float(item.get("generation_turns") or 0.0) for item in group]
        perception_calls = [
            float(item.get("num_perception_calls", len(item.get("perception_calls") or [])) or 0.0)
            for item in group
        ]
        status_counts = Counter(str(item.get("status", "unknown")) for item in group)
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "modality": modality,
                "total": total,
                "correct": correct,
                "accuracy": round(correct / total, 6) if total else 0.0,
                "avg_elapsed_seconds": round(mean(elapsed), 6),
                "median_elapsed_seconds": round(median(elapsed), 6),
                "avg_model_request_seconds": round(mean(model_time), 6),
                "avg_generation_turns": round(mean(turns), 6),
                "avg_perception_calls": round(mean(perception_calls), 6),
                "status_counts": dict(status_counts),
            }
        )

    if records:
        correct = sum(1 for item in records if is_correct(item.get("prediction", ""), item.get("gold_answer", "")))
        total = len(records)
        elapsed = [float(item.get("elapsed_seconds") or 0.0) for item in records]
        model_time = [float(item.get("model_request_seconds") or 0.0) for item in records]
        turns = [float(item.get("generation_turns") or 0.0) for item in records]
        perception_calls = [
            float(item.get("num_perception_calls", len(item.get("perception_calls") or [])) or 0.0)
            for item in records
        ]
        rows.append(
            {
                "method": method,
                "dataset": "ALL",
                "modality": "ALL",
                "total": total,
                "correct": correct,
                "accuracy": round(correct / total, 6) if total else 0.0,
                "avg_elapsed_seconds": round(mean(elapsed), 6),
                "median_elapsed_seconds": round(median(elapsed), 6),
                "avg_model_request_seconds": round(mean(model_time), 6),
                "avg_generation_turns": round(mean(turns), 6),
                "avg_perception_calls": round(mean(perception_calls), 6),
                "status_counts": dict(Counter(str(item.get("status", "unknown")) for item in records)),
            }
        )
    return rows


def format_markdown(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "method",
        "dataset",
        "modality",
        "total",
        "correct",
        "accuracy",
        "avg_elapsed_seconds",
        "median_elapsed_seconds",
        "avg_model_request_seconds",
        "avg_generation_turns",
        "avg_perception_calls",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def format_tsv(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "method",
        "dataset",
        "modality",
        "total",
        "correct",
        "accuracy",
        "avg_elapsed_seconds",
        "median_elapsed_seconds",
        "avg_model_request_seconds",
        "avg_generation_turns",
        "avg_perception_calls",
        "status_counts",
    ]
    lines = ["\t".join(headers)]
    for row in rows:
        lines.append("\t".join(json.dumps(row.get(header, ""), ensure_ascii=False) for header in headers))
    return "\n".join(lines)


def parse_method_paths(values: List[str]) -> Dict[str, str]:
    paths = dict(DEFAULT_METHOD_PATHS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"--method-path must be NAME=PATH, got: {value}")
        name, path = value.split("=", 1)
        paths[name.strip()] = path.strip()
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-path", action="append", default=[], help="Override/add result path as NAME=PATH.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-tsv", default=None)
    parser.add_argument("--skip-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    method_paths = parse_method_paths(args.method_path)
    all_rows: List[Dict[str, Any]] = []

    for method, raw_path in method_paths.items():
        path = Path(raw_path)
        if not path.exists():
            message = f"Result path not found for {method}: {path}"
            if args.skip_missing:
                print(f"Warning: {message}")
                continue
            raise FileNotFoundError(message)
        records = list(iter_jsonl(path))
        all_rows.extend(summarize_records(method, records))

    json_text = json.dumps(all_rows, ensure_ascii=False, indent=2)
    md_text = format_markdown(all_rows)
    tsv_text = format_tsv(all_rows)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json_text + "\n", encoding="utf-8")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(md_text + "\n", encoding="utf-8")
    if args.output_tsv:
        Path(args.output_tsv).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_tsv).write_text(tsv_text + "\n", encoding="utf-8")

    print(md_text)


if __name__ == "__main__":
    main()
