#!/usr/bin/env python
"""Concurrent multi-turn Qwen3-VL agent evaluation through a vLLM OpenAI server."""

import argparse
import asyncio
import base64
import io
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen3vl_transformers_agent_eval import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROMPT_TEMPLATE_PATH,
    DEFAULT_VIDEOMME_PATH,
    DEFAULT_VSTAR_PATH,
    IMAGE_PROMPT_TEMPLATE_KEY,
    QWEN3_SPATIAL_FACTOR,
    VIDEO_PROMPT_TEMPLATE_KEY,
    EvalSample,
    PythonMultimodalRuntime as _BasePythonMultimodalRuntime,
    extract_code_actions,
    extract_prediction,
    get_video_info_with_decord,
    load_prompt_templates,
    load_videomme_samples,
    load_vstar_samples,
    sanitize_messages_for_debug,
    strip_generation_special_tokens,
)


DEFAULT_OUTPUT_DIR = "eval/results/qwen3vl_vllm_online_agent"
DEFAULT_HRBENCH_4K_JSONL = "/share/home/sxjiang/zhzhu/dataset/HR-Bench/extracted/hr_bench_4k/metadata.jsonl"
DEFAULT_HRBENCH_8K_JSONL = "/share/home/sxjiang/zhzhu/dataset/HR-Bench/extracted/hr_bench_8k/metadata.jsonl"
DEFAULT_LONGVIDEOBENCH_DIR = "/share/home/sxjiang/zhzhu/dataset/LongVideoBench/data"
DEFAULT_LONGVIDEOBENCH_JSON = "/share/home/sxjiang/zhzhu/dataset/LongVideoBench/val_output_file_copy.json"
DEFAULT_MATHVISTA_JSONL = "/share/home/sxjiang/zhzhu/dataset/MathVista/extracted/testmini/metadata.jsonl"
DEFAULT_MATHVISTA_IMAGE_DIR = "/share/home/sxjiang/zhzhu/dataset/MathVista/images"
DEFAULT_MATHVISION_JSONL = "/share/home/sxjiang/zhzhu/dataset/MathVision/extracted/testmini/metadata.jsonl"
DEFAULT_MATHVISION_IMAGE_DIR = "/share/home/sxjiang/zhzhu/dataset/MathVision/images"
# DEFAULT_DATASETS = "vstar,videomme,hrbench4k,hrbench8k,longvideobench,mathvista,mathvision"
DEFAULT_DATASETS = "vstar,videomme,hrbench4k,hrbench8k,mathvista,mathvision"
LONGVIDEOBENCH_SKIP_CATEGORIES = {"T3E", "T3O", "TAA", "T2E", "T2O", "T2A"}
MODEL_NAME = "qwen3-vl-thinking-8b"


class PythonMultimodalRuntime(_BasePythonMultimodalRuntime):
    """Online eval runtime with virtual-image and PIL-indexing rewrites enabled."""

    def _rewrite_virtual_clue_opens(self, code: str) -> str:
        return super()._rewrite_virtual_clue_opens(code)

    def _rewrite_pil_indexing(self, code: str) -> str:
        return super()._rewrite_pil_indexing(code)


def write_jsonl_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_resume_key(dataset: str, question_id: str) -> str:
    return f"{dataset}\t{question_id}"


def load_completed_resume_keys(path: Path) -> set[str]:
    completed = set()
    if not path.exists():
        return completed

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skip malformed resume line {line_no} in {path}", file=sys.stderr)
                continue
            dataset = record.get("dataset")
            question_id = record.get("question_id")
            if dataset is not None and question_id is not None:
                completed.add(sample_resume_key(str(dataset), str(question_id)))
    return completed


def normalize_eval_answer(value: Any) -> str:
    text = str(value or "").strip().strip("'\"").strip()
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1].strip()
    return text.strip().strip("'\"").strip()


def parse_eval_float(text: str) -> Optional[float]:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def is_prediction_correct(prediction: Any, gold_answer: Any) -> bool:
    pred = normalize_eval_answer(prediction)
    gold = normalize_eval_answer(gold_answer)
    if not pred or not gold:
        return False
    if re.fullmatch(r"[A-E]", gold, flags=re.IGNORECASE):
        return pred[:1].upper() == gold.upper()
    pred_num = parse_eval_float(pred)
    gold_num = parse_eval_float(gold)
    if pred_num is not None and gold_num is not None:
        return math.isclose(pred_num, gold_num, rel_tol=1e-4, abs_tol=1e-4)
    return pred.lower() == gold.lower()


def load_rerun_keys(path: Path, mode: str) -> set[str]:
    keys = set()
    if not path.exists():
        raise FileNotFoundError(f"Rerun source does not exist: {path}")
    include_tool_errors = mode in {"both", "tool_errors"}
    include_incorrect = mode in {"both", "incorrect"}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skip malformed rerun line {line_no} in {path}", file=sys.stderr)
                continue
            dataset = record.get("dataset")
            question_id = record.get("question_id")
            if dataset is None or question_id is None:
                continue
            is_tool_error = record.get("status") == "tool_execution_error"
            is_wrong = not is_prediction_correct(record.get("prediction", ""), record.get("gold_answer", ""))
            if (include_tool_errors and is_tool_error) or (include_incorrect and is_wrong):
                keys.add(sample_resume_key(str(dataset), str(question_id)))
    return keys


def to_file_url(path: str) -> str:
    return Path(path).resolve().as_uri()


def pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def normalize_openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for message in messages:
        clean_message = dict(message)
        content = clean_message.get("content", "")
        if isinstance(content, str):
            clean_message["content"] = content
        elif isinstance(content, list):
            clean_content = []
            for item in content:
                if isinstance(item, str):
                    clean_content.append({"type": "text", "text": item})
                    continue
                if not isinstance(item, dict):
                    clean_content.append({"type": "text", "text": str(item)})
                    continue

                item_type = item.get("type")
                if item_type == "text":
                    clean_content.append({"type": "text", "text": item.get("text", "")})
                elif item_type == "image":
                    image_value = item["image"]
                    if isinstance(image_value, Image.Image):
                        url = pil_to_data_url(image_value)
                    else:
                        url = to_file_url(str(image_value))
                    clean_content.append({"type": "image_url", "image_url": {"url": url}})
                elif item_type == "video":
                    video_value = item["video"]
                    clean_content.append({"type": "video_url", "video_url": {"url": to_file_url(str(video_value))}})
                elif item_type in {"image_url", "video_url"}:
                    clean_content.append(item)
                else:
                    clean_content.append({"type": "text", "text": str(item)})
            clean_message["content"] = clean_content
        normalized.append(clean_message)
    return normalized


def sanitize_messages_for_debug_online(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized = sanitize_messages_for_debug(messages)
    for message in sanitized:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image_url" and "image_url" in item:
                item["image_url"] = {"url": "<image_url>"}
            if item.get("type") == "video_url" and "video_url" in item:
                item["video_url"] = {"url": "<video_url>"}
    return sanitized


def resolve_relative_image_path(image_dir: str, image_path: str) -> str:
    path = Path(image_path)
    if path.is_absolute():
        return str(path)
    return str(Path(image_dir) / path.name)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_hrbench_samples(metadata_path: str, dataset_name: str, limit: Optional[int]) -> List[EvalSample]:
    root = Path(metadata_path).resolve().parent
    samples: List[EvalSample] = []
    for row in iter_jsonl(metadata_path):
        options = "\n".join(f"({letter}) {row[letter]}" for letter in ["A", "B", "C", "D"])
        question = (
            f"{row['question']}\n"
            f"{options}\n"
            "Answer with the option's letter from the given choices directly."
        )
        media_path = str(root / row["image_path"])
        samples.append(
            EvalSample(
                dataset=dataset_name,
                modality="image",
                question_id=str(row["index"]),
                question=question,
                gold_answer=row["answer"],
                media_path=media_path,
                category=row.get("category"),
                metadata=row,
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def load_mathvista_samples(metadata_path: str, image_dir: str, limit: Optional[int]) -> List[EvalSample]:
    samples: List[EvalSample] = []
    for row in iter_jsonl(metadata_path):
        question = row.get("query") or row["question"]
        media_path = resolve_relative_image_path(image_dir, row.get("image_path") or row.get("image"))
        samples.append(
            EvalSample(
                dataset="MathVista",
                modality="image",
                question_id=str(row.get("pid", len(samples))),
                question=question,
                gold_answer=str(row["answer"]),
                media_path=media_path,
                category=(row.get("metadata") or {}).get("category"),
                metadata=row,
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def load_mathvision_samples(metadata_path: str, image_dir: str, limit: Optional[int]) -> List[EvalSample]:
    samples: List[EvalSample] = []
    for row in iter_jsonl(metadata_path):
        question_text = row["question"].replace("<image1>", "").strip()
        options = row.get("options") or []
        if options:
            option_lines = "\n".join(f"({chr(ord('A') + idx)}) {option}" for idx, option in enumerate(options))
            question = (
                f"{question_text}\n"
                f"{option_lines}\n"
                "Answer with the option's letter from the given choices directly."
            )
        else:
            question = question_text
        media_path = resolve_relative_image_path(image_dir, row.get("image_path") or row.get("image"))
        samples.append(
            EvalSample(
                dataset="MathVision",
                modality="image",
                question_id=str(row.get("id", len(samples))),
                question=question,
                gold_answer=str(row["answer"]),
                media_path=media_path,
                category=row.get("subject"),
                metadata=row,
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def load_longvideobench_samples(json_path: str, data_dir: str, limit: Optional[int]) -> List[EvalSample]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[EvalSample] = []
    for item in data:
        question_category = item.get("question_category")
        if question_category in LONGVIDEOBENCH_SKIP_CATEGORIES:
            continue

        option_lines = []
        for idx, letter in enumerate(["A", "B", "C", "D", "E"]):
            option_key = f"option{idx}"
            if option_key in item:
                option_lines.append(f"{letter}. {item[option_key]}")
        question = (
            f"Question: {item['question']}\n"
            + "\n".join(option_lines)
            + "\nAnswer with the option's letter from the given choices directly."
        )

        correct_choice = int(item["correct_choice"])
        gold_answer = "ABCDE"[correct_choice]
        video_path = str(Path(data_dir) / item["video_path"])
        samples.append(
            EvalSample(
                dataset="LongVideoBench",
                modality="video",
                question_id=str(item.get("id", len(samples))),
                question=question,
                gold_answer=gold_answer,
                media_path=video_path,
                category=question_category,
                metadata=item,
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def parse_dataset_names(value: str) -> List[str]:
    aliases = {
        "vstar": "vstar",
        "vstar_bench": "vstar",
        "videomme": "videomme",
        "video_mme": "videomme",
        "hrbench4k": "hrbench4k",
        "hr_bench_4k": "hrbench4k",
        "hrbench8k": "hrbench8k",
        "hr_bench_8k": "hrbench8k",
        "longvideobench": "longvideobench",
        "long_video_bench": "longvideobench",
        "mathvista": "mathvista",
        "math_vista": "mathvista",
        "mathvision": "mathvision",
        "math_vision": "mathvision",
    }
    names = []
    for raw_name in value.split(","):
        key = raw_name.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"Unknown dataset '{raw_name}'.")
        names.append(aliases[key])
    return names


def build_samples(args: argparse.Namespace) -> List[EvalSample]:
    samples: List[EvalSample] = []
    limit_per_dataset = args.first_n_per_dataset
    if limit_per_dataset is None:
        limit_per_dataset = args.limit_per_dataset
    for dataset_name in parse_dataset_names(args.datasets):
        if dataset_name == "vstar":
            samples.extend(load_vstar_samples(args.vstar_path, limit_per_dataset))
        elif dataset_name == "videomme":
            samples.extend(load_videomme_samples(args.videomme_path, limit_per_dataset))
        elif dataset_name == "hrbench4k":
            samples.extend(load_hrbench_samples(args.hrbench4k_jsonl, "HR-Bench-4K", limit_per_dataset))
        elif dataset_name == "hrbench8k":
            samples.extend(load_hrbench_samples(args.hrbench8k_jsonl, "HR-Bench-8K", limit_per_dataset))
        elif dataset_name == "longvideobench":
            samples.extend(
                load_longvideobench_samples(
                    args.longvideobench_json,
                    args.longvideobench_data_dir,
                    limit_per_dataset,
                )
            )
        elif dataset_name == "mathvista":
            samples.extend(load_mathvista_samples(args.mathvista_jsonl, args.mathvista_image_dir, limit_per_dataset))
        elif dataset_name == "mathvision":
            samples.extend(load_mathvision_samples(args.mathvision_jsonl, args.mathvision_image_dir, limit_per_dataset))
    return samples


def build_initial_messages(
    sample: EvalSample,
    args: argparse.Namespace,
    prompt_templates: Dict[str, str],
) -> List[Dict[str, Any]]:
    if sample.modality == "image":
        image = Image.open(sample.media_path).convert("RGB")
        width, height = image.size
        prompt = prompt_templates[IMAGE_PROMPT_TEMPLATE_KEY].format(
            query=sample.question,
            width=width,
            height=height,
        )
        content = [
            {"type": "text", "text": "<image_clue_0>"},
            {"type": "image", "image": sample.media_path},
            {"type": "text", "text": "</image_clue_0>\n" + prompt},
        ]
    else:
        video_info = get_video_info_with_decord(sample.media_path)
        if args.video_initial_frames > 0:
            initial_video_text = (
                f"{args.video_initial_frames} frames will be sampled from the video by the vLLM server "
                "and provided as the initial visual input. "
                "The original full video is still available in the Python runtime as `video_clue_0`."
            )
        else:
            initial_video_text = (
                "No frames are provided as initial visual input. "
                "The original full video is available in the Python runtime as `video_clue_0`."
            )
        video_info_text = (
            f"Frame Width: {video_info['width']}; Frame Height: {video_info['height']};\n"
            f"Video Length: {video_info['video_length']}; Sample FPS: {video_info['fps']:.2f}\n"
            f"{initial_video_text}"
        )
        prompt = prompt_templates[VIDEO_PROMPT_TEMPLATE_KEY].format(
            video_info=video_info_text,
            query=sample.question,
        )
        if args.video_initial_frames > 0:
            content = [
                {"type": "text", "text": "<video_clue_0>"},
                {"type": "video", "video": sample.media_path},
                {"type": "text", "text": "</video_clue_0>\n" + prompt},
            ]
        else:
            content = [{"type": "text", "text": prompt}]

    return [{"role": "user", "content": content}]


def append_tool_result_online(
    messages: List[Dict[str, Any]],
    tool_text: str,
    tool_images: List[Image.Image],
    start_image_clue_idx: int = 0,
) -> None:
    if not tool_text and not tool_images:
        tool_text = "Tool executed successfully with no stdout and no displayed figures."

    if tool_images:
        content: List[Dict[str, Any]] = [{"type": "text", "text": "<tool_response>\n"}]
        if tool_text:
            content.append({"type": "text", "text": f"Text Result:\n{tool_text}\n"})
        content.append({"type": "text", "text": "Image Result:\n"})
        for offset, img in enumerate(tool_images):
            clue_idx = start_image_clue_idx + offset
            content.extend(
                [
                    {"type": "text", "text": f"<image_clue_{clue_idx}>"},
                    {"type": "image_url", "image_url": {"url": pil_to_data_url(img)}},
                    {"type": "text", "text": f"</image_clue_{clue_idx}>\n"},
                ]
            )
        content.append({"type": "text", "text": "\n</tool_response>"})
        messages.append({"role": "user", "content": content})
    else:
        messages.append(
            {
                "role": "tool",
                "content": f"<tool_response>\nText Result:\n{tool_text}\n</tool_response>",
            }
        )


def make_extra_body(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "include_stop_str_in_output": True,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "mm_processor_kwargs": {
            "size": {
                "shortest_edge": args.min_pixels,
                "longest_edge": args.max_pixels,
            },
            "do_sample_frames": False,
        },
    }


async def generate_once(client: Any, messages: List[Dict[str, Any]], args: argparse.Namespace) -> str:
    request = client.chat.completions.create(
        model=args.served_model_name,
        messages=normalize_openai_messages(messages),
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=["</code>", "</answer>"],
        extra_body=make_extra_body(args),
    )
    if args.request_timeout_seconds and args.request_timeout_seconds > 0:
        response = await asyncio.wait_for(request, timeout=args.request_timeout_seconds)
    else:
        response = await request
    return response.choices[0].message.content or ""


async def evaluate_sample(
    sample: EvalSample,
    client: Any,
    args: argparse.Namespace,
    prompt_templates: Dict[str, str],
    tool_lock: asyncio.Lock,
) -> Dict[str, Any]:
    sample_start_time = time.perf_counter()
    messages = build_initial_messages(sample, args, prompt_templates)
    runtime = PythonMultimodalRuntime(sample)
    raw_trajectory: List[Dict[str, Any]] = []
    tool_calls_log: List[Dict[str, Any]] = []
    status = "max_turns"
    error = ""
    prediction = ""
    generation_turns = 0
    model_request_seconds = 0.0
    tool_execution_seconds = 0.0

    for turn_idx in range(args.max_turns):
        try:
            request_start_time = time.perf_counter()
            raw_output = await generate_once(client, messages, args)
            request_elapsed = time.perf_counter() - request_start_time
            model_request_seconds += request_elapsed
            generation_turns += 1
        except Exception as exc:
            status = "request_error"
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            break

        assistant_content = strip_generation_special_tokens(raw_output)
        raw_trajectory.append(
            {
                "turn": turn_idx,
                "role": "assistant",
                # "raw_output": raw_output,
                "assistant_content": assistant_content,
                "request_seconds": round(request_elapsed, 6),
            }
        )
        messages.append({"role": "assistant", "content": assistant_content})

        if "<answer>" in assistant_content and "</answer>" in assistant_content:
            status = "success"
            prediction = extract_prediction(assistant_content)
            break

        code_actions = extract_code_actions(assistant_content)
        if not code_actions:
            status = "no_code_or_answer"
            prediction = extract_prediction(assistant_content)
            break

        for call_idx, code in enumerate(code_actions):
            try:
                async with tool_lock:
                    tool_start_time = time.perf_counter()
                    tool_text, tool_images = runtime.execute(code)
                    tool_elapsed = time.perf_counter() - tool_start_time
                tool_execution_seconds += tool_elapsed
                tool_image_start_idx = runtime.next_image_clue_idx
                append_tool_result_online(
                    messages,
                    tool_text,
                    tool_images,
                    start_image_clue_idx=tool_image_start_idx,
                )
                runtime.next_image_clue_idx += len(tool_images)
                tool_log = {
                    "turn": turn_idx,
                    "call_index": call_idx,
                    "name": "python_code",
                    "code": code,
                    "text_result": tool_text,
                    "num_images": len(tool_images),
                    "image_clue_start_idx": tool_image_start_idx if tool_images else None,
                    "tool_seconds": round(tool_elapsed, 6),
                }
                tool_calls_log.append(tool_log)
                raw_trajectory.append({"turn": turn_idx, "role": "tool", **tool_log})
            except Exception as exc:
                status = "tool_execution_error"
                error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                break

        if error:
            prediction = extract_prediction(assistant_content)
            break

    if not prediction and raw_trajectory:
        prediction = extract_prediction(str(raw_trajectory[-1].get("raw_output", "")))

    elapsed_seconds = time.perf_counter() - sample_start_time

    return {
        "dataset": sample.dataset,
        "backend": "vllm_online",
        "modality": sample.modality,
        "question_id": sample.question_id,
        "question": sample.question,
        "gold_answer": sample.gold_answer,
        "media_path": sample.media_path,
        "prediction": prediction,
        "raw_trajectory": raw_trajectory,
        "messages_for_debug": sanitize_messages_for_debug_online(messages),
        "tool_calls": tool_calls_log,
        "status": status,
        "error": error,
        "generation_turns": generation_turns,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "model_request_seconds": round(model_request_seconds, 6),
        "tool_execution_seconds": round(tool_execution_seconds, 6),
        "category": sample.category,
        "metadata": sample.metadata or {},
    }


async def run_eval(args: argparse.Namespace) -> None:
    samples = build_samples(args)
    print(f"Loaded {len(samples)} samples for datasets={args.datasets}")
    prompt_templates = load_prompt_templates(args.prompt_template_path)

    output_dir = Path(args.output_dir)
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"qwen3vl_vllm_online_agent_{timestamp}.jsonl"

    if args.resume:
        completed_keys = load_completed_resume_keys(output_path)
        before_count = len(samples)
        samples = [
            sample
            for sample in samples
            if sample_resume_key(sample.dataset, sample.question_id) not in completed_keys
        ]
        skipped_count = before_count - len(samples)
        print(f"Resume enabled: skipped {skipped_count} completed samples from {output_path}")
    elif not args.dry_run_data and output_path.exists():
        output_path.unlink()

    if args.rerun_from_results:
        rerun_keys = load_rerun_keys(Path(args.rerun_from_results), args.rerun_mode)
        before_count = len(samples)
        samples = [
            sample
            for sample in samples
            if sample_resume_key(sample.dataset, sample.question_id) in rerun_keys
        ]
        print(
            f"Rerun filter enabled: selected {len(samples)} / {before_count} samples "
            f"from {args.rerun_from_results} with mode={args.rerun_mode}"
        )

    if args.dry_run_data:
        counts: Dict[str, int] = {}
        for sample in samples:
            counts[sample.dataset] = counts.get(sample.dataset, 0) + 1
        print(json.dumps({"dataset_counts": counts}, ensure_ascii=False))
        for sample in samples[: min(args.dry_run_preview_limit, len(samples))]:
            preview = sample.__dict__.copy()
            preview["messages_preview"] = sanitize_messages_for_debug_online(
                build_initial_messages(sample, args, prompt_templates)
            )
            print(json.dumps(preview, ensure_ascii=False))
        return

    if not samples:
        print(f"No samples to evaluate. Output path: {output_path}")
        return

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The `openai` package is required for vLLM online inference. "
            "Install it in the active environment or run via the provided shell script."
        ) from exc

    client = AsyncOpenAI(base_url=args.openai_base_url, api_key=args.openai_api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    tool_lock = asyncio.Lock()
    write_lock = asyncio.Lock()
    pbar = tqdm(total=len(samples), desc="Evaluating")

    async def worker(sample: EvalSample) -> Dict[str, Any]:
        async with semaphore:
            result = await evaluate_sample(sample, client, args, prompt_templates, tool_lock)
            async with write_lock:
                write_jsonl_record(output_path, result)
                pbar.update(1)
            return result

    try:
        await asyncio.gather(*(worker(sample) for sample in samples))
    finally:
        pbar.close()
        await client.close()

    print(f"Saved {len(samples)} new results to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--served-model-name", default=MODEL_NAME)
    parser.add_argument("--openai-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--openai-api-key", default="EMPTY")
    parser.add_argument("--vstar-path", default=DEFAULT_VSTAR_PATH)
    parser.add_argument("--videomme-path", default=DEFAULT_VIDEOMME_PATH)
    parser.add_argument("--hrbench4k-jsonl", default=DEFAULT_HRBENCH_4K_JSONL)
    parser.add_argument("--hrbench8k-jsonl", default=DEFAULT_HRBENCH_8K_JSONL)
    parser.add_argument("--longvideobench-data-dir", default=DEFAULT_LONGVIDEOBENCH_DIR)
    parser.add_argument("--longvideobench-json", default=DEFAULT_LONGVIDEOBENCH_JSON)
    parser.add_argument("--mathvista-jsonl", default=DEFAULT_MATHVISTA_JSONL)
    parser.add_argument("--mathvista-image-dir", default=DEFAULT_MATHVISTA_IMAGE_DIR)
    parser.add_argument("--mathvision-jsonl", default=DEFAULT_MATHVISION_JSONL)
    parser.add_argument("--mathvision-image-dir", default=DEFAULT_MATHVISION_IMAGE_DIR)
    parser.add_argument("--prompt-template-path", default=DEFAULT_PROMPT_TEMPLATE_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-from-results", default=None)
    parser.add_argument("--rerun-mode", choices=["both", "tool_errors", "incorrect"], default="both")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--first-n-per-dataset", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--request-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--video-initial-frames", type=int, default=64)
    parser.add_argument("--dry-run-data", action="store_true")
    parser.add_argument("--dry-run-preview-limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "Visual pixel budget: "
        f"min_pixels={args.min_pixels} ({args.min_pixels // (QWEN3_SPATIAL_FACTOR ** 2)} visual tokens), "
        f"max_pixels={args.max_pixels} ({args.max_pixels // (QWEN3_SPATIAL_FACTOR ** 2)} visual tokens); "
        f"video_initial_frames={args.video_initial_frames}; "
        f"concurrency={args.concurrency}"
    )
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
