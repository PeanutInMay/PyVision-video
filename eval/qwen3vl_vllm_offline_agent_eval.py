#!/usr/bin/env python
"""Multi-turn Qwen3-VL agent evaluation with a vLLM offline backend."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata

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
    QWEN3_SPATIAL_FACTOR,
    PythonMultimodalRuntime,
    append_tool_result,
    build_initial_messages,
    build_samples,
    extract_code_actions,
    extract_prediction,
    load_prompt_templates,
    normalize_messages_for_processor,
    sanitize_messages_for_debug,
    strip_generation_special_tokens,
)


DEFAULT_OUTPUT_DIR = "eval/results/qwen3vl_vllm_offline_agent"


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def video_metadata_to_dict(metadata: Any, video: np.ndarray) -> Dict[str, Any]:
    if isinstance(metadata, VideoMetadata):
        metadata_dict = {
            "total_num_frames": metadata.total_num_frames,
            "fps": metadata.fps,
            "width": metadata.width,
            "height": metadata.height,
            "duration": metadata.duration,
            "video_backend": metadata.video_backend,
            "frames_indices": metadata.frames_indices,
        }
    elif isinstance(metadata, dict):
        metadata_dict = dict(metadata)
    else:
        metadata_dict = {}

    if not metadata_dict.get("total_num_frames"):
        metadata_dict["total_num_frames"] = int(video.shape[0])
    if metadata_dict.get("fps") is None:
        metadata_dict["fps"] = 24.0
    if metadata_dict.get("height") is None and video.ndim >= 3:
        metadata_dict["height"] = int(video.shape[1])
    if metadata_dict.get("width") is None and video.ndim >= 3:
        metadata_dict["width"] = int(video.shape[2])
    if metadata_dict.get("frames_indices") is None:
        metadata_dict["frames_indices"] = list(range(int(video.shape[0])))
    metadata_dict["do_sample_frames"] = False
    return metadata_dict


def collect_vllm_multi_modal_data(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    images: List[Image.Image] = []
    videos: List[Any] = []

    for message in messages:
        for item in message.get("content", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image" and "image" in item:
                image_value = item["image"]
                if isinstance(image_value, str):
                    images.append(Image.open(image_value).convert("RGB"))
                elif isinstance(image_value, Image.Image):
                    images.append(image_value.convert("RGB"))
                else:
                    images.append(image_value)
            elif item.get("type") == "video" and "video" in item:
                video_value = item["video"]
                metadata = video_metadata_to_dict(item.get("video_metadata"), video_value)
                videos.append((video_value, metadata))

    multi_modal_data: Dict[str, Any] = {}
    if images:
        multi_modal_data["image"] = images if len(images) > 1 else images[0]
    if videos:
        multi_modal_data["video"] = videos if len(videos) > 1 else videos[0]
    return multi_modal_data


def build_vllm_prompt(
    processor: Any,
    messages: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    processor_messages = normalize_messages_for_processor(messages)
    prompt = processor.apply_chat_template(
        processor_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    request: Dict[str, Any] = {"prompt": prompt}
    multi_modal_data = collect_vllm_multi_modal_data(processor_messages)
    if multi_modal_data:
        request["multi_modal_data"] = multi_modal_data
        request["mm_processor_kwargs"] = {
            "size": {
                "shortest_edge": args.min_pixels,
                "longest_edge": args.max_pixels,
            },
            "do_sample_frames": False,
        }
    return request


def make_sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
        stop=["</code>", "</answer>"],
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )


def generate_once(llm: Any, processor: Any, messages: List[Dict[str, Any]], args: argparse.Namespace) -> str:
    request = build_vllm_prompt(processor, messages, args)
    outputs = llm.generate([request], make_sampling_params(args), use_tqdm=False)
    return outputs[0].outputs[0].text


def evaluate_sample(
    sample: Any,
    llm: Any,
    processor: Any,
    args: argparse.Namespace,
    prompt_templates: Dict[str, str],
) -> Dict[str, Any]:
    messages = build_initial_messages(sample, args, prompt_templates)
    runtime = PythonMultimodalRuntime(sample)
    raw_trajectory: List[Dict[str, Any]] = []
    tool_calls_log: List[Dict[str, Any]] = []
    status = "max_turns"
    error = ""
    prediction = ""

    for turn_idx in range(args.max_turns):
        raw_output = generate_once(llm, processor, messages, args)
        assistant_content = strip_generation_special_tokens(raw_output)
        raw_trajectory.append(
            {
                "turn": turn_idx,
                "role": "assistant",
                "raw_output": raw_output,
                "assistant_content": assistant_content,
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
                tool_text, tool_images = runtime.execute(code)
                tool_image_start_idx = runtime.next_image_clue_idx
                append_tool_result(
                    messages,
                    tool_text,
                    tool_images,
                    args,
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

    return {
        "dataset": sample.dataset,
        "backend": "vllm_offline",
        "modality": sample.modality,
        "question_id": sample.question_id,
        "question": sample.question,
        "gold_answer": sample.gold_answer,
        "media_path": sample.media_path,
        "prediction": prediction,
        "raw_trajectory": raw_trajectory,
        "messages_for_debug": sanitize_messages_for_debug(messages),
        "tool_calls": tool_calls_log,
        "status": status,
        "error": error,
        "category": sample.category,
        "metadata": sample.metadata or {},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vstar-path", default=DEFAULT_VSTAR_PATH)
    parser.add_argument("--videomme-path", default=DEFAULT_VIDEOMME_PATH)
    parser.add_argument("--prompt-template-path", default=DEFAULT_PROMPT_TEMPLATE_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modality", choices=["image", "video", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument(
        "--video-initial-frames",
        type=int,
        default=64,
        help="Uniformly sample this many frames as the initial video input. Set 0 to disable initial video input.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--limit-mm-images", type=int, default=16)
    parser.add_argument("--limit-mm-videos", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run-data", action="store_true", help="Only parse data and write no model outputs.")
    return parser.parse_args()


def build_llm(args: argparse.Namespace) -> Any:
    from vllm import LLM

    llm_kwargs: Dict[str, Any] = {
        "model": args.model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": "auto",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
        "limit_mm_per_prompt": {
            "image": args.limit_mm_images,
            "video": args.limit_mm_videos,
        },
        "mm_processor_kwargs": {
            "size": {
                "shortest_edge": args.min_pixels,
                "longest_edge": args.max_pixels,
            },
            "do_sample_frames": False,
        },
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    return LLM(**llm_kwargs)


def main() -> None:
    args = parse_args()
    print(
        "Visual pixel budget: "
        f"min_pixels={args.min_pixels} ({args.min_pixels // (QWEN3_SPATIAL_FACTOR ** 2)} visual tokens), "
        f"max_pixels={args.max_pixels} ({args.max_pixels // (QWEN3_SPATIAL_FACTOR ** 2)} visual tokens); "
        f"video_initial_frames={args.video_initial_frames}; "
        f"tensor_parallel_size={args.tensor_parallel_size}"
    )

    samples = build_samples(args)
    print(f"Loaded {len(samples)} samples for modality={args.modality}")
    prompt_templates = load_prompt_templates(args.prompt_template_path)
    if args.dry_run_data:
        for sample in samples[:5]:
            preview = sample.__dict__.copy()
            preview["messages_preview"] = sanitize_messages_for_debug(
                build_initial_messages(sample, args, prompt_templates)
            )
            print(json.dumps(preview, ensure_ascii=False))
        return

    print(f"Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"Loading vLLM offline engine from {args.model_path}")
    llm = build_llm(args)

    output_dir = Path(args.output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"qwen3vl_vllm_offline_agent_{args.modality}_{timestamp}.jsonl"

    results = []
    for sample in tqdm(samples, desc="Evaluating"):
        result = evaluate_sample(sample, llm, processor, args, prompt_templates)
        results.append(result)
        write_jsonl(output_path, results)

    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
