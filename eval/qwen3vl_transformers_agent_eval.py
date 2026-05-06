#!/usr/bin/env python
"""Multi-turn Qwen3-VL agent evaluation with a Transformers backend."""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.video_utils import VideoMetadata

try:
    from decord import VideoReader, cpu
except Exception:  # pragma: no cover - optional until video cases run.
    VideoReader = None
    cpu = None


DEFAULT_MODEL_PATH = "/share/home/sxjiang/model/Qwen3-VL-8B-Thinking/Qwen3-VL-8B-Thinking"
DEFAULT_VSTAR_PATH = "/share/home/sxjiang/zhzhu/dataset/vstar_bench/test_questions.jsonl"
DEFAULT_VIDEOMME_PATH = "/share/home/sxjiang/zhzhu/dataset/VideoMME/eval_template_copy.json"
DEFAULT_OUTPUT_DIR = "eval/results/qwen3vl_transformers_agent"
DEFAULT_PROMPT_TEMPLATE_PATH = "verl_agents/verl/utils/dataset/rl_system_prompt_template.json"

QWEN3_SPATIAL_FACTOR = 32
DEFAULT_MIN_VISUAL_TOKENS = 256
DEFAULT_MAX_VISUAL_TOKENS = 1024
DEFAULT_MIN_PIXELS = DEFAULT_MIN_VISUAL_TOKENS * QWEN3_SPATIAL_FACTOR * QWEN3_SPATIAL_FACTOR
DEFAULT_MAX_PIXELS = DEFAULT_MAX_VISUAL_TOKENS * QWEN3_SPATIAL_FACTOR * QWEN3_SPATIAL_FACTOR

IMAGE_PROMPT_TEMPLATE_KEY = "vistool_with_img_info_v2"
VIDEO_PROMPT_TEMPLATE_KEY = "vis_tool_with_img_info_video_v7"


class StopOnSubstrings(StoppingCriteria):
    """Stop generation once any target substring appears in the newly generated text."""

    def __init__(self, tokenizer: Any, prompt_length: int, stop_strings: List[str]):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.stop_strings = stop_strings

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        # This eval script generates one sample at a time, so a batch-level bool is enough.
        generated_ids = input_ids[0, self.prompt_length :]
        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return any(stop_string in text for stop_string in self.stop_strings)


@dataclass
class EvalSample:
    dataset: str
    modality: str
    question_id: str
    question: str
    gold_answer: str
    media_path: str
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PythonMultimodalRuntime:
    """Persistent lightweight runtime for one evaluation sample."""

    FORBIDDEN_RE = re.compile(r"(^|\s)(input|os\.system|subprocess|shutil\.rmtree)\s*\(")

    def __init__(self, sample: EvalSample):
        self.sample = sample
        self.globals: Dict[str, Any] = {}
        self._captured_images: List[str] = []
        self.next_image_clue_idx = 1 if sample.modality == "image" else 0
        self._init_runtime()

    def _init_runtime(self) -> None:
        self.globals.update(
            {
                "plt": plt,
                "Image": Image,
                "io": io,
                "base64": base64,
                "torch": torch,
                "_captured_images": self._captured_images,
            }
        )

        import numpy as np

        self.globals["np"] = np

        def _internal_capture_plt_figure() -> None:
            fig = plt.gcf()
            width_px = fig.get_figwidth() * fig.dpi
            height_px = fig.get_figheight() * fig.dpi
            if min(width_px, height_px) <= 0:
                raise RuntimeError("Invalid matplotlib figure size.")
            aspect_ratio = max(width_px, height_px) / min(width_px, height_px)
            if aspect_ratio >= 200:
                raise RuntimeError(f"Image aspect ratio too extreme: {aspect_ratio:.2f}")

            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            self._captured_images.append(base64.b64encode(buffer.read()).decode("utf-8"))
            plt.close(fig)

        self.globals["_internal_capture_plt_figure"] = _internal_capture_plt_figure

        if self.sample.modality == "image":
            self.globals["image_clue_0"] = Image.open(self.sample.media_path).convert("RGB")
        elif self.sample.modality == "video":
            if VideoReader is None or cpu is None:
                raise RuntimeError("decord is required for video tool runtime but is not importable.")
            self.globals["VideoReader"] = VideoReader
            self.globals["cpu"] = cpu
            self.globals["video_clue_0"] = VideoReader(self.sample.media_path, ctx=cpu(0))

    def execute(self, code: str) -> Tuple[str, List[Image.Image]]:
        if self.FORBIDDEN_RE.search(code):
            raise RuntimeError("Forbidden function call detected in generated code.")

        pre_image_count = len(self._captured_images)
        stdout = io.StringIO()
        modified_code = self._rewrite_virtual_clue_opens(code)
        modified_code = modified_code.replace("plt.show()", "_internal_capture_plt_figure()")

        try:
            with redirect_stdout(stdout):
                exec(modified_code, self.globals)
        finally:
            plt.close("all")

        text = stdout.getvalue().strip()
        new_images = [
            Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
            for img_b64 in self._captured_images[pre_image_count:]
        ]
        return text, new_images

    def _rewrite_virtual_clue_opens(self, code: str) -> str:
        """Treat image_clue_i/image_hint_i as in-memory variables, not filesystem paths."""
        clue_names = [
            name
            for name, value in self.globals.items()
            if re.fullmatch(r"(image_clue|image_hint)_\d+", name) and isinstance(value, Image.Image)
        ]
        rewritten = code
        for name in clue_names:
            quoted = rf"['\"]{re.escape(name)}['\"]"
            rewritten = re.sub(rf"Image\.open\(\s*{quoted}\s*\)", f"{name}.copy()", rewritten)
            rewritten = re.sub(rf"Image\.open\(\s*{re.escape(name)}\s*\)", f"{name}.copy()", rewritten)
        return rewritten


def get_video_info_with_decord(video_path: str) -> Dict[str, Any]:
    if VideoReader is None or cpu is None:
        raise RuntimeError("decord is required for video metadata but is not importable.")
    video_reader = VideoReader(video_path, ctx=cpu(0))
    video_len = len(video_reader)
    if video_len <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")
    first_frame = video_reader[0].asnumpy()
    height, width = first_frame.shape[:2]
    return {
        "width": int(width),
        "height": int(height),
        "video_length": int(video_len),
        "fps": float(video_reader.get_avg_fps()),
    }


def sample_video_for_initial_input(video_path: str, num_frames: int) -> Tuple[np.ndarray, VideoMetadata]:
    if VideoReader is None or cpu is None:
        raise RuntimeError("decord is required for initial video sampling but is not importable.")
    video_reader = VideoReader(video_path, ctx=cpu(0))
    video_len = len(video_reader)
    if video_len <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")

    sampled_frames = max(1, min(num_frames, video_len))
    indices = np.linspace(0, video_len - 1, sampled_frames, dtype=int)
    frames = video_reader.get_batch(indices).asnumpy()
    height, width = frames.shape[1:3]
    metadata = VideoMetadata(
        total_num_frames=int(video_len),
        fps=float(video_reader.get_avg_fps()),
        width=int(width),
        height=int(height),
        frames_indices=indices.tolist(),
    )
    return frames, metadata


def sanitize_messages_for_debug(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sanitize_content(content: Any) -> Any:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return content
        sanitized = []
        for item in content:
            if not isinstance(item, dict):
                sanitized.append(item)
                continue
            clean = dict(item)
            if "image" in clean and not isinstance(clean["image"], str):
                clean["image"] = "<PIL.Image>"
            if "video" in clean and not isinstance(clean["video"], str):
                video_value = clean["video"]
                shape = getattr(video_value, "shape", None)
                clean["video"] = f"<decoded_video shape={tuple(shape) if shape is not None else 'unknown'}>"
            if "video_metadata" in clean:
                metadata = clean["video_metadata"]
                clean["video_metadata"] = {
                    "total_num_frames": getattr(metadata, "total_num_frames", None),
                    "fps": getattr(metadata, "fps", None),
                    "width": getattr(metadata, "width", None),
                    "height": getattr(metadata, "height", None),
                    "frames_indices": getattr(metadata, "frames_indices", None),
                }
            sanitized.append(clean)
        return sanitized

    output = []
    for msg in messages:
        clean_msg = dict(msg)
        clean_msg["content"] = sanitize_content(clean_msg.get("content"))
        output.append(clean_msg)
    return output


def normalize_messages_for_processor(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Qwen3-VL processor expects content parts, even for pure text messages."""
    normalized = []
    for message in messages:
        clean_message = dict(message)
        content = clean_message.get("content", "")
        if isinstance(content, str):
            clean_message["content"] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            clean_content = []
            for item in content:
                if isinstance(item, str):
                    clean_content.append({"type": "text", "text": item})
                else:
                    clean_content.append(item)
            clean_message["content"] = clean_content
        normalized.append(clean_message)
    return normalized


def load_vstar_samples(path: str, limit: Optional[int]) -> List[EvalSample]:
    base_dir = Path(path).resolve().parent
    samples: List[EvalSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            image_path = str(base_dir / item["image"])
            samples.append(
                EvalSample(
                    dataset="vstar_bench",
                    modality="image",
                    question_id=str(item.get("question_id", len(samples))),
                    question=item["text"],
                    gold_answer=item["label"],
                    media_path=image_path,
                    category=item.get("category"),
                    metadata=item,
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples


def flatten_videomme_question(video_item: Dict[str, Any], question_item: Dict[str, Any]) -> str:
    options = "\n".join(question_item["options"])
    return (
        f"Question: {question_item['question']}\n"
        f"{options}\n"
        "Answer with the option's letter from the given choices directly."
    )


def load_videomme_samples(path: str, limit: Optional[int]) -> List[EvalSample]:
    base_dir = Path(path).resolve().parent
    samples: List[EvalSample] = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for video_item in data:
        video_path = str(base_dir / "data" / f"{video_item['video_name']}.mp4")
        for question_item in video_item["questions"]:
            metadata = {
                "video_id": video_item.get("video_id"),
                "video_name": video_item.get("video_name"),
                "duration": video_item.get("duration"),
                "domain": video_item.get("domain"),
                "sub_category": video_item.get("sub_category"),
                "task_type": question_item.get("task_type"),
            }
            samples.append(
                EvalSample(
                    dataset="VideoMME",
                    modality="video",
                    question_id=str(question_item["question_id"]),
                    question=flatten_videomme_question(video_item, question_item),
                    gold_answer=question_item["answer"],
                    media_path=video_path,
                    category=video_item.get("duration"),
                    metadata=metadata,
                )
            )
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


def load_prompt_templates(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
            {
                "type": "image",
                "image": sample.media_path,
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
            },
            {"type": "text", "text": "</image_clue_0>\n" + prompt},
        ]
    else:
        video_info = get_video_info_with_decord(sample.media_path)
        initial_video: Optional[np.ndarray] = None
        initial_video_metadata: Optional[VideoMetadata] = None
        initial_video_text = ""
        if args.video_initial_frames > 0:
            initial_video, initial_video_metadata = sample_video_for_initial_input(
                sample.media_path,
                args.video_initial_frames,
            )
            initial_video_text = (
                f"{initial_video.shape[0]} frames have been uniformly sampled from the video "
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
        if initial_video is not None:
            content = [
                {"type": "text", "text": "<video_clue_0>"},
                {
                    "type": "video",
                    "video": initial_video,
                    "video_metadata": initial_video_metadata,
                    "min_pixels": args.min_pixels,
                    "max_pixels": args.max_pixels,
                },
                {"type": "text", "text": "</video_clue_0>\n" + prompt},
            ]
        else:
            content = [{"type": "text", "text": prompt}]

    return [{"role": "user", "content": content}]


def extract_code_actions(text: str) -> List[str]:
    code_blocks = re.findall(r"<code>\s*```python\s*(.*?)\s*```\s*</code>", text, flags=re.DOTALL)
    if code_blocks:
        return [code.strip() for code in code_blocks if code.strip()]

    fenced_blocks = re.findall(r"```python\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced_blocks:
        return [code.strip() for code in fenced_blocks if code.strip()]

    # Backward-compatible fallback for the earlier eval protocol. Training-aligned prompts
    # should use <code>...</code>, but this keeps old generations debuggable.
    tool_call_blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL)
    codes = []
    for block in tool_call_blocks:
        try:
            call = json.loads(block.strip().strip("`"))
        except json.JSONDecodeError:
            continue
        arguments = call.get("arguments", {})
        code = arguments.get("code")
        if isinstance(code, str) and code.strip():
            codes.append(code.strip())
    return codes


def extract_prediction(text: str) -> str:
    answer_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    candidate = answer_matches[-1] if answer_matches else text

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", candidate)
    if boxed:
        candidate = boxed[-1]

    letter = re.search(r"\b([A-E])\b", candidate.strip())
    return letter.group(1) if letter else candidate.strip()


def strip_generation_special_tokens(text: str) -> str:
    """Remove outer generation delimiters before putting assistant text back into chat."""
    stripped = text.strip()
    special_suffixes = ("<|im_end|>", "<|endoftext|>")
    changed = True
    while changed:
        changed = False
        for suffix in special_suffixes:
            if stripped.endswith(suffix):
                stripped = stripped[: -len(suffix)].rstrip()
                changed = True
    return stripped


def append_tool_result(
    messages: List[Dict[str, Any]],
    tool_text: str,
    tool_images: List[Image.Image],
    args: argparse.Namespace,
    start_image_clue_idx: int = 0,
) -> None:
    if not tool_text and not tool_images:
        tool_text = "Tool executed successfully with no stdout and no displayed figures."

    if tool_images:
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": "<tool_response>\n"},   # interpreter可以删去
        ]
        if tool_text:
            content.append({"type": "text", "text": f"Text Result:\n{tool_text}\n"})
        content.append({"type": "text", "text": "Image Result:\n"})
        for offset, img in enumerate(tool_images):
            clue_idx = start_image_clue_idx + offset
            content.extend(
                [
                    {"type": "text", "text": f"<image_clue_{clue_idx}>"},
                    {
                        "type": "image",
                        "image": img,
                        "min_pixels": args.min_pixels,
                        "max_pixels": args.max_pixels,
                    },
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


def generate_once(
    model: Any,
    processor: Any,
    messages: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[str, str]:
    processor_messages = normalize_messages_for_processor(messages)
    video_metadata = [
        item["video_metadata"]
        for message in processor_messages
        for item in message.get("content", [])
        if isinstance(item, dict) and item.get("type") == "video" and item.get("video_metadata") is not None
    ]
    template_kwargs: Dict[str, Any] = {
        "images_kwargs": {
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
        "videos_kwargs": {
            "do_sample_frames": False,
            "size": {
                "shortest_edge": args.min_pixels,
                "longest_edge": args.max_pixels,
            },
        },
    }
    if video_metadata:
        template_kwargs["videos_kwargs"]["video_metadata"] = video_metadata[0] if len(video_metadata) == 1 else video_metadata

    inputs = processor.apply_chat_template(
        processor_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
    )
    inputs = inputs.to(model.device)
    input_ids = inputs['input_ids']
    text = text = processor.tokenizer.decode(
        input_ids[0],
        skip_special_tokens=False,
    )
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList(
                [
                    StopOnSubstrings(
                        tokenizer=processor.tokenizer,
                        prompt_length=prompt_len,
                        stop_strings=["</code>", "</answer>"],
                    )
                ]
            ),
        )

    new_ids = generated_ids[:, prompt_len:]
    raw_output = processor.batch_decode(
        new_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    # text_output = processor.batch_decode(
    #     new_ids,
    #     skip_special_tokens=True,
    #     clean_up_tokenization_spaces=False,
    # )[0]
    return raw_output


def evaluate_sample(
    sample: EvalSample,
    model: Any,
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
        raw_output = generate_once(model, processor, messages, args)
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


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_samples(args: argparse.Namespace) -> List[EvalSample]:
    samples: List[EvalSample] = []
    if args.modality in ("image", "both"):
        samples.extend(load_vstar_samples(args.vstar_path, args.limit))
    if args.modality in ("video", "both"):
        samples.extend(load_videomme_samples(args.videomme_path, args.limit))
    return samples


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
    parser.add_argument("--dry-run-data", action="store_true", help="Only parse data and write no model outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "Visual pixel budget: "
        f"min_pixels={args.min_pixels} ({args.min_pixels // (QWEN3_SPATIAL_FACTOR ** 2)} visual tokens), "
        f"max_pixels={args.max_pixels} ({args.max_pixels // (QWEN3_SPATIAL_FACTOR ** 2)} visual tokens); "
        f"video_initial_frames={args.video_initial_frames}"
    )

    samples = build_samples(args)
    print(f"Loaded {len(samples)} samples for modality={args.modality}")
    if args.dry_run_data:
        prompt_templates = load_prompt_templates(args.prompt_template_path)
        for sample in samples[:5]:
            preview = sample.__dict__.copy()
            preview["messages_preview"] = sanitize_messages_for_debug(
                build_initial_messages(sample, args, prompt_templates)
            )
            print(json.dumps(preview, ensure_ascii=False))
        return

    prompt_templates = load_prompt_templates(args.prompt_template_path)
    print(f"Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"Loading model from {args.model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    output_dir = Path(args.output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"qwen3vl_transformers_agent_{args.modality}_{timestamp}.jsonl"

    results = []
    for sample in tqdm(samples, desc="Evaluating"):
        result = evaluate_sample(sample, model, processor, args, prompt_templates)
        results.append(result)
        write_jsonl(output_path, results)

    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
