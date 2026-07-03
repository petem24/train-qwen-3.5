#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests
import torch
from PIL import Image


def log(message: str) -> None:
    print(message, flush=True)


def env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def env_bool(name: str, default: bool = False) -> bool:
    value = env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = env(name)
    return default if value is None else int(value)


def env_float(name: str, default: float) -> float:
    value = env(name)
    return default if value is None else float(value)


def parse_json_env(name: str) -> dict[str, Any]:
    value = env(name)
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


def parse_list_env(name: str, default: str) -> list[str]:
    value = env(name, default) or ""
    return [part.strip() for part in value.split(",") if part.strip()]


def add_api_key(url: str, api_key: str | None) -> str:
    if not api_key:
        return url
    parsed = urlparse(url)
    if parsed.netloc != "api.roboflow.com":
        return url
    query = parse_qs(parsed.query, keep_blank_values=True)
    if "api_key" not in query:
        query["api_key"] = [api_key]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def parse_roboflow_ref(ref: str) -> tuple[str, str, str] | None:
    value = ref.strip()
    if not value:
        return None

    if "://" not in value:
        parts = [part for part in value.strip("/").split("/") if part]
    else:
        parsed = urlparse(value)
        parts = [part for part in parsed.path.strip("/").split("/") if part]

    if len(parts) == 3 and parts[2].isdigit():
        return parts[0], parts[1], parts[2]

    for marker in ("dataset", "version"):
        if marker in parts:
            index = parts.index(marker)
            if index >= 2 and index + 1 < len(parts):
                return parts[index - 2], parts[index - 1], parts[index + 1]

    for index, part in enumerate(parts):
        if part.isdigit() and index >= 2:
            return parts[index - 2], parts[index - 1], part

    return None


def is_direct_zip_url(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    path = parsed.path.lower()
    if path.endswith(".zip"):
        return True
    if parsed.netloc.endswith("roboflow.com") and path.startswith("/ds/"):
        return True
    return False


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        downloaded = 0
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                file.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\rDownloading dataset: {pct:5.1f}%", end="", flush=True)
    if total:
        print(flush=True)


def extract_zip(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)


def find_dataset_root(base_dir: Path) -> Path:
    candidates = [base_dir]
    candidates.extend(path for path in base_dir.iterdir() if path.is_dir())

    for candidate in candidates:
        if any((candidate / split / "_annotations.coco.json").exists() for split in ("train", "valid", "test")):
            return candidate
        if (candidate / "train" / "images").exists():
            return candidate

    return base_dir


def download_with_direct_url(url: str, dataset_dir: Path, api_key: str | None) -> Path:
    clear_directory(dataset_dir)
    zip_path = dataset_dir / "roboflow.zip"
    download_file(add_api_key(url, api_key), zip_path)
    extract_zip(zip_path, dataset_dir)
    zip_path.unlink(missing_ok=True)
    return find_dataset_root(dataset_dir)


def download_with_api(
    workspace: str,
    project: str,
    version: str,
    dataset_format: str,
    dataset_dir: Path,
    api_key: str,
) -> Path:
    clear_directory(dataset_dir)

    from roboflow import Roboflow

    log(f"Downloading Roboflow dataset {workspace}/{project}/{version} as {dataset_format}...")
    rf = Roboflow(api_key=api_key)
    dataset = rf.workspace(workspace).project(project).version(int(version)).download(
        model_format=dataset_format,
        location=str(dataset_dir),
        overwrite=True,
    )
    location = Path(getattr(dataset, "location", dataset_dir))
    return find_dataset_root(location)


def resolve_dataset() -> Path:
    dataset_dir = Path(env("DATASET_DIR", "/workspace/dataset")).expanduser()
    dataset_format = env("DATASET_FORMAT", "coco")
    api_key = env("ROBOFLOW_API_KEY")
    dataset_url = env("ROBOFLOW_DATASET_URL")

    if env_bool("SKIP_DATASET_DOWNLOAD", False):
        log("SKIP_DATASET_DOWNLOAD=true, using existing DATASET_DIR.")
        return find_dataset_root(dataset_dir)

    if dataset_url and is_direct_zip_url(dataset_url):
        log("Downloading dataset from direct Roboflow ZIP URL...")
        return download_with_direct_url(dataset_url, dataset_dir, api_key)

    workspace = env("ROBOFLOW_WORKSPACE")
    project = env("ROBOFLOW_PROJECT")
    version = env("ROBOFLOW_VERSION")

    if dataset_url:
        parsed = parse_roboflow_ref(dataset_url)
        if parsed:
            workspace, project, version = parsed

    if not workspace or not project or not version:
        raise RuntimeError(
            "Set ROBOFLOW_DATASET_URL to a Roboflow dataset URL, or set "
            "ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, and ROBOFLOW_VERSION."
        )
    if not api_key:
        raise RuntimeError("Set ROBOFLOW_API_KEY as a runtime environment variable.")

    return download_with_api(workspace, project, version, dataset_format, dataset_dir, api_key)


def find_coco_annotation(dataset_root: Path, split: str) -> Path | None:
    candidates = [
        dataset_root / split / "_annotations.coco.json",
        dataset_root / f"{split}.json",
        dataset_root / "annotations" / f"instances_{split}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_image_path(dataset_root: Path, split: str, file_name: str) -> Path | None:
    raw = Path(file_name)
    candidates = [
        dataset_root / raw,
        dataset_root / split / raw,
        dataset_root / split / "images" / raw,
        dataset_root / split / "images" / raw.name,
        dataset_root / split / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def clean_label(label: str) -> str:
    value = label.strip()
    for prefix in parse_list_env("LABEL_PREFIXES_TO_STRIP", ""):
        if value.lower().startswith(prefix):
            value = value[len(prefix) :]
    return value.strip()


def compose_answer(labels: list[str]) -> str:
    cleaned = [clean_label(label) for label in labels if clean_label(label)]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if all(len(label) == 1 for label in cleaned):
        return "".join(cleaned)
    return " ".join(cleaned)


def records_from_coco(dataset_root: Path, split: str) -> list[dict[str, str]]:
    annotation_path = find_coco_annotation(dataset_root, split)
    if annotation_path is None:
        return []

    with annotation_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    categories = {category["id"]: category["name"] for category in coco.get("categories", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco.get("annotations", []):
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    records: list[dict[str, str]] = []
    prompt = env(
        "OCR_PROMPT",
        "Read the text in this image. Answer with only the text.",
    )
    for image in coco.get("images", []):
        image_id = image["id"]
        image_path = resolve_image_path(dataset_root, split, image["file_name"])
        if image_path is None:
            continue

        annotations = annotations_by_image.get(image_id, [])
        annotations.sort(key=lambda item: (item.get("bbox") or [0, 0, 0, 0])[0])
        labels = [categories.get(item.get("category_id"), "") for item in annotations]
        answer = compose_answer(labels)
        if not answer:
            continue

        records.append(
            {
                "image": str(image_path),
                "question": prompt,
                "answer": answer,
                "split": split,
            }
        )

    return records


def prepare_records(dataset_root: Path, output_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]] | None]:
    train_records = records_from_coco(dataset_root, "train")
    eval_records = records_from_coco(dataset_root, "valid")

    if not train_records:
        raise RuntimeError(
            f"No OCR training records found in {dataset_root}. Expected Roboflow COCO files like "
            "train/_annotations.coco.json with labeled images."
        )

    sft_dir = output_dir / "qwen35_sft"
    sft_dir.mkdir(parents=True, exist_ok=True)
    for split, records in (("train", train_records), ("valid", eval_records)):
        if not records:
            continue
        with (sft_dir / f"{split}.jsonl").open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    log(f"Prepared {len(train_records)} train records and {len(eval_records)} validation records.")
    log(f"Wrote converted SFT records to {sft_dir}.")
    return train_records, eval_records or None


class RoboflowOCRDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, str]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.records[index]


@dataclass
class QwenVLCollator:
    processor: Any
    mask_prompt: bool = True

    def __post_init__(self) -> None:
        self.process_vision_info = None
        try:
            from qwen_vl_utils import process_vision_info

            self.process_vision_info = process_vision_info
        except Exception:
            self.process_vision_info = None

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "right"
            if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

    def make_messages(self, record: dict[str, str], include_answer: bool) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": record["image"]},
                    {"type": "text", "text": record["question"]},
                ],
            }
        ]
        if include_answer:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": record["answer"]}],
                }
            )
        return messages

    def encode(self, records: list[dict[str, str]], include_answer: bool) -> dict[str, torch.Tensor]:
        messages = [self.make_messages(record, include_answer=include_answer) for record in records]
        texts = [
            self.processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=not include_answer,
            )
            for message in messages
        ]

        if self.process_vision_info is not None:
            images, videos = self.process_vision_info(messages)
            kwargs = {
                "text": texts,
                "images": images,
                "padding": True,
                "return_tensors": "pt",
            }
            if videos is not None:
                kwargs["videos"] = videos
            try:
                return self.processor(**kwargs)
            except TypeError:
                kwargs.pop("videos", None)
                return self.processor(**kwargs)

        images = [Image.open(record["image"]).convert("RGB") for record in records]
        return self.processor(text=texts, images=images, padding=True, return_tensors="pt")

    def __call__(self, records: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        batch = self.encode(records, include_answer=True)
        labels = batch["input_ids"].clone()

        pad_token_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        if self.mask_prompt:
            prompt_batch = self.encode(records, include_answer=False)
            prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
            for index, prompt_length in enumerate(prompt_lengths):
                labels[index, : int(prompt_length)] = -100

        batch["labels"] = labels
        return batch


def resolve_dtype() -> torch.dtype:
    value = env("TORCH_DTYPE", "bfloat16").lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError("TORCH_DTYPE must be one of bfloat16, float16, or float32")


def load_model_and_processor() -> tuple[Any, Any]:
    from transformers import AutoProcessor

    model_id = env("MODEL_ID", "Qwen/Qwen3.5-0.8B")
    dtype = resolve_dtype()
    attn_implementation = env("ATTN_IMPLEMENTATION", "sdpa")
    trust_remote_code = env_bool("TRUST_REMOTE_CODE", True)
    bits = env_int("BITS", 16)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
        "attn_implementation": attn_implementation,
    }
    if bits in {4, 8}:
        from transformers import BitsAndBytesConfig

        model_kwargs["device_map"] = env("DEVICE_MAP", "auto")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=bits == 4,
            load_in_8bit=bits == 8,
            bnb_4bit_quant_type=env("BNB_4BIT_QUANT_TYPE", "nf4"),
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=env_bool("BNB_4BIT_USE_DOUBLE_QUANT", True),
        )

    model_kwargs.update(parse_json_env("MODEL_LOAD_JSON"))

    import transformers

    for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            continue
        try:
            log(f"Loading {model_id} with {class_name}...")
            model = model_class.from_pretrained(model_id, **model_kwargs)
            break
        except Exception as exc:
            log(f"{class_name} failed: {exc}")
    else:
        raise RuntimeError("Could not load model with any supported Transformers auto class.")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return model, processor


def apply_lora(model: Any) -> Any:
    if not env_bool("LORA", True):
        return model

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bits = env_int("BITS", 16)
    gradient_checkpointing = env_bool("GRADIENT_CHECKPOINTING", True)
    if bits in {4, 8}:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )

    target_modules = parse_list_env(
        "LORA_TARGET_MODULES",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    modules_to_save = parse_list_env("LORA_MODULES_TO_SAVE", "")
    config = LoraConfig(
        r=env_int("LORA_R", 16),
        lora_alpha=env_int("LORA_ALPHA", 32),
        lora_dropout=env_float("LORA_DROPOUT", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=modules_to_save or None,
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def build_training_args(output_dir: Path) -> Any:
    from transformers import TrainingArguments

    report_to = parse_list_env("REPORT_TO", "tensorboard" if env_bool("TENSORBOARD", True) else "none")
    if env_bool("WANDB", False) and "wandb" not in report_to:
        report_to.append("wandb")

    args = {
        "output_dir": str(output_dir),
        "num_train_epochs": env_float("EPOCHS", 3.0),
        "per_device_train_batch_size": env_int("BATCH_SIZE", 1),
        "per_device_eval_batch_size": env_int("EVAL_BATCH_SIZE", env_int("BATCH_SIZE", 1)),
        "gradient_accumulation_steps": env_int("GRAD_ACCUM_STEPS", 8),
        "learning_rate": env_float("LR", 2e-4),
        "weight_decay": env_float("WEIGHT_DECAY", 0.0),
        "warmup_ratio": env_float("WARMUP_RATIO", 0.03),
        "logging_steps": env_int("LOGGING_STEPS", 10),
        "save_steps": env_int("SAVE_STEPS", 100),
        "save_total_limit": env_int("SAVE_TOTAL_LIMIT", 3),
        "dataloader_num_workers": env_int("NUM_WORKERS", 2),
        "gradient_checkpointing": env_bool("GRADIENT_CHECKPOINTING", True),
        "remove_unused_columns": False,
        "bf16": resolve_dtype() == torch.bfloat16,
        "fp16": resolve_dtype() == torch.float16,
        "report_to": report_to,
    }

    if env_bool("EVAL", True):
        args["eval_strategy"] = env("EVAL_STRATEGY", "steps")
        args["eval_steps"] = env_int("EVAL_STEPS", env_int("SAVE_STEPS", 100))
    else:
        args["eval_strategy"] = "no"

    args.update(parse_json_env("TRAINING_ARGS_JSON"))

    try:
        return TrainingArguments(**args)
    except TypeError:
        if "eval_strategy" in args:
            args["evaluation_strategy"] = args.pop("eval_strategy")
        return TrainingArguments(**args)


def main() -> int:
    output_dir = Path(env("OUTPUT_DIR", "/workspace/output")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = resolve_dataset()
    log(f"Using dataset directory: {dataset_root}")
    train_records, eval_records = prepare_records(dataset_root, output_dir)

    model, processor = load_model_and_processor()
    if env_bool("GRADIENT_CHECKPOINTING", True) and hasattr(model, "config"):
        model.config.use_cache = False
    model = apply_lora(model)

    from transformers import Trainer

    collator = QwenVLCollator(processor=processor, mask_prompt=env_bool("MASK_PROMPT", True))
    trainer = Trainer(
        model=model,
        args=build_training_args(output_dir),
        train_dataset=RoboflowOCRDataset(train_records),
        eval_dataset=RoboflowOCRDataset(eval_records) if eval_records else None,
        data_collator=collator,
    )

    log("Starting Qwen3.5 OCR fine-tuning...")
    trainer.train(resume_from_checkpoint=env("RESUME") or None)
    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))
    log(f"Training job complete. Final adapter/model written to {output_dir / 'final'}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
