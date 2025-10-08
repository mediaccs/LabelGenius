# =========================
# Imports
# =========================
import os
import sys
import time
import json
import re
import base64
import hashlib
import threading
import csv
import warnings
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import CLIPProcessor, CLIPModel, get_linear_schedule_with_warmup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision import transforms

import sqlitedict
from loguru import logger
from openai import OpenAI

tqdm.pandas()
warnings.filterwarnings("ignore")

# =========================
# Device / Env
# =========================
def pick_device(prefer: str | None = None) -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = pick_device()
print(f"Using device: {device}")

FRIENDLY_MODEL_MESSAGE = (
    "You are encountering this error very likely because OpenAI updated model names "
    "and/or parameters. Please check the current list of available models here:\n"
    "https://platform.openai.com/docs/models\n\n"
    "After you identify the new model to use, please test it in the Playground to see "
    "which parameters are supported:\n"
    "https://platform.openai.com/playground\n\n"
    "If you continue to encounter errors, please email the author "
    "[email removed due to double-blind revision process].")

_SAFE_DEFAULTS = dict(temperature=0.2, max_tokens=512)

_MODEL_FALLBACKS = {
    "gpt-o4-mini": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"],
    "gpt-4o-mini": ["gpt-4.1-mini", "gpt-4o", "gpt-4.1"],
    "gpt-5-nano":  ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1"],
}

def _strip_unknown_kwargs(kwargs: dict) -> dict:
    bad = {"reasoning", "reasoning_effort", "type", "top_logprobs"}  # extend as needed
    return {k: v for k, v in kwargs.items() if k not in bad}

def _is_model_or_param_error(err: Exception) -> bool:
    t = str(err).lower()
    return any(s in t for s in [
        "invalid_request_error", "model_not_found", "unknown parameter", "unrecognized",
        "unsupported", "does not exist", "not permitted", "missing required property",
    ])

def print_env_info():
    import platform, sys as _sys
    try:
        import openai as _oai
        sdk_ver = getattr(_oai, "__version__", "unknown")
    except Exception:
        sdk_ver = "unknown"
    logger.info(f"[ENV] Python: {_sys.version.replace(chr(10),' ')}")
    logger.info(f"[ENV] OpenAI SDK: {sdk_ver}")
    logger.info(f"[ENV] Platform: {platform.platform()}")

# =========================
# CLIP (zero-shot + fine-tune)
# =========================
base_model = "openai/clip-vit-base-patch32"

def classification_CLIP_0_shot(
    text_path,
    img_dir=None,
    mode=None,
    prompt=None,
    text_column=["headline"],
    predict_column="label",
):
    if mode not in ["text", "image", "both"]:
        raise ValueError("mode must be 'text', 'image', or 'both'")
    if prompt is None:
        # NOTE: your original code referenced prompt_D1_CLIP; left unchanged
        prompt = prompt_D1_CLIP  # noqa: F821 (assumed provided elsewhere)

    use_text  = mode in ["text", "both"]
    use_image = mode in ["image", "both"]

    if use_text and not text_path:
        raise ValueError("text_path cannot be empty")
    if use_image and not img_dir:
        raise ValueError("img_dir cannot be empty")

    model     = CLIPModel.from_pretrained(base_model).to(device)
    processor = CLIPProcessor.from_pretrained(base_model)

    if text_path.endswith(".csv") or text_path.endswith(".txt"):
        df = pd.read_csv(text_path)
    elif text_path.endswith(".jsonl"):
        df = pd.read_json(text_path, lines=True)
    elif text_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(text_path)
    else:
        raise ValueError("Unsupported file format")
    print(f"Loaded {len(df)} records")

    with torch.no_grad():
        t_inputs        = processor(text=prompt, return_tensors="pt", padding=True).to(device)
        prompt_features = model.get_text_features(**t_inputs)
        prompt_features = F.normalize(prompt_features, p=2, dim=1)

    predictions = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        sample_text = ""
        if use_text:
            sample_text = " ".join(str(row[c]).strip() for c in text_column if c in row and pd.notna(row[c]))

        image = None
        if use_image:
            img_path = os.path.join(img_dir, f"{row['image_id']}.jpg")
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
            else:
                print(f"Image does not exist: {img_path}")
                image = Image.new("RGB", (224, 224), color="white")

        with torch.no_grad():
            if use_text and use_image:
                inputs      = processor(text=sample_text, images=image, return_tensors="pt").to(device)
                text_f      = model.get_text_features(
                    **{k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask", "position_ids"]}
                )
                img_f       = model.get_image_features(inputs.pixel_values)
                sample_feat = F.normalize((text_f + img_f) / 2, p=2, dim=1)
            elif use_text:
                inputs      = processor(text=sample_text, return_tensors="pt").to(device)
                text_f      = model.get_text_features(**inputs)
                sample_feat = F.normalize(text_f, p=2, dim=1)
            else:
                inputs      = processor(images=image, return_tensors="pt").to(device)
                img_f       = model.get_image_features(inputs.pixel_values)
                sample_feat = F.normalize(img_f, p=2, dim=1)

            sim      = sample_feat @ prompt_features.t()
            pred_cls = sim.argmax().item() + 1
            predictions.append(pred_cls)

    df[predict_column] = predictions
    return df

class CLIPClassifier(torch.nn.Module):
    def __init__(self, clip_model, num_classes, use_text=True, use_image=True):
        super().__init__()
        self.clip_model = clip_model
        self.use_text   = use_text
        self.use_image  = use_image
        self.dropout    = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(clip_model.config.projection_dim, num_classes)

    def forward(self, **inputs):
        feats = []
        if self.use_text:
            text_feats = self.clip_model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None)
            )
            feats.append(text_feats)
        if self.use_image:
            img_feats = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            feats.append(img_feats)

        combined = (feats[0] + feats[1]) / 2 if len(feats) == 2 else feats[0]
        combined = torch.nn.functional.normalize(combined, p=2, dim=-1)
        out      = self.dropout(combined)
        logits   = self.classifier(out)
        return logits, None

class NewsDataset(Dataset):
    def __init__(
        self,
        dataframe,
        processor,
        text_column=None,
        img_dir=None,
        use_text=True,
        use_image=True,
        true_label=None,
        prompt=None,
    ):
        self.df = dataframe
        self.processor = processor
        self.text_column = text_column
        self.img_dir = img_dir
        self.use_text = use_text
        self.use_image = use_image
        self.true_label = true_label
        self.prompt = prompt
        self.max_length = processor.tokenizer.model_max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = None
        if self.use_text:
            if isinstance(self.text_column, list):
                text = " ".join(
                    str(row[col]).strip()
                    for col in self.text_column
                    if col in row and pd.notna(row[col])
                )
            else:
                text = str(row[self.text_column]).strip()
            if self.prompt:
                text = f"{self.prompt} {text}"

        image = None
        if self.use_image and self.img_dir:
            img_id = row.get("image_id", row.name)
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
            else:
                print(f"Image not found, using blank image for {img_id}")
                image = Image.new("RGB", (224, 224), color="white")

        proc_kwargs = {
            "return_tensors": "pt",
            "padding": "max_length",
            "truncation": True,
            "max_length": self.max_length,
        }
        if self.use_text:
            proc_kwargs["text"] = text
        if self.use_image:
            proc_kwargs["images"] = image

        inputs = self.processor(**proc_kwargs)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        if self.true_label:
            label = int(row[self.true_label])
            if self.df[self.true_label].min() == 1:
                label -= 1
        else:
            label = 0

        return inputs, label

def classification_CLIP_finetuned(
    mode=None,
    text_path=None,
    text_column=["headline"],
    img_dir=None,
    prompt=None,
    model_name="best_clip_model.pth",
    batch_size=8,
    num_classes=24,
    predict_column="label",
    true_label=None):
    if mode not in ["text", "image", "both"]:
        raise ValueError("mode must be one of 'text', 'image', or 'both'")

    if text_path.endswith(".csv") or text_path.endswith("txt"):
        df = pd.read_csv(text_path)
    elif text_path.endswith(".jsonl"):
        df = pd.read_json(text_path, lines=True)
    elif text_path.endswith(".xlsx") or text_path.endswith(".xls"):
        df = pd.read_excel(text_path)
    else:
        raise ValueError("Unsupported file format")
    print(f"{len(df)} pieces of data loaded")

    _device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(base_model)

    model = CLIPClassifier(
        CLIPModel.from_pretrained(base_model),
        num_classes,
        use_text=(mode in ["text", "both"]),
        use_image=(mode in ["image", "both"]),
    ).to(_device)

    if not os.path.exists(model_name):
        raise ValueError(f"Model weights file does not exist: {model_name}")

    checkpoint = torch.load(model_name, map_location=_device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = NewsDataset(
        df, text_column=text_column, img_dir=img_dir, processor=processor, prompt=prompt, true_label=true_label
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    with torch.no_grad():
        for _, (inputs, _) in enumerate(tqdm(dataloader, desc="predicting")):
            for k, v in inputs.items():
                inputs[k] = v.to(_device)
            logits, _ = model(**inputs)
            _, predicted = logits.max(1)
            predicted = predicted + 1
            predictions.extend(predicted.cpu().numpy())

    df[predict_column] = predictions
    print("Labeling completed!")
    return df

def finetune_CLIP(
    mode="both",
    text_path="Demo_data/D1_1.csv",
    text_column=["headline"],
    img_dir="Demo_data/D1_imgs_1",
    true_label="section_numeric",
    prompt=None,
    model_name="best_clip_model.pth",
    num_epochs=20,
    batch_size=8,
    learning_rate=1e-5):
    if mode not in ["text", "image", "both"]:
        raise ValueError("mode must be one of 'text', 'image', or 'both'")
    use_text  = mode in ["text", "both"]
    use_image = mode in ["image", "both"]
    if use_text and not text_path:
        raise ValueError("text_path cannot be empty")
    if use_image and not img_dir:
        raise ValueError("img_dir cannot be empty")
    if not use_text and not use_image:
        raise ValueError("At least one of text or image mode must be enabled")

    if text_path.endswith(".csv") or text_path.endswith("txt"):
        df = pd.read_csv(text_path)
    elif text_path.endswith(".jsonl"):
        df = pd.read_json(text_path, lines=True)
    elif text_path.endswith(".xlsx") or text_path.endswith(".xls"):
        df = pd.read_excel(text_path)
    else:
        raise ValueError("Unsupported file format")
    print(f"Loaded {len(df)} records")

    val_size = int(len(df) * 0.2)
    train_df, val_df = df.iloc[:-val_size], df.iloc[-val_size:]

    if min(df[true_label].unique()) == 1:
        num_classes = int(df[true_label].max())
    else:
        num_classes = int(df[true_label].max()) + 1

    print(f"Number of classes: {num_classes}")
    print(f"Use text: {use_text}, Use image: {use_image}")
    print(f"Text columns: {text_column}")
    print(f"Label column: {true_label}")
    if prompt:
        print(f"Using prompt: {prompt}")

    clip_model = CLIPModel.from_pretrained(base_model)
    processor = CLIPProcessor.from_pretrained(base_model)

    train_dataset = NewsDataset(
        train_df, processor, text_column=text_column, img_dir=img_dir,
        use_text=use_text, use_image=use_image, true_label=true_label, prompt=prompt
    )
    val_dataset = NewsDataset(
        val_df, processor, text_column=text_column, img_dir=img_dir,
        use_text=use_text, use_image=use_image, true_label=true_label, prompt=prompt
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CLIPClassifier(clip_model, num_classes, use_text, use_image).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")):
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(**inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}: Loss: {loss.item():.4f} | Acc: {100.*train_correct/train_total:.2f}%")

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                labels = labels.to(device)

                logits, _ = model(**inputs)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        train_loss = train_loss / len(train_loader)
        train_acc  = 100.0 * train_correct / train_total
        val_loss   = val_loss / len(val_loader)
        val_acc    = 100.0 * val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_accuracy": best_accuracy}, model_name)
            print(f"Model saved! Best validation accuracy: {best_accuracy:.2f}%")

    print(f"Fine-tuning complete! Best validation accuracy: {best_accuracy:.2f}%")
    return best_accuracy

# =========================
# Shared utilities
# =========================
def price_estimation(
    response,
    num_rows: int,
    input_cost_per_million: float,
    output_cost_per_million: float,
    num_votes: int = 1) -> float:
    usage = getattr(response, "usage", None)
    if usage is None:
        usage = response.get("usage", {})

    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    else:
        input_tokens = getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0))
        output_tokens = getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0))

    in_price  = input_cost_per_million / 1_000_000
    out_price = output_cost_per_million / 1_000_000

    cost_per_call = input_tokens * in_price + output_tokens * out_price
    total_calls   = num_rows * num_votes
    total         = cost_per_call * total_calls
    low, high     = total * 0.90, total * 1.10

    print(f"\n🧮 Estimated Cost for {total_calls:,} calls ({num_rows:,} rows × {num_votes} votes)")
    print(f"• Avg prompt tokens/call:     {input_tokens}")
    print(f"• Avg completion tokens/call: {output_tokens}")
    print(f"• Pricing ($/1M tokens): prompt=${input_cost_per_million}, completion=${output_cost_per_million}")
    print(f"💰 Total: ${total:.4f}    (±10% → ${low:.4f}–${high:.4f})\n")
    return total

def image_file_to_data_url(path: str) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logger.error(f"[image_data_url] {path}: {e}")
        return ""

# =========================
# GPT classification - data classes & cache
# =========================
@dataclass
class ClassificationQuestion:
    prompt: str
    model_name: str
    valid_values: list[str]
    reasoning_effort: str | None
    column_4_labeling: str            # "text_class" | "image_class" | "final_class"
    text: str                         # text snippet OR data URL OR multimodal combo
    label_num: int = 1
    max_verify_retry: int = 2

    def get_key(self) -> str:
        parts = [
            self.prompt,
            self.model_name,
            ",".join(self.valid_values),
            str(self.reasoning_effort),
            self.column_4_labeling,
            self.text or "",
            str(self.label_num),
            str(self.max_verify_retry),
        ]
        return hashlib.md5("|".join(parts).encode()).hexdigest()

@dataclass
class ClassificationTask:
    column: str
    prompt: str
    model_name: str
    valid_values: list[str]
    reasoning_effort: str | None
    column_4_labeling: str
    label_num: int = 1
    once_verify_num: int = 1
    max_verify_retry: int = 3

    def create_question(self, content: str) -> ClassificationQuestion:
        return ClassificationQuestion(
            prompt=self.prompt,
            model_name=self.model_name,
            valid_values=self.valid_values,
            reasoning_effort=self.reasoning_effort,
            column_4_labeling=self.column_4_labeling,
            text=content,
            label_num=self.label_num,
            max_verify_retry=self.max_verify_retry,
        )

class DBCache:
    def __init__(self):
        self.db = sqlitedict.SqliteDict("db.sqlite", autocommit=True)

    def add(self, q: ClassificationQuestion, res):
        self.db[q.get_key()] = res

    def get(self, q: ClassificationQuestion):
        return self.db.get(q.get_key())

class MaxRetryException(Exception):
    pass

# =========================
# >>> GPTClassifier (TEMPERATURE SUPPORT ADDED FOR GPT-4) <<<
# =========================
class GPTClassifier:
    def __init__(self, client: OpenAI):
        self.client = client
        self.cache  = DBCache()

    # -- Helpers: Responses-API wrapping --
    @staticmethod
    def _to_responses_input(messages):
        converted = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            items = []
            if isinstance(content, str):
                items.append({"type": "input_text", "text": content})
            elif isinstance(content, list):
                for it in content:
                    t = it.get("type", "text")
                    if t in ("text", "input_text"):
                        items.append({"type": "input_text", "text": it.get("text", "")})
                    elif t in ("image_url", "input_image"):
                        v = it.get("image_url")
                        url = v.get("url") if isinstance(v, dict) else v
                        items.append({"type": "input_image", "image_url": str(url)})
                    else:
                        items.append({"type": "input_text", "text": str(it)})
            else:
                items.append({"type": "input_text", "text": str(content)})
            converted.append({"role": role, "content": items})
        return converted

    @staticmethod
    def _wrap_responses_output(resp):
        try:
            text = resp.output_text
        except Exception:
            try:
                text = "".join([o.get("content", "") for o in resp.output[0].get("content", [])])
            except Exception:
                text = ""
        class _DummyChoice:
            def __init__(self, txt): self.message = type("m", (), {"content": txt})
        return type("R", (), {"choices": [_DummyChoice(text)]})

    # --- Core API call (modified to accept temperature for GPT-4) ---
    def fetch(self, messages, model, reasoning_effort, n, temperature: float | None = None):
        """
        Resilient call:
          - Uses Responses API for gpt-5* models (accepts reasoning.effort)
          - Uses Chat Completions for other models
          - GPT-4 family: now accepts 'temperature'
        """
        candidates = [model] + _MODEL_FALLBACKS.get(model, [])
        last_err = None

        for pass_id in (0, 1):
            for cand in candidates:
                try:
                    if cand.startswith("gpt-5"):
                        inputs = self._to_responses_input(messages)
                        kwargs = dict(
                            model=cand,
                            input=inputs,
                            reasoning={"effort": (reasoning_effort or "minimal")},
                        )
                        if pass_id == 1:
                            kwargs.pop("reasoning", None)
                        resp = self.client.responses.create(**kwargs)
                        return self._wrap_responses_output(resp)

                    # GPT-4 family (and others) -> Chat Completions; add temperature if provided
                    kwargs = dict(model=cand, messages=messages, n=n)
                    if temperature is not None:
                        kwargs["temperature"] = float(temperature)
                    if pass_id == 1:
                        kwargs.update(_SAFE_DEFAULTS)
                    return self.client.chat.completions.create(**kwargs)

                except Exception as e:
                    last_err = e
                    if _is_model_or_param_error(e):
                        continue
                    break

        logger.error("OpenAI call failed across requested model and fallbacks. Showing guidance.")
        logger.error(FRIENDLY_MODEL_MESSAGE)
        if last_err:
            logger.exception(last_err)
        raise MaxRetryException("Failed after retries and fallbacks")

    # -- SINGLE call → parsed labels (now forwards temperature) --
    def classify(self, q: ClassificationQuestion, n: int, temperature: float | None = None):
        if q.column_4_labeling == "text_class":
            content = [
                {"type": "text", "text": str(q.prompt)},
                {"type": "text", "text": str(q.text)},
            ]
        elif q.column_4_labeling == "image_class":
            content = [
                {"type": "text", "text": str(q.prompt)},
                {"type": "image_url", "image_url": {"url": q.text}},
            ]
        else:
            if "data:image" in q.text:
                txt, img = q.text.split("data:image", 1)
                img = "data:image" + img
            else:
                txt, img = q.text, ""
            img = re.sub(r"\s+", "", img)
            content = [{"type": "text", "text": f"{str(q.prompt)}\nText: {str(txt).strip()}"}]
            if img.startswith("data:image"):
                content.append({"type": "image_url", "image_url": {"url": img}})

        resp = self.fetch(
            [{"role": "user", "content": content}],
            q.model_name,
            q.reasoning_effort or "minimal",
            n,
            temperature=temperature,
        )

        parsed = []
        for reply in (c.message.content.strip() for c in resp.choices):
            if reply.startswith("[") and reply.endswith("]"):
                try:
                    arr = json.loads(reply)
                    if isinstance(arr, list) and len(arr) == q.label_num and all(str(x) in q.valid_values for x in arr):
                        parsed.append([int(x) for x in arr])
                        continue
                except Exception:
                    pass
            flat = re.findall(r"\b[01]\b", reply)
            if len(flat) == q.label_num:
                parsed.append([int(x) for x in flat])
                continue
            matches = [p.strip(' ."') for p in reply.split(",") if p.strip(' ."') in q.valid_values]
            if matches:
                parsed.append(matches[: q.label_num])
                continue
            m = re.search(r"\b(\d{1,2})\b", reply)
            if m and m.group(1) in q.valid_values:
                parsed.append([m.group(1)])

        if not parsed:
            logger.error(f"No valid labels parsed. Raw reply: {resp.choices}")
        return parsed

    # -- majority vote / cache (now forwards temperature) --
    def multi_verify(self, q: ClassificationQuestion, n, retry=1, freq=None, temperature: float | None = None):
        res = self.classify(q, n, temperature=temperature)
        if not res and retry < q.max_verify_retry:
            return self.multi_verify(q, n, retry + 1, temperature=temperature)
        if not res:
            raise MaxRetryException("No valid responses")

        if isinstance(res[0], list) and len(res[0]) == q.label_num:
            self.cache.add(q, res[0])
            return res[0]

        freq = freq or {}
        for r in res:
            lbl = r[0]
            freq[lbl] = freq.get(lbl, 0) + 1
        top = sorted(freq, key=freq.get, reverse=True)[: q.label_num]
        self.cache.add(q, top)
        return top

    # -- DataFrame helper (now forwards temperature) --
    def classify_df(self, df: pd.DataFrame, task: ClassificationTask,
                    return_sample_response=False, temperature: float | None = None):
        out, sample = [], None
        for rec in tqdm(df.to_dict("records"), desc=f"Classifying {task.column_4_labeling}", unit="item"):
            q = task.create_question(rec.get(task.column, ""))

            if return_sample_response and sample is None:
                sample = self.fetch(
                    [{"role": "user",
                      "content": [{"type": "text", "text": str(q.prompt)},
                                  {"type": "text", "text": str(q.text)}]}],
                    q.model_name,
                    q.reasoning_effort or "minimal",
                    1,
                    temperature=temperature,
                )

            try:
                lbl = self.multi_verify(q, task.once_verify_num, temperature=temperature)
            except Exception as e:
                logger.error(e)
                lbl = ["99"] * task.label_num

            rec[task.column_4_labeling] = lbl
            out.append(rec)

        df_out = pd.DataFrame(out)
        return (df_out, sample) if return_sample_response else df_out

# =========================
# classification_GPT (signature now includes temperature; forwarded)
# =========================
def classification_GPT(
    text_path: str | None = None,
    category: list[str] | None = None,
    image_dir: str | None = None,
    prompt: list[str] | str | None = None,
    column_4_labeling: list[str] | None = None,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,     # <-- added
    mode: str = "both",                   # "text" | "image" | "both"
    output_column_name: str = "label",
    num_themes: int = 1,
    num_votes: int = 1,
    batch_size: int = 5,
    wait_time: float = 1.2
) -> pd.DataFrame:

    print_env_info()

    category = [str(c) for c in (category or [])]
    valid_efforts = {"minimal", "low", "medium", "high"}
    if reasoning_effort is None:
        reasoning_effort = "minimal"
    if reasoning_effort not in valid_efforts:
        raise ValueError(f"reasoning_effort must be one of {valid_efforts}, got {reasoning_effort!r}")
    is_reasoning = reasoning_effort != "minimal"

    # -- load data
    if mode == "image":
        if text_path and text_path.lower().endswith(".json"):
            df0 = pd.DataFrame(json.load(open(text_path, encoding="utf-8")))
            if "image_dir" not in df0.columns:
                df0["image_dir"] = df0["image_id"].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))
        else:
            if not image_dir:
                raise ValueError("image_dir required (mode='image')")
            files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            df0 = pd.DataFrame({
                "image_id":  [os.path.splitext(f)[0] for f in files],
                "image_dir": [os.path.join(image_dir, f) for f in files],
            })
    else:
        if not text_path:
            raise ValueError("text_path required")
        ext = os.path.splitext(text_path)[1].lower()
        if ext == ".json":
            df0 = pd.DataFrame(json.load(open(text_path, encoding="utf-8")))
        elif ext == ".csv":
            df0 = pd.read_csv(text_path)
        elif ext in (".xls", ".xlsx"):
            df0 = pd.read_excel(text_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        if mode == "both" and "image_dir" not in df0.columns:
            if not image_dir:
                raise ValueError("image_dir required for mode='both'")
            df0["image_dir"] = df0["image_id"].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))

    df = df0.copy()
    df["text_content"] = (df.apply(
        lambda r: " ".join(
            str(r[c]) for c in (column_4_labeling or [])
            if c in r and pd.notna(r[c])
        ), axis=1) if column_4_labeling else "")

    if mode in ("image", "both"):
        df["image_data_url"] = df["image_dir"].apply(image_file_to_data_url)
    else:
        df["image_data_url"] = ""

    if mode == "both":
        df["final_input"] = df["text_content"] + "\n" + df["image_data_url"]
    elif mode == "image":
        df["final_input"] = df["image_data_url"]
    else:
        df["final_input"] = df["text_content"]

    if isinstance(prompt, str) and prompt.strip():
        base_prompt = prompt.strip()
    else:
        defs = "; ".join(f"{c}: {d}" for c, d in zip(category, prompt)) if prompt else ""
        base_prompt = (
            f"Themes: {', '.join(category)}. {defs} "
            f"Return the top {num_themes} theme number(s) "
            "(or an 8-element JSON array of 0/1). No extra words."
        )

    if mode == "text":
        tasks = [("text_content",  "text_class",  base_prompt)]
    elif mode == "image":
        tasks = [("image_data_url","image_class", base_prompt)]
    else:
        tasks = [("final_input",   "final_class", base_prompt)]

    clf   = GPTClassifier(OpenAI(api_key=api_key) if api_key else OpenAI())
    first = True

    outputs = []
    n = len(df)
    for start in range(0, n, max(1, batch_size)):
        end = min(n, start + max(1, batch_size))
        sub = df.iloc[start:end].copy()

        for col, lab, pr in tasks:
            task = ClassificationTask(
                column=col, prompt=pr, model_name=model,
                valid_values=category, reasoning_effort=reasoning_effort,
                column_4_labeling=lab, label_num=num_themes,
                once_verify_num=num_votes, max_verify_retry=num_votes,
            )
            if first:
                sub, _ = clf.classify_df(sub, task, return_sample_response=True, temperature=temperature)
                first = False
            else:
                sub = clf.classify_df(sub, task, temperature=temperature)

        outputs.append(sub)
        time.sleep(wait_time)

    df = pd.concat(outputs, ignore_index=True)

    df.rename(columns={lab: output_column_name}, inplace=True)

    if isinstance(df[output_column_name].iloc[0], list) and len(df[output_column_name].iloc[0]) == num_themes:
        raw = output_column_name + "_raw"
        df[raw] = df[output_column_name]
        for i in range(num_themes):
            df[f"{output_column_name}_{i+1}"] = df[output_column_name].apply(
                lambda v, idx=i: int(v[idx]) if isinstance(v, list) else np.nan
            )

    df[output_column_name] = df[output_column_name].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x
    )
    return df

# =========================
# Fine-tune prep & jobs
# =========================
def generate_GPT_finetune_jsonl(
    df: pd.DataFrame,
    output_path: str = "classification_result.jsonl",
    label_col: str | list[str] = "true_class",
    system_prompt: str | list[str] | None = None,
    input_col: str | list[str] = "text_content") -> None:

    if isinstance(system_prompt, (list, tuple)):
        sys_txt = "\n".join(system_prompt).strip()
    else:
        sys_txt = system_prompt.strip() if isinstance(system_prompt, str) else None

    label_cols = list(label_col) if isinstance(label_col, (list, tuple)) else [label_col]
    input_cols = list(input_col) if isinstance(input_col, (list, tuple)) else [input_col]

    for c in input_cols:
        if c not in df.columns:
            raise ValueError(f"Missing input column: {c}")
    for c in label_cols:
        if c not in df.columns:
            logger.warning(f"Missing label column: {c}; skipping export")
            return

    with open(output_path, "w", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            parts = [str(row[c]) for c in input_cols if pd.notna(row[c])]
            if not parts:
                continue
            user_text = " ".join(parts).strip()

            raw = [row[c] for c in label_cols]
            flat = []
            for x in raw:
                flat.extend(x if isinstance(x, (list, np.ndarray)) else [x])
            clean = []
            for v in flat:
                if pd.isna(v):
                    continue
                try:
                    clean.append(str(int(v)))
                except:
                    clean.append(str(v))
            if not clean:
                continue
            label_str = ", ".join(clean)

            msgs = []
            if sys_txt:
                msgs.append({"role": "system",    "content": sys_txt})
            msgs.append({"role": "user",      "content": user_text})
            msgs.append({"role": "assistant", "content": label_str.strip()})

            fout.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")

def finetune_GPT(
    training_file_path: str,
    model: str = "gpt-4o-mini",
    method_type: str = "supervised",
    hyperparameters: dict = None,
    poll_interval: int = 15,
    max_wait_time: int = 60*60,
    api_key: str = None) -> str:
    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    filename = os.path.basename(training_file_path)
    with open(training_file_path, 'rb') as f:
        upload_resp = client.files.create(file=(filename, f), purpose="fine-tune")
    try:
        training_file_id = upload_resp.id
    except AttributeError:
        training_file_id = upload_resp['id']

    method = {"type": method_type, method_type: {"hyperparameters": hyperparameters or {}}}

    job_resp = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=model,
        method=method
    )
    try:
        job_id = job_resp.id
    except AttributeError:
        job_id = job_resp['id']
    print("Started fine-tune job", job_id)

    elapsed = 0
    while elapsed < max_wait_time:
        status = client.fine_tuning.jobs.retrieve(job_id)
        try:
            st = status.status
        except AttributeError:
            st = status['status']
        print(f"[{elapsed}s] status={st}")

        if st == "succeeded":
            try:
                fine_model = status.fine_tuned_model
            except AttributeError:
                fine_model = status['fine_tuned_model']
            print("✅ succeeded:", fine_model)
            return fine_model

        if st in ("failed", "canceled", "cancelled"):
            try:
                error_info = status.error
            except AttributeError:
                error_info = status.get('error', None)
            print(f"❌ Job {job_id} ended with {st}. Error info: {error_info}")
            raise RuntimeError(f"Fine-tune job {job_id} ended with status '{st}'"
                               + (f": {error_info}" if error_info else ""))

        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(f"Job {job_id} didn’t finish within {max_wait_time}s")

# =========================
# Verification
# =========================
def auto_verification(
    df: pd.DataFrame,
    predicted_cols,
    true_cols,
    category: list = None,
    sample_size: int = None) -> dict:

    def _extract_scalar(x):
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        try:
            return int(x)
        except:
            return np.nan

    def _normalize_series(col: pd.Series) -> pd.Series:
        if category and col.dtype == object and not pd.api.types.is_list_like(col.iloc[0]):
            mapping = {name: idx + 1 for idx, name in enumerate(category)}
            col = col.map(mapping).astype(float)
        if col.dtype == object or pd.api.types.is_list_like(col.iloc[0]):
            col = col.map(_extract_scalar)
        return col

    if isinstance(predicted_cols, str):
        predicted_cols = [predicted_cols]
    if isinstance(true_cols, str):
        true_cols = [true_cols]
    if len(predicted_cols) != len(true_cols):
        raise ValueError("The number of predicted columns must match the number of true columns.")

    total_correct, total_count = 0, 0
    overall_results = {}

    for p, t in zip(predicted_cols, true_cols):
        if p not in df or t not in df:
            raise KeyError(f"Column '{p}' or '{t}' not in DataFrame")

        s_pred = _normalize_series(df[p])
        s_true = _normalize_series(df[t])

        valid = pd.concat([s_pred, s_true], axis=1).dropna()
        if len(valid) == 0:
            print(f"No valid data to compare for '{p}' vs '{t}'. Skipping.")
            continue

        if sample_size and len(valid) > sample_size:
            valid = valid.sample(sample_size, random_state=42)

        y_pred, y_true = valid.iloc[:, 0], valid.iloc[:, 1]

        results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
            "precision_micro": precision_score(y_true, y_pred, average='micro', zero_division=0),
            "recall_micro": recall_score(y_true, y_pred, average='micro', zero_division=0),
            "f1_micro": f1_score(y_true, y_pred, average='micro', zero_division=0),
            "report": classification_report(y_true, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred)
        }

        print(f"\n== Verification of '{p}' vs. '{t}' ==")
        print(f"Accuracy:   {results['accuracy']:.2%}")
        print(f"Macro F1:   {results['f1_macro']:.2%}")
        print(f"Micro  F1:  {results['f1_micro']:.2%}")
        print("\nFull classification report:")
        print(results["report"])
        print("\nConfusion matrix:")
        print(results["confusion_matrix"])

        total_correct += (y_pred == y_true).sum()
        total_count   += len(valid)
        overall_results[f"{p} vs {t}"] = results

    overall_accuracy = total_correct / total_count if total_count else 0.0
    print(f"\n>> Overall accuracy: {overall_accuracy:.2%}")
    overall_results["overall_accuracy"] = overall_accuracy
    return overall_results
