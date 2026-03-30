"""LLM 추론 런타임 (구 runtime_core.py).

Qwen2.5 모델 로드 및 프롬프트 후보 생성/랭킹 로직.
FastAPI 서버(server.py)는 이 모듈의 build_* 함수를 사용합니다.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

try:
    from utils.env import env_flag
except ModuleNotFoundError:
    def env_flag(name: str, default: bool) -> bool:  # type: ignore[misc]
        raw = os.getenv(name)
        if raw is None:
            return default
        return str(raw).strip().lower() not in {"0", "false", "no", "off"}


SYSTEM_CAND = """You are a visual object labeling assistant for SAM3 (Segment Anything Model 3).
SAM3 supports open-vocabulary segmentation covering ~207,000 unique concepts across all domains.
Task: Given a user description (often in Korean), output the most likely specific visible object labels to detect and segment.

CRITICAL RULES for choosing the label:
- Output the SPECIFIC PART or OBJECT the user wants to segment, NOT a vague parent.
  Example: "자동차 후미등" → "tail light" (NOT "car")
  Example: "강아지 얼굴" → "dog face" (NOT "dog")
  Example: "사과 꼭지" → "apple stem" (NOT "fruit")
- Cover ALL domains: vehicles, animals, people, food, furniture, electronics, nature, clothing, sports, buildings, etc.
- Include modifiers that distinguish the object: color, position, material, state, size.
  Example: "빨간 장미" → "red rose"  |  "나무 의자" → "wooden chair"
- Use concrete English noun phrases (1–4 words) that vision segmentation models recognize.

Output ONLY N lines in this exact format:
<english_prompt> | <korean_gloss>

Rules:
- Exactly N lines. No numbering, no bullets, no blank lines, no extra text.
- english_prompt: 1 to 4 words, concrete noun phrase in English.
- Rank from most specific/accurate to least specific.
- NEVER output generic words: thing, object, area, scene, someone, something, vehicle part.
"""

_FEW_SHOT_MESSAGES: list[dict[str, str]] = [
    # ── 차량 부품 ──
    {
        "role": "user",
        "content": "User input:\n자동차 후미등\n\nReturn exactly 3 lines now.",
    },
    {
        "role": "assistant",
        "content": "tail light | 후미등\nrear lamp | 리어 램프\nbrake light | 브레이크등",
    },
    {
        "role": "user",
        "content": "User input:\n버스 번호판\n\nReturn exactly 3 lines now.",
    },
    {
        "role": "assistant",
        "content": "license plate | 번호판\nfront license plate | 앞 번호판\nregistration plate | 등록 번호판",
    },
    # ── 동물 ──
    {
        "role": "user",
        "content": "User input:\n강아지 귀\n\nReturn exactly 3 lines now.",
    },
    {
        "role": "assistant",
        "content": "dog ear | 강아지 귀\npuppy ear | 강아지 귀\ndog | 강아지",
    },
    {
        "role": "user",
        "content": "User input:\n얼룩 고양이\n\nReturn exactly 3 lines now.",
    },
    {
        "role": "assistant",
        "content": "tabby cat | 얼룩 고양이\nstriped cat | 줄무늬 고양이\ncat | 고양이",
    },
    # ── 음식 ──
    {
        "role": "user",
        "content": "User input:\n그릇에 담긴 라면\n\nReturn exactly 3 lines now.",
    },
    {
        "role": "assistant",
        "content": "ramen bowl | 라면 그릇\ninstant noodles | 인스턴트 라면\nnoodle soup | 국수",
    },
    {
        "role": "user",
        "content": "User input:\n빨간 사과\n\nReturn exactly 3 lines now.",
    },
    {
        "role": "assistant",
        "content": "red apple | 빨간 사과\napple | 사과\nfruit | 과일",
    },
    # ── 사람 / 보호장구 ──
    {
        "role": "user",
        "content": "User input:\n사람이 쓴 안전모\n\nReturn exactly 3 lines now.",
    },
    {
        "role": "assistant",
        "content": "hard hat | 안전모\nsafety helmet | 헬멧\nhelmet | 헤드기어",
    },
    {
        "role": "user",
        "content": "User input:\n노인 손\n\nReturn exactly 3 lines now.",
    },
    {
        "role": "assistant",
        "content": "elderly hand | 노인 손\nhand | 손\naged hand | 주름진 손",
    },
    # ── 가구 / 실내 ──
    {
        "role": "user",
        "content": "User input:\n나무 의자 다리\n\nReturn exactly 3 lines now.",
    },
    {
        "role": "assistant",
        "content": "chair leg | 의자 다리\nwooden chair leg | 나무 의자 다리\nchair | 의자",
    },
    # ── 자연 / 식물 ──
    {
        "role": "user",
        "content": "User input:\n빨간 장미꽃\n\nReturn exactly 3 lines now.",
    },
    {
        "role": "assistant",
        "content": "red rose | 빨간 장미\nrose flower | 장미꽃\nflower | 꽃",
    },
    # ── 전자기기 ──
    {
        "role": "user",
        "content": "User input:\n스마트폰 화면\n\nReturn exactly 3 lines now.",
    },
    {
        "role": "assistant",
        "content": "phone screen | 폰 화면\nsmartphone display | 스마트폰 디스플레이\nmobile phone | 스마트폰",
    },
    # ── 스포츠 ──
    {
        "role": "user",
        "content": "User input:\n축구공\n\nReturn exactly 3 lines now.",
    },
    {
        "role": "assistant",
        "content": "soccer ball | 축구공\nfootball | 축구공\nball | 공",
    },
    # ── 복합 설명: 색상 + 착장 + 사람 ──
    {
        "role": "user",
        "content": "User input:\n빨간색 유니폼을 입은 축구선수\n\nReturn exactly 3 lines now.",
    },
    {
        "role": "assistant",
        "content": "red uniform soccer player | 빨간 유니폼 축구선수\nred soccer player | 빨간 축구선수\nsoccer player | 축구선수",
    },
    {
        "role": "user",
        "content": "User input:\n파란 모자를 쓴 야구선수\n\nReturn exactly 3 lines now.",
    },
    {
        "role": "assistant",
        "content": "blue cap baseball player | 파란 모자 야구선수\nbaseball player | 야구선수\nblue hat | 파란 모자",
    },
    {
        "role": "user",
        "content": "User input:\n흰 셔츠를 입은 남자\n\nReturn exactly 3 lines now.",
    },
    {
        "role": "assistant",
        "content": "man in white shirt | 흰 셔츠 남자\nwhite shirt man | 흰 셔츠 입은 남자\nman | 남자",
    },
    # ── 도로 시설 ──
    {
        "role": "user",
        "content": "User input:\n도로 위 안전 콘\n\nReturn exactly 3 lines now.",
    },
    {
        "role": "assistant",
        "content": "traffic cone | 안전 콘\ncone | 콘\nroad marker | 도로 표지",
    },
]

def _default_model_id() -> str:
    env_remote = str(os.getenv("LLM_REMOTE_MODEL_ID", "")).strip()
    if env_remote:
        return env_remote
    env_local = str(os.getenv("LLM_MODEL_ID", "")).strip()
    if env_local:
        return env_local
    # 하드코딩 폴백: config.local.json["remote"]["models"]["llm"] 로 설정 권장
    remote_dir = os.getenv("LLM_REMOTE_MODEL_ID", "")
    if remote_dir and os.path.isdir(remote_dir):
        return remote_dir
    return "Qwen/Qwen2.5-7B-Instruct"


DEFAULT_MODEL_ID = _default_model_id()
DEFAULT_TOP_N = int(os.getenv("LLM_DEFAULT_TOP_N", "3"))
MAX_TOP_N = int(os.getenv("LLM_MAX_TOP_N", "8"))
MAX_INPUT_CHARS = int(os.getenv("LLM_MAX_INPUT_CHARS", "2000"))
GEN_MAX_NEW_TOKENS = int(os.getenv("LLM_GEN_MAX_NEW_TOKENS", "96"))
GEN_TEMPERATURE = float(os.getenv("LLM_GEN_TEMPERATURE", "0.75"))
GEN_TOP_P = float(os.getenv("LLM_GEN_TOP_P", "0.92"))
SCORE_BATCH_SIZE = max(1, int(os.getenv("LLM_SCORE_BATCH_SIZE", "8")))


ENABLE_RERANK = env_flag("LLM_ENABLE_RERANK", True)
ENABLE_STRICT_RETRY = env_flag("LLM_ENABLE_STRICT_RETRY", True)
GEN_DO_SAMPLE = env_flag("LLM_GEN_DO_SAMPLE", False)


@dataclass
class LLMRuntime:
    model_id: str
    tokenizer: Any
    model: Any
    device: torch.device
    load_mode: str
    lock: threading.Lock


_RUNTIMES: dict[str, LLMRuntime] = {}
_RUNTIMES_LOCK = threading.Lock()


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _supports_bf16() -> bool:
    if not _has_cuda():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 8


def _has_bitsandbytes() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def _resolve_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if _has_cuda() else "cpu")


def _load_runtime(model_id: str) -> LLMRuntime:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if _has_cuda() and _has_bitsandbytes():
        compute_dtype = torch.bfloat16 if _supports_bf16() else torch.float16
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=compute_dtype,
                trust_remote_code=True,
            )
            load_mode = f"cuda-4bit ({compute_dtype})"
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to("cuda")
            load_mode = "cuda-fp16-fallback"
    elif _has_cuda():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to("cuda")
        load_mode = "cuda-fp16"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to("cpu")
        load_mode = "cpu-fp32"

    model.eval()
    device = _resolve_device(model)
    return LLMRuntime(
        model_id=model_id,
        tokenizer=tokenizer,
        model=model,
        device=device,
        load_mode=load_mode,
        lock=threading.Lock(),
    )


def get_runtime(model_id: str | None = None) -> LLMRuntime:
    key = str(model_id or DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    with _RUNTIMES_LOCK:
        runtime = _RUNTIMES.get(key)
        if runtime is None:
            runtime = _load_runtime(key)
            _RUNTIMES[key] = runtime
    return runtime


def get_cached_runtime(model_id: str | None = None) -> LLMRuntime | None:
    key = str(model_id or DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    with _RUNTIMES_LOCK:
        return _RUNTIMES.get(key)


def _extract_lines_pipe(text: str) -> list[tuple[str, str]]:
    generic_words = {
        "thing",
        "object",
        "stuff",
        "area",
        "scene",
        "someone",
        "something",
        "item",
    }

    def _valid_en_prompt(en_text: str) -> bool:
        candidate = en_text.strip()
        if not candidate:
            return False
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 '/,\-]*", candidate):
            return False
        if re.search(r"[.!?]", candidate):
            return False
        words = [word for word in candidate.split() if word]
        if len(words) < 1 or len(words) > 4:
            return False
        if candidate.lower() in generic_words:
            return False
        return True

    def _normalize_korean_gloss(ko_text: str) -> str:
        candidate = ko_text.strip()
        if not candidate:
            return ""
        if re.search(r"[\uac00-\ud7a3]", candidate):
            return candidate
        return ""

    lines: list[tuple[str, str]] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or "|" not in raw:
            continue
        raw = re.sub(r"^\s*[\-\*]?\s*\d+[\)\.\:]?\s*", "", raw).strip()
        left, right = raw.split("|", 1)
        en = re.sub(r"\s{2,}", " ", left.strip())
        ko = _normalize_korean_gloss(re.sub(r"\s{2,}", " ", right.strip()))
        if en and ko and _valid_en_prompt(en):
            lines.append((en, ko))
        elif en and _valid_en_prompt(en):
            lines.append((en, ""))

    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for en, ko in lines:
        key = en.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((en, ko))
    return out


def _clip_en_prompt(en: str, hi: int = 4) -> str:
    words = en.split()
    if len(words) > hi:
        words = words[:hi]
    return " ".join(words)


def _chat_template_to_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = _sanitize_text_input(_coerce_text_input(tokenizer, rendered))
        if text:
            return text
    except Exception:
        pass
    return _manual_chat_prompt_text(tokenizer, messages)


def _manual_chat_prompt_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str((message or {}).get("role", "user")).strip().lower() or "user"
        content = _sanitize_text_input(_coerce_text_input(tokenizer, (message or {}).get("content", "")))
        if content:
            lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


def _coerce_text_input(tokenizer: Any, value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        content = value.get("content")
        if content is not None:
            return _coerce_text_input(tokenizer, content)
        try:
            return json.dumps(dict(value), ensure_ascii=False)
        except Exception:
            return str(value)
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes, bytearray))):
        if value and all(isinstance(item, int) for item in value):
            try:
                return str(tokenizer.decode(list(value), skip_special_tokens=False))
            except Exception:
                pass
        parts = [_coerce_text_input(tokenizer, item) for item in value]
        parts = [part for part in parts if part]
        return "\n".join(parts)
    return str(value)


def _sanitize_text_input(text: str) -> str:
    normalized = str(text or "")
    normalized = normalized.replace("\x00", " ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(ch if (ch >= " " or ch in "\n\t") else " " for ch in normalized)
    return normalized.strip()


def _to_long_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone().to(dtype=torch.long)
    else:
        if hasattr(value, "tolist") and (not isinstance(value, (str, bytes, bytearray))):
            try:
                value = value.tolist()
            except Exception:
                pass
        try:
            tensor = torch.as_tensor(value, dtype=torch.long)
        except Exception:
            return None
    if tensor.ndim == 0:
        return tensor.view(1, 1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    return tensor


def _normalize_tokenized_inputs(tokenized: Any) -> dict[str, torch.Tensor] | None:
    if isinstance(tokenized, Mapping):
        input_ids = _to_long_tensor(tokenized.get("input_ids"))
        if input_ids is None:
            return None
        attention_mask = _to_long_tensor(tokenized.get("attention_mask"))
        if attention_mask is None or attention_mask.shape != input_ids.shape:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    input_ids = _to_long_tensor(tokenized)
    if input_ids is None:
        return None
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids, dtype=torch.long)}


def _tokenize_prompt_text(runtime: LLMRuntime, prompt_text: Any) -> dict[str, Any]:
    normalized = _sanitize_text_input(_coerce_text_input(runtime.tokenizer, prompt_text))
    if not normalized:
        raise ValueError("prompt text is empty")
    try:
        tokenized = runtime.tokenizer(normalized, return_tensors="pt")
        normalized_inputs = _normalize_tokenized_inputs(tokenized)
        if normalized_inputs is not None:
            return normalized_inputs
    except Exception:
        pass
    try:
        tokenized = runtime.tokenizer([normalized], return_tensors="pt", padding=True)
        normalized_inputs = _normalize_tokenized_inputs(tokenized)
        if normalized_inputs is not None:
            return normalized_inputs
    except Exception:
        pass
    try:
        tokenized = runtime.tokenizer(normalized, add_special_tokens=False)
        normalized_inputs = _normalize_tokenized_inputs(tokenized)
        if normalized_inputs is not None:
            return normalized_inputs
    except Exception:
        pass
    try:
        pieces = runtime.tokenizer.tokenize(str(normalized))
        token_ids = runtime.tokenizer.convert_tokens_to_ids(pieces)
    except Exception:
        token_ids = []
    if not token_ids:
        try:
            token_ids = runtime.tokenizer.encode(str(normalized), add_special_tokens=False)
        except Exception:
            token_ids = []
    if not token_ids:
        raise ValueError("prompt text produced no tokens")
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _tokenize_chat_messages(runtime: LLMRuntime, messages: list[dict[str, str]]) -> dict[str, Any]:
    for kwargs in (
        {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt", "return_dict": True},
        {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"},
        {"tokenize": True, "add_generation_prompt": True},
    ):
        try:
            tokenized = runtime.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            continue
        normalized_inputs = _normalize_tokenized_inputs(tokenized)
        if normalized_inputs is not None:
            return normalized_inputs
    prompt_text = _chat_template_to_text(runtime.tokenizer, messages)
    try:
        return _tokenize_prompt_text(runtime, prompt_text)
    except Exception:
        manual_prompt = _manual_chat_prompt_text(runtime.tokenizer, messages)
        if manual_prompt != prompt_text:
            return _tokenize_prompt_text(runtime, manual_prompt)
        raise


@torch.inference_mode()
def _generate_candidates(runtime: LLMRuntime, user_text: str, n: int, debug: bool) -> list[tuple[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_CAND},
        *_FEW_SHOT_MESSAGES,
        {"role": "user", "content": f"User input:\n{user_text}\n\nReturn exactly {n} lines now."},
    ]
    inputs = _tokenize_chat_messages(runtime, messages)
    if runtime.device.type == "cuda":
        inputs = {key: value.to(runtime.device) for key, value in inputs.items()}

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": GEN_MAX_NEW_TOKENS,
        "do_sample": GEN_DO_SAMPLE,
        "pad_token_id": runtime.tokenizer.pad_token_id,
    }
    if GEN_DO_SAMPLE:
        gen_kwargs["temperature"] = GEN_TEMPERATURE
        gen_kwargs["top_p"] = GEN_TOP_P
    generated = runtime.model.generate(**inputs, generation_config=GenerationConfig(**gen_kwargs))
    prompt_len = inputs["input_ids"].shape[1]
    text = runtime.tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()
    if debug:
        print(text, flush=True)

    candidates = _extract_lines_pipe(text)
    if len(candidates) >= n or (not ENABLE_STRICT_RETRY):
        return candidates

    strict_messages = [
        {"role": "system", "content": SYSTEM_CAND + f"\nIf you violate the format, you fail. Return only {n} lines."},
        *_FEW_SHOT_MESSAGES,
        {"role": "user", "content": f"User input:\n{user_text}\n\nReturn exactly {n} lines now."},
    ]
    strict_inputs = _tokenize_chat_messages(runtime, strict_messages)
    if runtime.device.type == "cuda":
        strict_inputs = {key: value.to(runtime.device) for key, value in strict_inputs.items()}
    strict_generated = runtime.model.generate(
        **strict_inputs,
        generation_config=GenerationConfig(
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=runtime.tokenizer.pad_token_id,
        ),
    )
    strict_prompt_len = strict_inputs["input_ids"].shape[1]
    strict_text = runtime.tokenizer.decode(strict_generated[0, strict_prompt_len:], skip_special_tokens=True).strip()
    fallback = _extract_lines_pipe(strict_text)
    seen = {en.lower() for en, _ in candidates}
    for en, ko in fallback:
        key = en.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append((en, ko))
    return candidates


def _normalize_losses_to_probs(losses: list[float]) -> list[float]:
    if not losses:
        return []
    xs = [-loss for loss in losses]
    max_x = max(xs)
    exps = [math.exp(value - max_x) for value in xs]
    total = sum(exps)
    return [value / total for value in exps]


@torch.inference_mode()
def _score_candidates_batched(
    runtime: LLMRuntime,
    user_text: str,
    english_prompts: list[str],
    batch_size: int,
) -> list[float]:
    if not english_prompts:
        return []

    messages = [
        {"role": "system", "content": "Return only the English noun-phrase prompt. No extra text."},
        {"role": "user", "content": f"User input:\n{user_text}\n\nAnswer:"},
    ]
    base = _chat_template_to_text(runtime.tokenizer, messages)
    prompt_len = int(_tokenize_prompt_text(runtime, base)["input_ids"].shape[1])

    losses: list[float] = []
    for start in range(0, len(english_prompts), max(1, batch_size)):
        chunk = english_prompts[start : start + batch_size]
        full_texts = [base + prompt for prompt in chunk]
        tokenized = runtime.tokenizer(full_texts, return_tensors="pt", padding=True)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        if runtime.device.type == "cuda":
            input_ids = input_ids.to(runtime.device)
            attention_mask = attention_mask.to(runtime.device)

        labels = input_ids.clone()
        prefix_cols = min(prompt_len, int(labels.shape[1]))
        labels[:, :prefix_cols] = -100
        labels = labels.masked_fill(attention_mask == 0, -100)

        out = runtime.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.size(0), shift_labels.size(1))
        valid = shift_labels.ne(-100)
        denom = valid.sum(dim=1).clamp(min=1)
        seq_loss = (token_loss * valid).sum(dim=1) / denom
        losses.extend(float(value.item()) for value in seq_loss.detach().cpu())
    return losses


def _make_prompt_ranked(runtime: LLMRuntime, user_text: str, n: int, debug: bool) -> list[tuple[tuple[str, str], float, float]]:
    generated = _generate_candidates(runtime, user_text=user_text, n=n, debug=debug)
    cleaned: list[tuple[str, str]] = []
    seen: set[str] = set()
    for en, ko in generated:
        clipped = _clip_en_prompt(en)
        key = clipped.lower()
        if not clipped or key in seen:
            continue
        seen.add(key)
        cleaned.append((clipped, ko))
        if len(cleaned) >= n:
            break

    if not cleaned:
        raise ValueError("No candidates parsed from model output.")

    if not ENABLE_RERANK:
        losses = [float(index) for index in range(len(cleaned))]
        probs = _normalize_losses_to_probs(losses)
        return list(zip(cleaned, probs, losses))

    losses = _score_candidates_batched(
        runtime,
        user_text=user_text,
        english_prompts=[en for en, _ in cleaned],
        batch_size=SCORE_BATCH_SIZE,
    )
    probs = _normalize_losses_to_probs(losses)
    return sorted(zip(cleaned, probs, losses), key=lambda item: item[1], reverse=True)


def build_health_payload(model_id: str | None = None) -> dict[str, Any]:
    selected_model_id = str(model_id or DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    runtime = get_cached_runtime(selected_model_id)
    if runtime is None:
        return {
            "status": "ok",
            "model_id": selected_model_id,
            "load_mode": "lazy",
            "device": "pending",
            "loaded": False,
        }
    return {
        "status": "ok",
        "model_id": runtime.model_id,
        "load_mode": runtime.load_mode,
        "device": str(runtime.device),
        "loaded": True,
    }


def build_warmup_payload(model_id: str | None = None) -> dict[str, Any]:
    runtime = get_runtime(model_id)
    return {
        "status": "ok",
        "model_id": runtime.model_id,
        "load_mode": runtime.load_mode,
        "device": str(runtime.device),
        "loaded": True,
    }


def build_rank_payload(
    *,
    user_text: str,
    n: int = DEFAULT_TOP_N,
    debug: bool = False,
    model_id: str | None = None,
) -> dict[str, Any]:
    text = str(user_text or "").strip()
    if not text:
        raise ValueError("user_text is empty")
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError(f"user_text exceeds {MAX_INPUT_CHARS} characters")
    top_n = max(1, min(int(n), MAX_TOP_N))
    runtime = get_runtime(model_id)
    with runtime.lock:
        ranked = _make_prompt_ranked(runtime, user_text=text, n=top_n, debug=bool(debug))
    return {
        "model_id": runtime.model_id,
        "load_mode": runtime.load_mode,
        "device": str(runtime.device),
        "items": [
            {
                "english_prompt": en,
                "korean_gloss": ko,
                "probability": float(prob),
                "loss": float(loss),
            }
            for (en, ko), prob, loss in ranked
        ],
    }
