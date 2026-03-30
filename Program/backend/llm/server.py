"""LLM FastAPI 서버.

ML 추론 코드는 runtime.py에서 제공합니다. 이 파일은 HTTP 엔드포인트만 정의합니다.
(구 fastapi_server.py - runtime_core.py 중복 코드 제거)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.llm.runtime import (
    DEFAULT_MODEL_ID,
    DEFAULT_TOP_N,
    MAX_INPUT_CHARS,
    MAX_TOP_N,
    build_health_payload,
    build_rank_payload,
    build_warmup_payload,
    get_runtime,
)


class RankRequest(BaseModel):
    user_text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)
    n: int = Field(default=DEFAULT_TOP_N, ge=1, le=MAX_TOP_N)
    debug: bool = False


class RankedItem(BaseModel):
    english_prompt: str
    korean_gloss: str
    probability: float
    loss: float


class RankResponse(BaseModel):
    model_id: str
    load_mode: str
    device: str
    items: list[RankedItem]


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    model_id = os.getenv("LLM_MODEL_ID", DEFAULT_MODEL_ID)
    get_runtime(model_id)  # 서버 시작 시 모델 로드
    yield


app = FastAPI(
    title="Auto Labeling LLM API",
    version="1.0.0",
    lifespan=_lifespan,
)


@app.get("/health")
def health() -> dict:
    return build_health_payload()


@app.get("/warmup")
def warmup() -> dict:
    return build_warmup_payload()


@app.post("/rank-prompts", response_model=RankResponse)
def rank_prompts(req: RankRequest) -> RankResponse:
    try:
        payload = build_rank_payload(
            user_text=req.user_text,
            n=req.n,
            debug=req.debug,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"추론 실패: {exc}") from exc

    items = [
        RankedItem(
            english_prompt=item["english_prompt"],
            korean_gloss=item["korean_gloss"],
            probability=float(item["probability"]),
            loss=float(item["loss"]),
        )
        for item in payload["items"]
    ]
    return RankResponse(
        model_id=payload["model_id"],
        load_mode=payload["load_mode"],
        device=payload["device"],
        items=items,
    )


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="LLM FastAPI 서버 실행")
    parser.add_argument("--host", default=os.getenv("LLM_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LLM_SERVER_PORT", "8008")))
    parser.add_argument(
        "--model-id",
        default=os.getenv("LLM_MODEL_ID", DEFAULT_MODEL_ID),
        help="HuggingFace 모델 ID 또는 로컬 모델 경로",
    )
    args = parser.parse_args()

    if args.model_id:
        os.environ["LLM_MODEL_ID"] = str(args.model_id)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
