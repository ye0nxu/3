"""공통 환경변수 헬퍼 함수 모듈.

_env_flag / _env_int 가 여러 파일에 중복 정의되어 있던 것을 여기서 통합합니다.
"""
from __future__ import annotations

import os


def env_flag(name: str, default: bool) -> bool:
    """환경변수를 bool로 읽습니다. 0/false/no/off 이면 False, 그 외 설정된 값은 True."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def env_int(name: str, default: int) -> int:
    """환경변수를 int로 읽습니다. 파싱 실패 시 default 반환."""
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
