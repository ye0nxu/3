# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 실행 방법

```bash
# 데스크톱 애플리케이션 실행
conda activate PJ_310_LLM_SAM3
python run.py

# LLM FastAPI 서버 단독 실행 (기본: http://127.0.0.1:8008)
conda activate PJ_310_LLM_SAM3
python run_llm_server.py

# 전체 테스트 실행
python -m unittest discover -s tests -v

# 단일 테스트 파일 실행
python -m unittest tests.test_llm_api_client -v

# 문법 검사
python -m py_compile run.py run_llm_server.py backend/labeling/sam3_runner.py
```

## 환경 설정

- **conda 환경**: `PJ_310_LLM_SAM3` (Python 3.12, PyTorch 2.7.0 + CUDA 12.6)
- **환경 파일**: `environment_final.yml` (팀 배포용), `environment_sam3.yml` (SAM3 전용)
- **로컬 설정**: `config.local.json` (SSH 자격증명, 모델 경로, 기능 토글 포함 — git에 포함 안 됨)
  - 템플릿: `config.local.example.json`

## 아키텍처 개요

**2-티어 하이브리드 시스템**: 로컬 PC(PyQt6 UI) ↔ 원격 GPU 서버(SSH/SFTP)

```
로컬 PC                          원격 GPU 서버
─────────────────────────────    ──────────────────────────
PyQt6 UI (app/)              ↔   SAM3 추적 (segment-anything-3)
필터링 엔진 (backend/filters/)    Qwen2.5-7B LLM 추론
데이터셋 내보내기                  YOLO/RT-DETR 학습
로컬 모델 테스트                   원격 학습
LLM 서버 (FastAPI, 로컬 선택사항)
```

## 핵심 워크플로우

1. **비디오 로드 → SAM3 자동 레이블링** (`backend/labeling/sam3_runner.py`)
   - SSH로 비디오 업로드, 원격 SAM3 실행, `tracks.csv` 다운로드
2. **필터링** (`backend/filters/engine.py`)
   - 흐림/노출/중복 해시 기반 Keep/Hold/Drop 3-상태 분류
3. **데이터셋 내보내기** (`app/studio/export_ops.py`)
   - YOLO 포맷 (train/valid/test 분할)
4. **모델 학습** (`app/studio/workers.py:YoloTrainWorker`)
   - 로컬 또는 원격 SSH 서버에서 Ultralytics YOLO v8/v11 또는 RT-DETR 학습
5. **모델 테스트** (`app/studio/workers.py:ModelTestWorker`)

## 디렉터리 구조

| 디렉터리 | 역할 |
|---|---|
| `app/` | PyQt6 UI — 메인 윈도우, 페이지 빌더, 커스텀 위젯, QThread 워커 |
| `app/studio/` | 작업별 UI 로직 (내보내기, 미리보기, 비디오, 세션 처리) |
| `app/studio/mixins/` | 윈도우 믹스인 (레이아웃, 세션, 학습 플로우) |
| `backend/labeling/` | SAM3 실행 오케스트레이션, 아티팩트 파싱 |
| `backend/filters/` | 품질 필터링 엔진, 퍼셉추얼 해싱 |
| `backend/llm/` | Qwen2.5 추론, FastAPI 서버, 프롬프트 생성 |
| `backend/storage/` | SSH 기반 원격 파일 동기화 |
| `backend/pipelines/` | 미리보기 후처리 파이프라인 |
| `core/` | 공유 데이터 모델 (FrameAnnotation, BoxAnnotation, TrackObject) |
| `config/` | `config.local.json` 로드 및 환경 변수 주입 |
| `vendor/` | 써드파티 코드 (SAM3) |

## 핵심 파일 위치

- **메인 윈도우**: `app/window.py` (1400+ 줄)
- **백그라운드 워커**: `app/studio/workers/` 패키지 (워커별 파일 분리: `auto_label_worker.py`, `export_worker.py`, `merge_worker.py`, `model_test_worker.py`, `train_worker.py`)
- **SAM3 실행기**: `backend/labeling/sam3_runner.py`
- **필터 엔진**: `backend/filters/engine.py` (SampleFilterEngine)
- **LLM 서버**: `backend/llm/server.py` (FastAPI, `/rank-prompts` 엔드포인트)
- **데이터 모델**: `core/dataset.py`, `core/models.py`, `core/tracking.py`
- **경로 상수**: `core/paths.py`

## 중요 설계 패턴

- **UI 스레드 분리**: 모든 무거운 작업은 `app/studio/workers.py`의 QThread 워커에서 실행. 워커는 `pyqtSignal`로 진행 상황을 UI에 전달.
- **LLM 서버 자동 관리**: `app/main.py`의 `LLMServerManager`가 앱 시작/종료 시 LLM 서버를 자동 시작/종료 (`config.local.json`의 `llm_enabled` 토글로 제어).
- **Keep/Hold/Drop 상태머신**: `core/tracking.py:TrackObject` — 필터링 결과의 3-상태 분류.
- **원격 실행**: `backend/storage/remote_storage.py`와 `backend/labeling/sam3_runner.py`가 paramiko를 통해 SSH/SFTP로 원격 서버와 통신.
- **OpenMP 충돌 방지**: Windows에서 PyTorch(libomp)와 MKL(libiomp5md) 충돌 방지를 위해 `KMP_DUPLICATE_LIB_OK=TRUE`가 자동 설정됨 (`config/settings.py`).

## keep / hold / drop 기준

| 상태 | 색상 | 의미 |
|------|------|------|
| `keep` | 초록 | SAM3 직접 감지, confidence ≥ 임계값 |
| `hold` | 노랑 | tracker 예측 또는 중간 confidence |
| `drop` | 파랑 | 낮은 confidence, 블러, 중복 해시 탈락 |

## 저장 경로

| 경로 | 설명 |
|------|------|
| `assets/dataset_save_dir/` | 내보낸 YOLO 데이터셋 |
| `assets/crop_save_dir/` | 객체 크롭 이미지 |
| `assets/models/YOLO_models/` | YOLO 학습 모델 (.pt) |
| `assets/models/RT-DETR_models/` | RT-DETR 학습 모델 (.pt) |
| `assets/styles/main_window.qss` | 다크 테마 QSS |
| `assets/styles/main_window_light.qss` | 라이트 테마 QSS |
| `runs/train/` | 학습 실행 결과 |
