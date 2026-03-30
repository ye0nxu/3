# Auto Labeling Tool ver11

## 개요

PyQt6 기반 Windows 데스크탑 자동 라벨링 도구입니다.

- 영상(비디오)에서 객체를 **자연어 프롬프트(한국어 포함)** 로 탐지·추적하여 **YOLO 형식 데이터셋**을 자동 생성합니다.
- 무거운 AI 추론(SAM3 추적, Qwen2.5 프롬프트 생성, YOLO/RT-DETR 학습)은 **원격 GPU 서버(SSH)** 에서 실행됩니다.
- 로컬 PC는 PyQt6 UI, 결과 후처리, 데이터셋 내보내기, 모델 테스트만 담당합니다.

---

## 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────────────┐
│                        로컬 PC (Windows)                              │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │                   PyQt6 데스크탑 UI                          │     │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐  │     │
│  │  │ 홈/영상  │  │2단계라벨 │  │  학습 페이│  │ 모델 테스│  │     │
│  │  │ 로드 페이│  │링 페이지 │  │  지       │  │ 트 페이지│  │     │
│  │  └──────────┘  └──────────┘  └───────────┘  └───────────┘  │     │
│  └──────────────────────┬──────────────────────────────────────┘     │
│                         │                                            │
│  ┌──────────────────────▼──────────────────────────────────────┐     │
│  │                     백엔드 레이어                             │     │
│  │  ┌─────────────┐  ┌────────────┐  ┌──────────┐  ┌────────┐ │     │
│  │  │ LLM Client  │  │ SAM3 Runner│  │ Filter   │  │ Export │ │     │
│  │  │ (prompting) │  │(SSH 오케스 │  │ Engine   │  │ Worker │ │     │
│  │  │             │  │ 트레이션)  │  │(keep/drop│  │(YOLO   │ │     │
│  │  └──────┬──────┘  └─────┬──────┘  │ /hold)   │  │format) │ │     │
│  │         │               │         └──────────┘  └────────┘ │     │
│  └─────────┼───────────────┼────────────────────────────────── ┘     │
│            │               │                                         │
│  ┌─────────▼──────┐        │  SSH (paramiko)                         │
│  │ LLM FastAPI 서버│        │                                         │
│  │ Qwen2.5-7B     │        │                                         │
│  │ (로컬 선택적)  │        │                                         │
│  └────────────────┘        │                                         │
└───────────────────────────┼──────────────────────────────────────────┘
                            │ SSH
┌───────────────────────────▼──────────────────────────────────────────┐
│                     원격 GPU 서버                                      │
│                                                                      │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐  │
│  │  SAM3 Tracker    │   │  Qwen2.5-7B LLM  │   │  YOLO / RT-DETR  │  │
│  │  (Text-guided    │   │  (프롬프트 생성·  │   │  학습 엔진       │  │
│  │   Object Track)  │   │   랭킹)           │   │  (Ultralytics)   │  │
│  └──────────────────┘   └──────────────────┘   └──────────────────┘  │
│                                                                      │
│  결과: tracks.csv / summary.json → SSH로 로컬 전송                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **자동 라벨링** | SAM3(Segment Anything Model 3) + ByteTrack으로 영상 내 객체 자동 탐지·추적 |
| **LLM 프롬프트 생성** | Qwen2.5-7B가 한국어 입력을 영어 SAM 프롬프트로 변환·랭킹 |
| **품질 필터링** | 블러·과/저노출·중복(해시) 기반 자동 필터링 (keep/hold/drop 3단계) |
| **2단계 라벨링** | 기존 세션 결과를 기반으로 신규 객체 추가 라벨링 |
| **데이터셋 내보내기** | YOLO 형식(train/valid/test 분할) 자동 저장 |
| **모델 학습** | YOLO / RT-DETR 학습 (로컬 또는 원격 GPU 서버) |
| **모델 테스트** | 학습된 모델로 영상에서 실시간 추론·결과 시각화 |
| **원격 스토리지** | SSH를 통한 데이터셋·모델 파일 원격 동기화 |
| **다크/라이트 테마** | QSS 기반 테마 전환 |

---

## 디렉터리 구조

```text
Auto_Labeling_Tool_ver10/
├── run.py                        # 앱 실행 진입점
├── run_llm_server.py             # LLM FastAPI 서버 실행 진입점
├── requirements.txt              # pip 의존성
├── environment_final.yml         # 팀 배포용 conda 환경 (PJ_310_LLM_SAM3)
├── environment_sam3.yml          # SAM3 전용 conda 환경
├── config.local.json             # 로컬 환경 설정 (git 미포함, 개인 정보)
├── config.local.example.json     # 설정 템플릿
│
├── app/                          # 프론트엔드 (PyQt6 UI)
│   ├── main.py                   # MainController (LLMServerManager 포함 앱 부트스트랩)
│   ├── window.py                 # AutoLabelStudioWindow (메인 윈도우 클래스)
│   ├── studio/                   # 스튜디오 워크플로우 모듈
│   │   ├── config.py             # 테마·경로·기본값 상수
│   │   ├── export_ops.py         # 데이터셋 내보내기 연산
│   │   ├── preview_ops.py        # 썸네일 미리보기 연산
│   │   ├── runtime.py            # cv2·yaml·필터 런타임 임포트 게이트웨이
│   │   ├── utils.py              # YAML·경로·학습 유틸
│   │   ├── video_ops.py          # 비디오 프레임·BBox 시각화 연산
│   │   ├── workers.py            # QThread 기반 백그라운드 워커 모음
│   │   └── mixins/               # 윈도우 믹스인 (세션·레이아웃·학습 흐름)
│   │       ├── session_processing.py
│   │       ├── setup_layout.py
│   │       └── training_flow.py
│   ├── ui/                       # UI 구성 요소
│   │   ├── main_window.py        # 메인 윈도우 팩토리
│   │   ├── pages/                # 페이지별 빌더·네비게이션 함수
│   │   ├── dialogs/              # 다이얼로그 (학습 YAML 선택 등)
│   │   └── widgets/              # 재사용 커스텀 위젯 (차트·스피너·ROI 등)
│   └── widgets/                  # 라벨링·썸네일 위젯
│       ├── labeling_widget.py
│       ├── new_object_labeling_widget.py
│       └── result_thumbnail_browser.py
│
├── backend/                      # 백엔드 로직
│   ├── filters/                  # 품질·중복 필터링 엔진
│   │   ├── config.py             # FilterConfig (임계값 설정)
│   │   ├── engine.py             # SampleFilterEngine (keep/hold/drop 판정)
│   │   ├── global_index.py       # 전역 중복 해시 인덱스
│   │   ├── hash.py               # dHash64 퍼셉추얼 해싱
│   │   ├── quality.py            # 블러·노출 품질 평가
│   │   └── track_state.py        # 트랙별 상태 저장소
│   ├── labeling/                 # SAM3 원격 실행 오케스트레이션
│   │   ├── sam3_runner.py        # SSH 업로드·실행·결과 다운로드
│   │   └── artifacts.py          # 결과 아티팩트 파싱
│   ├── llm/                      # LLM 프롬프트 생성 모듈
│   │   ├── client.py             # HTTP 클라이언트 (로컬/원격 LLM API 호출)
│   │   ├── manager.py            # LLMServerManager (자동 시작/중지)
│   │   ├── prompting.py          # SAM 프롬프트 후보 빌더
│   │   ├── runtime.py            # Qwen2.5 모델 로드·추론·랭킹 (4-bit 양자화 지원)
│   │   └── server.py             # FastAPI 서버 (/rank-prompts 엔드포인트)
│   ├── pipelines/                # 후처리 파이프라인
│   │   └── preview_postprocess.py  # 미리보기 아이템 후처리·카테고리 요약
│   └── storage/                  # 원격 스토리지 동기화
│       └── remote_storage.py     # SSH 기반 파일 업로드/다운로드
│
├── core/                         # 공유 데이터 모델·경로 상수
│   ├── dataset.py                # FrameAnnotation, BoxAnnotation 데이터클래스
│   ├── models.py                 # PreviewThumbnail, WorkerOutput, ProgressEvent 등
│   ├── paths.py                  # 저장 경로 상수 및 디렉터리 초기화
│   └── tracking.py               # TrackObject (KEEP/HOLD/DROP 상태 머신)
│
├── config/                       # 설정 모듈
│   └── settings.py               # config.local.json 로더·원격 환경변수 주입
│
├── utils/                        # 환경변수 유틸리티
│   └── env.py                    # env_flag, env_int 헬퍼
│
├── vendor/                       # 서드파티 소스
│   └── sam_3/                    # SAM3 (Segment Anything Model 3) 소스
│
├── assets/                       # UI 리소스 (QSS·아이콘·.ui 파일)
│   └── styles/
│       ├── main_window.qss       # 다크 테마
│       └── main_window_light.qss # 라이트 테마 (선택적)
│
├── tests/                        # 단위 테스트
│   ├── test_llm_api_client.py
│   ├── test_llm_runtime_core.py
│   ├── test_preview_postprocess.py
│   ├── test_prompting.py
│   ├── test_remote_config.py
│   ├── test_sam3_preview_items.py
│   └── test_sam3_runtime_perf.py
│
└── runs/                         # 학습 실행 결과 저장소
```

---

## 주요 워커 (QThread 기반)

| 워커 클래스 | 역할 |
|-------------|------|
| `AutoLabelWorker` | SAM3 파이프라인 실행 (영상 업로드 → 원격 추론 → 결과 수신 → 필터링) |
| `DatasetExportWorker` | YOLO 형식 데이터셋 내보내기 (train/valid/test 분할, crop 저장) |
| `YoloTrainWorker` | YOLO / RT-DETR 학습 실행 (로컬 또는 원격 SSH) |
| `MultiDatasetMergeWorker` | 여러 데이터셋 병합 |
| `ReplayDatasetMergeWorker` | 리플레이 기반 데이터셋 병합 학습 |
| `ModelTestWorker` | 학습된 모델로 영상 추론·결과 시각화 |

---

## 주요 플로우

### 1단계: SAM3 자동 라벨링

```
영상 로드 → 클래스명·프롬프트 입력 → SAM3 실행
    → SSH로 영상·런타임 업로드
    → 원격 서버: SAM3 Text Tracker 프레임별 탐지·추적
    → tracks.csv + summary.json 로컬 다운로드
    → 품질 필터링 (블러·중복 제거)
    → BBox 오버레이 시각화 (keep=녹색 / hold=노랑 / drop=파랑)
    → 결과 브라우저에서 검토
    → YOLO 데이터셋 내보내기
```

### 2단계: 신규 객체 추가 라벨링

```
기존 세션 영상 로드 → 신규 클래스명 입력
    → SAM3로 신규 객체만 추적
    → 기존 라벨에 병합하여 내보내기
```

### 3단계: 모델 학습

```
데이터셋 YAML 선택 → 엔진 선택 (YOLO / RT-DETR)
    → 학습 모드 선택 (신규 학습 / 리트레인)
    → 하이퍼파라미터 설정 (epochs, lr, freeze 등)
    → 학습 실행 (진행 차트 실시간 표시)
    → 학습된 .pt 모델 저장
```

### LLM 프롬프트 랭킹

```
한국어 입력 (예: "트럭 앞 범퍼")
    → Qwen2.5-7B 추론 → 영어 후보 생성
        예: front bumper, bumper, front grille
    → 배치 스코어링 (cross-entropy loss 기반 재랭킹)
    → 확률 높은 순 정렬 → SAM3 프롬프트로 전달
```

---

## keep / hold / drop 기준

| 상태 | 색상 | 의미 |
|------|------|------|
| `keep` | 초록 | SAM3 직접 감지, confidence ≥ 임계값 |
| `hold` | 노랑 | tracker 예측 또는 중간 confidence |
| `drop` | 파랑 | 낮은 confidence, 블러, 중복 해시 탈락 |

---

## 환경 설정

### 1. 가상환경 생성

```powershell
conda env create -f environment_final.yml
conda activate PJ_310_LLM_SAM3
```

### 2. config.local.json 작성

`config.local.example.json`을 복사해 `config.local.json`을 만들고 아래 항목을 입력합니다.

```json
{
  "remote": {
    "ssh": {
      "host": "원격서버IP",
      "port": 22,
      "user": "사용자ID",
      "password": "비밀번호"
    },
    "python": {
      "llm":   "G:/conda/envs/PJ_310_LLM_SAM3/python.exe",
      "sam3":  "G:/conda/envs/PJ_310_LLM_SAM3/python.exe",
      "train": "G:/conda/envs/PJ_310_LLM_SAM3/python.exe"
    },
    "storage_base_root": "G:/KDT10_3_1team_KLIK/0_Program_",
    "models": {
      "llm":      "G:/models/Qwen2.5-7B-Instruct",
      "sam3_root": "G:/models/sam3"
    },
    "features": {
      "llm_enabled":   true,
      "sam3_enabled":  true,
      "train_enabled": true
    }
  }
}
```

---

## 실행 방법

### 데스크탑 앱 실행

```powershell
conda activate PJ_310_LLM_SAM3
python run.py
```

### LLM FastAPI 서버 단독 실행 (로컬 LLM 사용 시)

```powershell
conda activate PJ_310_LLM_SAM3
python run_llm_server.py
# 기본 주소: http://127.0.0.1:8008
# 엔드포인트: GET /health, GET /warmup, POST /rank-prompts
```

---

## 테스트

```powershell
# 문법 검사
python -m py_compile run.py run_llm_server.py backend\labeling\sam3_runner.py

# 단위 테스트 전체 실행
python -m unittest discover -s tests -v
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| UI | PyQt6 |
| 객체 탐지·추적 | SAM3 (Segment Anything Model 3) + ByteTrack |
| LLM | Qwen2.5-7B-Instruct (4-bit NF4 양자화, HuggingFace Transformers) |
| 학습 엔진 | Ultralytics (YOLOv8/v11, RT-DETR) |
| LLM 서버 | FastAPI + uvicorn |
| 원격 접속 | paramiko (SSH/SFTP) |
| 영상 처리 | OpenCV (cv2) |
| 데이터셋 형식 | YOLO (images/ + labels/ + data.yaml) |

---

## 저장 경로

| 경로 | 설명 |
|------|------|
| `assets/dataset_save_dir/` | 내보낸 YOLO 데이터셋 |
| `assets/dataset_save_dir/merged_dataset_save_dir/` | 병합된 데이터셋 |
| `assets/crop_save_dir/` | 객체 크롭 이미지 |
| `assets/models/YOLO_models/` | YOLO 학습 모델 (.pt) |
| `assets/models/RT-DETR_models/` | RT-DETR 학습 모델 (.pt) |
| `runs/train/` | 학습 실행 결과 |
| `runtime_preview_cache/` | 실행 중 미리보기 캐시 |

---

## 주의 사항

- `config.local.json`에는 개인 접속 정보가 포함되므로 절대 공유·커밋하지 마세요.
- SAM3 프롬프트에 따옴표를 포함하지 마세요. (`car roof` O, `'car roof'` X)
- 원격 서버 Python 경로가 `config.local.json`에 없으면 기본값 `G:/conda/envs/PJ_310_LLM_SAM3/python.exe`를 사용합니다.
- Windows 환경에서 PyTorch(libomp)와 MKL(libiomp5md) OpenMP 충돌 방지를 위해 `KMP_DUPLICATE_LIB_OK=TRUE`가 자동 설정됩니다.
