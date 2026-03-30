from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PyQt6.QtCore import QSettings, QThread, Qt
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QDialog, QFileDialog, QListWidgetItem, QMessageBox, QWidget

from app.studio.config import (
    PROVENANCE_JSON_BEGIN,
    PROVENANCE_JSON_END,
    TRAIN_DEFAULT_FREEZE,
    TRAIN_DEFAULT_LR0,
    TRAIN_REPLAY_RATIO_DEFAULT,
    TRAIN_RETRAIN_SEED_DEFAULT,
    TRAIN_STAGE1_EPOCHS_DEFAULT,
    TRAIN_STAGE2_EPOCHS_DEFAULT,
    TRAIN_STAGE2_LR_FACTOR_DEFAULT,
    TRAIN_STAGE_UNFREEZE_BACKBONE_LAST,
    TRAIN_STAGE_UNFREEZE_CHOICES,
    TRAIN_STAGE_UNFREEZE_NECK_ONLY,
)
from core.paths import (
    DATASET_SAVE_DIR,
    MERGED_DATASET_SAVE_DIR,
    TRAIN_RTDETR_MODELS_DIR,
    TRAIN_RUNS_DIR,
    TRAIN_YOLO_MODELS_DIR,
)
from core.models import PreviewThumbnail
from app.studio.runtime import TqdmFormatter, yaml
from app.studio.utils import (
    _build_unique_path,
    _dedupe_preserve_order,
    _load_yaml_dict,
    _metric_to_unit_interval,
    _parse_names_from_yaml_payload,
    _resolve_dataset_root_from_yaml_payload,
    _sync_ultralytics_datasets_dir,
    _try_autofix_data_yaml_path,
    _try_autofix_data_yaml_splits,
    build_dataset_folder_name,
    collect_original_components_from_yaml,
    extract_training_metrics_and_losses,
    read_provenance_from_readme,
    sanitize_class_name,
    write_provenance_readme,
)
from app.studio.workers import ModelTestWorker, ReplayDatasetMergeWorker, YoloTrainWorker
from app.ui.dialogs.training_yaml_picker import TrainingYamlPickerDialog
from app.ui.dialogs.yolo_params_dialog import YoloParamsDialog

_log = logging.getLogger(__name__)


class StudioTrainingFlowMixin:
    def _update_training_ui_state(self) -> None:
        """?숈뒿 愿???꾩젽???쒖꽦???쒖떆 ?곹깭瑜??꾩옱 ?숈뒿 ?ㅽ뻾 ?곹깭? ?숆린?뷀빀?덈떎."""
        running = bool(self.is_training)
        if self.btnTrainStart is not None:
            self.btnTrainStart.setEnabled((not running) and (not self.is_processing) and (not self.is_exporting) and (not self.is_model_testing))
        if self.btnTrainStop is not None:
            self.btnTrainStop.setEnabled(running and (self.training_worker is not None))
        if self.btnTrainPickDataYaml is not None:
            self.btnTrainPickDataYaml.setEnabled(not running)
        if self.btnTrainAddDataYamls is not None:
            self.btnTrainAddDataYamls.setEnabled(not running)
        if self.btnTrainRemoveDataYaml is not None:
            self.btnTrainRemoveDataYaml.setEnabled((not running) and bool(self.training_data_yaml_paths))
        if self.btnTrainClearDataYamls is not None:
            self.btnTrainClearDataYamls.setEnabled((not running) and bool(self.training_data_yaml_paths))
        if self.listTrainDataYamls is not None:
            self.listTrainDataYamls.setEnabled(not running)
        if self.comboTrainModel is not None:
            self.comboTrainModel.setEnabled(not running)
        if self.spinTrainEpochs is not None:
            self.spinTrainEpochs.setEnabled(not running)
        if self.spinTrainImgsz is not None:
            self.spinTrainImgsz.setEnabled(not running)
        if self.spinTrainBatch is not None:
            self.spinTrainBatch.setEnabled(not running)
        if self.checkTrainNew is not None:
            self.checkTrainNew.setEnabled(not running)
        if self.checkTrainRetrain is not None:
            self.checkTrainRetrain.setEnabled(not running)
        if self.comboTrainEngine is not None:
            self.comboTrainEngine.setEnabled(not running)
        # Feature 1 & 2: 새 위젯 활성화 상태
        if self.checkTrainLocal is not None:
            self.checkTrainLocal.setEnabled(not running)
        if self.btnTrainAdvancedParams is not None:
            self.btnTrainAdvancedParams.setEnabled(not running)
        self._update_training_advanced_ui()

    def _update_model_test_ui_state(self) -> None:
        """紐⑤뜽 ?뚯뒪???섏씠吏 ?꾩젽???쒖꽦???쒖떆 ?곹깭瑜??꾩옱 ?ㅽ뻾 ?곹깭? ?숆린?뷀빀?덈떎."""
        running = bool(self.is_model_testing)
        if self.btnModelTestStart is not None:
            self.btnModelTestStart.setEnabled((not running) and (not self.is_processing) and (not self.is_exporting) and (not self.is_training))
        if self.btnModelTestStop is not None:
            self.btnModelTestStop.setEnabled(running)
        if self.btnModelTestPickModel is not None:
            self.btnModelTestPickModel.setEnabled(not running)
        if self.btnModelTestPickVideo is not None:
            self.btnModelTestPickVideo.setEnabled(not running)
        if self.spinModelTestConf is not None:
            self.spinModelTestConf.setEnabled(not running)
        if self.spinModelTestIou is not None:
            self.spinModelTestIou.setEnabled(not running)
        if self.spinModelTestImgsz is not None:
            self.spinModelTestImgsz.setEnabled(not running)
        if self.labelModelTestStatus is not None:
            self.labelModelTestStatus.setText("status: running" if running else "status: idle")

    def _append_model_test_log(self, text: str) -> None:
        """紐⑤뜽 ?뚯뒪??濡쒓렇李쎌뿉 timestamp瑜?遺숈뿬 ??以?濡쒓렇瑜?異붽??⑸땲??"""
        if self.textModelTestLog is None:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        self.textModelTestLog.append(f"[{stamp}] {text}")
        sb = self.textModelTestLog.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    def _on_pick_model_test_model(self) -> None:
        """紐⑤뜽 ?뚯뒪?몄슜 .pt ?뚯씪???좏깮???대? ?곹깭? UI??諛섏쁺?⑸땲??"""
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "?뚯뒪??紐⑤뜽 ?좏깮",
            str(TRAIN_RUNS_DIR if TRAIN_RUNS_DIR.is_dir() else Path.home()),
            "PyTorch 紐⑤뜽 (*.pt);;紐⑤뱺 ?뚯씪 (*)",
        )
        if not selected:
            return
        path = Path(selected).resolve()
        if not path.is_file():
            QMessageBox.warning(self, "Model Test", f"Model file not found.\n{path}")
            return
        self.model_test_model_path = path
        if self.editModelTestModel is not None:
            self.editModelTestModel.setText(str(path))
        self._append_model_test_log(f"user action: test model selected ({path.name})")

    def _on_pick_model_test_video(self) -> None:
        """紐⑤뜽 ?뚯뒪?몄슜 ?곸긽 ?뚯씪???좏깮???대? ?곹깭? UI??諛섏쁺?⑸땲??"""
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "?뚯뒪???곸긽 ?좏깮",
            str(Path.home()),
            "?곸긽 ?뚯씪 (*.mp4 *.avi *.mov *.mkv);;紐⑤뱺 ?뚯씪 (*)",
        )
        if not selected:
            return
        path = Path(selected).resolve()
        if not path.is_file():
            QMessageBox.warning(self, "Model Test", f"Video file not found.\n{path}")
            return
        self.model_test_video_path = path
        if self.editModelTestVideo is not None:
            self.editModelTestVideo.setText(str(path))
        self._append_model_test_log(f"user action: test video selected ({path.name})")

    def on_start_model_test(self) -> None:
        """?숈뒿??紐⑤뜽濡??뚯뒪???곸긽??異붾줎?섎뒗 諛깃렇?쇱슫???뚯빱瑜??쒖옉?⑸땲??"""
        if self.is_model_testing:
            return
        if self.is_processing or self.is_exporting or self.is_training:
            QMessageBox.information(self, "Model Test", "Finish the current task first, then start model test.")
            return
        if self.model_test_model_path is None or (not self.model_test_model_path.is_file()):
            QMessageBox.warning(self, "Model Test", "Select a model (.pt) first.")
            return
        if self.model_test_video_path is None or (not self.model_test_video_path.is_file()):
            QMessageBox.warning(self, "Model Test", "Select a test video first.")
            return

        conf = float(self.spinModelTestConf.value()) if self.spinModelTestConf is not None else 0.25
        iou = float(self.spinModelTestIou.value()) if self.spinModelTestIou is not None else 0.45
        imgsz = int(self.spinModelTestImgsz.value()) if self.spinModelTestImgsz is not None else 960
        if self.progressModelTest is not None:
            self.progressModelTest.setValue(0)
        if self.labelModelTestFrame is not None:
            self.labelModelTestFrame.setText("frame: - / -")

        self.model_test_thread = QThread(self)
        self.model_test_worker = ModelTestWorker(
            model_path=self.model_test_model_path,
            video_path=self.model_test_video_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
        )
        self.model_test_worker.moveToThread(self.model_test_thread)
        self.model_test_thread.started.connect(self.model_test_worker.run)
        self.model_test_worker.log_message.connect(self._on_model_test_log_message)
        self.model_test_worker.frame_ready.connect(self._on_model_test_frame_ready)
        self.model_test_worker.finished.connect(self._on_model_test_finished)
        self.model_test_worker.failed.connect(self._on_model_test_failed)
        self.model_test_worker.finished.connect(self.model_test_thread.quit)
        self.model_test_worker.failed.connect(self.model_test_thread.quit)
        self.model_test_thread.finished.connect(self._on_model_test_thread_finished)
        self.model_test_thread.finished.connect(self.model_test_worker.deleteLater)
        self.model_test_thread.finished.connect(self.model_test_thread.deleteLater)

        self.is_model_testing = True
        self._update_model_test_ui_state()
        self._update_button_state()
        self._append_model_test_log("start model test")
        self.model_test_thread.start()

    def on_stop_model_test(self) -> None:
        """?ㅽ뻾 以묒씤 紐⑤뜽 ?뚯뒪??異붾줎??以묒? ?붿껌??蹂대깄?덈떎."""
        if not self.is_model_testing or self.model_test_worker is None:
            return
        self.model_test_worker.request_stop()
        self._append_model_test_log("stop requested")

    def _on_model_test_log_message(self, message: str) -> None:
        self._append_model_test_log(str(message))

    def _on_model_test_frame_ready(self, frame_obj: object, boxes_obj: object, frame_index: int, total_frames: int) -> None:
        if not isinstance(frame_obj, np.ndarray):
            return
        boxes = boxes_obj if isinstance(boxes_obj, list) else []
        draw_frame = self._draw_boxes(frame_obj, boxes)
        if self.labelModelTestPreview is not None:
            pixmap = self._frame_to_pixmap(
                draw_frame,
                max_width=max(320, self.labelModelTestPreview.width()),
                max_height=max(240, self.labelModelTestPreview.height()),
            )
            self.labelModelTestPreview.setPixmap(pixmap)
            self.labelModelTestPreview.setText("")
        if self.progressModelTest is not None and total_frames > 0:
            percent = int(round((max(0, int(frame_index)) / float(max(1, int(total_frames)))) * 100.0))
            self.progressModelTest.setValue(max(0, min(100, percent)))
        if self.labelModelTestFrame is not None:
            self.labelModelTestFrame.setText(f"frame: {max(0, int(frame_index))} / {max(1, int(total_frames))}")

    def _on_model_test_finished(self) -> None:
        if self.progressModelTest is not None:
            self.progressModelTest.setValue(100)
        self._append_model_test_log("finish model test")

    def _on_model_test_failed(self, message: str) -> None:
        self._append_model_test_log(f"error: model test failed ({message})")
        QMessageBox.critical(self, "紐⑤뜽 ?뚯뒪???ㅻ쪟", f"紐⑤뜽 ?뚯뒪??以??ㅻ쪟媛 諛쒖깮?덉뒿?덈떎.\n{message}")

    def _on_model_test_thread_finished(self) -> None:
        self.is_model_testing = False
        self.model_test_worker = None
        self.model_test_thread = None
        self._update_model_test_ui_state()
        self._update_button_state()

    def _on_training_mode_checkbox_toggled(self, _checked: bool) -> None:
        """?좉퇋/?ы븰??泥댄겕諛뺤뒪媛 ?숈떆???좏깮?섏? ?딅룄濡??숆린?뷀빀?덈떎."""
        if self._training_mode_syncing:
            return
        self._training_mode_syncing = True
        try:
            sender = self.sender()
            if sender is self.checkTrainNew and self.checkTrainNew is not None and self.checkTrainNew.isChecked():
                if self.checkTrainRetrain is not None:
                    self.checkTrainRetrain.setChecked(False)
            elif sender is self.checkTrainRetrain and self.checkTrainRetrain is not None and self.checkTrainRetrain.isChecked():
                if self.checkTrainNew is not None:
                    self.checkTrainNew.setChecked(False)

            new_checked = bool(self.checkTrainNew is not None and self.checkTrainNew.isChecked())
            retrain_checked = bool(self.checkTrainRetrain is not None and self.checkTrainRetrain.isChecked())
            if (not new_checked) and (not retrain_checked):
                if sender is self.checkTrainRetrain:
                    if self.checkTrainRetrain is not None:
                        self.checkTrainRetrain.setChecked(True)
                else:
                    if self.checkTrainNew is not None:
                        self.checkTrainNew.setChecked(True)
        finally:
            self._training_mode_syncing = False
        self._update_training_mode_ui()

    def _training_settings(self) -> QSettings:
        return QSettings("SL_TEAM", "AutoLabelStudio")

    def _load_training_ui_settings(self) -> None:
        if self.spinTrainLr0 is None:
            return

        settings = self._training_settings()

        self._training_settings_syncing = True
        try:
            if self.spinTrainFreeze is not None:
                self.spinTrainFreeze.setValue(
                    max(
                        0,
                        int(
                            settings.value(
                                "training/freeze",
                                TRAIN_DEFAULT_FREEZE,
                            )
                        ),
                    )
                )
            self.spinTrainLr0.setValue(
                max(
                    0.000001,
                    float(
                        settings.value(
                            "training/lr0",
                            TRAIN_DEFAULT_LR0,
                        )
                    ),
                )
            )
            if self.spinTrainReplayRatio is not None:
                self.spinTrainReplayRatio.setValue(
                    min(
                        0.30,
                        max(
                            0.05,
                            float(settings.value("training/replay_ratio_old", TRAIN_REPLAY_RATIO_DEFAULT)),
                        ),
                    )
                )
            if self.spinTrainStage1Epochs is not None:
                self.spinTrainStage1Epochs.setValue(
                    min(20, max(5, int(settings.value("training/stage1_epochs", TRAIN_STAGE1_EPOCHS_DEFAULT))))
                )
            if self.spinTrainStage2Epochs is not None:
                self.spinTrainStage2Epochs.setValue(
                    min(80, max(20, int(settings.value("training/stage2_epochs", TRAIN_STAGE2_EPOCHS_DEFAULT))))
                )
            if self.spinTrainStage2LrFactor is not None:
                self.spinTrainStage2LrFactor.setValue(
                    min(
                        1.0,
                        max(
                            0.05,
                            float(
                                settings.value(
                                    "training/stage2_lr_factor",
                                    TRAIN_STAGE2_LR_FACTOR_DEFAULT,
                                )
                            ),
                        ),
                    )
                )
            if self.comboTrainUnfreezeMode is not None:
                mode_value = str(settings.value("training/unfreeze_mode", TRAIN_STAGE_UNFREEZE_NECK_ONLY)).strip()
                target_index = self.comboTrainUnfreezeMode.findData(mode_value)
                if target_index < 0:
                    target_index = self.comboTrainUnfreezeMode.findData(TRAIN_STAGE_UNFREEZE_NECK_ONLY)
                self.comboTrainUnfreezeMode.setCurrentIndex(max(0, target_index))
            if self.spinTrainReplaySeed is not None:
                self.spinTrainReplaySeed.setValue(
                    max(0, int(settings.value("training/replay_seed", TRAIN_RETRAIN_SEED_DEFAULT)))
                )
        except Exception:
            if self.spinTrainFreeze is not None:
                self.spinTrainFreeze.setValue(TRAIN_DEFAULT_FREEZE)
            if self.spinTrainLr0 is not None:
                self.spinTrainLr0.setValue(TRAIN_DEFAULT_LR0)
            if self.spinTrainReplayRatio is not None:
                self.spinTrainReplayRatio.setValue(TRAIN_REPLAY_RATIO_DEFAULT)
            if self.spinTrainStage1Epochs is not None:
                self.spinTrainStage1Epochs.setValue(TRAIN_STAGE1_EPOCHS_DEFAULT)
            if self.spinTrainStage2Epochs is not None:
                self.spinTrainStage2Epochs.setValue(TRAIN_STAGE2_EPOCHS_DEFAULT)
            if self.spinTrainStage2LrFactor is not None:
                self.spinTrainStage2LrFactor.setValue(TRAIN_STAGE2_LR_FACTOR_DEFAULT)
            if self.comboTrainUnfreezeMode is not None:
                self.comboTrainUnfreezeMode.setCurrentIndex(0)
            if self.spinTrainReplaySeed is not None:
                self.spinTrainReplaySeed.setValue(TRAIN_RETRAIN_SEED_DEFAULT)
        finally:
            self._training_settings_syncing = False

        # Feature 1: 로컬 학습 체크박스 복원
        if self.checkTrainLocal is not None:
            raw_local = settings.value("training/use_local", False)
            is_local = bool(raw_local is True or str(raw_local).lower() == "true")
            self.checkTrainLocal.setChecked(is_local)
            if is_local:
                device_str, device_desc = self._detect_local_device()
                if self.labelTrainLocalDevice is not None:
                    self.labelTrainLocalDevice.setText(f"로컬 장치: {device_desc}")

        self._update_training_advanced_ui()

    def _persist_training_ui_settings(self, *_args: object) -> None:
        if self._training_settings_syncing:
            return
        settings = self._training_settings()
        if self.spinTrainFreeze is not None:
            settings.setValue("training/freeze", int(self.spinTrainFreeze.value()))
        if self.spinTrainLr0 is not None:
            settings.setValue("training/lr0", float(self.spinTrainLr0.value()))
        if self.spinTrainReplayRatio is not None:
            settings.setValue("training/replay_ratio_old", float(self.spinTrainReplayRatio.value()))
        if self.spinTrainStage1Epochs is not None:
            settings.setValue("training/stage1_epochs", int(self.spinTrainStage1Epochs.value()))
        if self.spinTrainStage2Epochs is not None:
            settings.setValue("training/stage2_epochs", int(self.spinTrainStage2Epochs.value()))
        if self.spinTrainStage2LrFactor is not None:
            settings.setValue("training/stage2_lr_factor", float(self.spinTrainStage2LrFactor.value()))
        if self.comboTrainUnfreezeMode is not None:
            settings.setValue(
                "training/unfreeze_mode",
                str(self.comboTrainUnfreezeMode.currentData() or TRAIN_STAGE_UNFREEZE_NECK_ONLY),
            )
        if self.spinTrainReplaySeed is not None:
            settings.setValue("training/replay_seed", int(self.spinTrainReplaySeed.value()))
        # Feature 1: 로컬 학습 체크박스 저장
        if self.checkTrainLocal is not None:
            settings.setValue("training/use_local", bool(self.checkTrainLocal.isChecked()))

    def _current_training_advanced_settings(self) -> dict[str, Any]:
        freeze = int(self.spinTrainFreeze.value()) if self.spinTrainFreeze is not None else TRAIN_DEFAULT_FREEZE
        lr0 = float(self.spinTrainLr0.value()) if self.spinTrainLr0 is not None else TRAIN_DEFAULT_LR0
        return {
            "freeze": int(max(0, freeze)),
            "lr0": float(max(0.000001, lr0)),
            "replay_ratio_old": float(
                min(
                    0.30,
                    max(
                        0.05,
                        self.spinTrainReplayRatio.value() if self.spinTrainReplayRatio is not None else TRAIN_REPLAY_RATIO_DEFAULT,
                    ),
                )
            ),
            "stage1_epochs": int(
                min(20, max(5, self.spinTrainStage1Epochs.value() if self.spinTrainStage1Epochs is not None else TRAIN_STAGE1_EPOCHS_DEFAULT))
            ),
            "stage2_epochs": int(
                min(80, max(20, self.spinTrainStage2Epochs.value() if self.spinTrainStage2Epochs is not None else TRAIN_STAGE2_EPOCHS_DEFAULT))
            ),
            "stage2_lr_factor": float(
                min(
                    1.0,
                    max(
                        0.05,
                        self.spinTrainStage2LrFactor.value() if self.spinTrainStage2LrFactor is not None else TRAIN_STAGE2_LR_FACTOR_DEFAULT,
                    ),
                )
            ),
            "unfreeze_mode": str(
                self.comboTrainUnfreezeMode.currentData() if self.comboTrainUnfreezeMode is not None else TRAIN_STAGE_UNFREEZE_NECK_ONLY
            ).strip()
            or TRAIN_STAGE_UNFREEZE_NECK_ONLY,
            "replay_seed": int(
                max(0, self.spinTrainReplaySeed.value() if self.spinTrainReplaySeed is not None else TRAIN_RETRAIN_SEED_DEFAULT)
            ),
        }

    # ── Feature 1: 로컬/서버 학습 전환 ────────────────────────────────────────

    def _detect_local_device(self) -> tuple[str, str]:
        ## =====================================
        ## 함수 기능 : 로컬 CUDA GPU 가용 여부를 탐지하고 장치 문자열과 설명을 반환합니다
        ## 매개 변수 : 없음
        ## 반환 결과 : tuple[str, str] -> (device_str, device_description)
        ##             device_str: "0" (GPU) 또는 "cpu"
        ## =====================================
        try:
            from app.studio.runtime import torch
            if torch is not None and torch.cuda.is_available():
                count = torch.cuda.device_count()
                name = torch.cuda.get_device_name(0) if count > 0 else "Unknown GPU"
                _log.info("로컬 CUDA 장치 감지: %s (device count=%d)", name, count)
                return "0", f"CUDA ({name})"
        except Exception as exc:
            _log.warning("CUDA 장치 탐지 실패: %s", exc)
        _log.info("로컬 장치: CPU (GPU 미감지)")
        return "cpu", "CPU (GPU 미감지)"

    def _on_train_local_toggled(self, checked: bool) -> None:
        ## =====================================
        ## 함수 기능 : 로컬 PC 학습 체크박스 토글 시 장치 레이블 업데이트 및 설정 저장
        ## 매개 변수 : checked(bool) -> 체크박스 선택 여부
        ## 반환 결과 : None
        ## =====================================
        self._persist_training_ui_settings()
        if not checked:
            if self.labelTrainLocalDevice is not None:
                self.labelTrainLocalDevice.setText("")
            _log.info("학습 경로: 원격 서버 모드")
            return
        device_str, device_desc = self._detect_local_device()
        if self.labelTrainLocalDevice is not None:
            self.labelTrainLocalDevice.setText(f"로컬 장치: {device_desc}")
        _log.info("학습 경로: 로컬 PC 모드 — 장치=%s (%s)", device_str, device_desc)

    # ── Feature 2: YOLO 고급 파라미터 다이얼로그 ─────────────────────────────

    def _on_open_yolo_params_dialog(self) -> None:
        ## =====================================
        ## 함수 기능 : YOLO 고급 파라미터 설정 다이얼로그를 열고 적용 시 파라미터를 저장합니다
        ## 매개 변수 : 없음
        ## 반환 결과 : None
        ## =====================================
        dialog = YoloParamsDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._yolo_extra_params = dialog.current_params()
            _log.info(
                "YOLO 고급 파라미터 적용: lr0=%(lr0)s, momentum=%(momentum)s, "
                "mosaic=%(mosaic)s, box=%(box)s, cls=%(cls)s, dfl=%(dfl)s",
                self._yolo_extra_params,
            )
            self._append_train_log(
                f"고급 파라미터 적용됨 — lr0={self._yolo_extra_params.get('lr0', 'default')}, "
                f"mosaic={self._yolo_extra_params.get('mosaic', 'default')}, "
                f"box={self._yolo_extra_params.get('box', 'default')}"
            )

    def _update_training_advanced_ui(self) -> None:
        running = bool(self.is_training)
        retrain_checked = self._is_retrain_mode_checked()
        if self.frameTrainAdvancedOptions is not None:
            self.frameTrainAdvancedOptions.setEnabled(not running)
        if self.frameTrainFreezeOption is not None:
            self.frameTrainFreezeOption.setVisible(False)
            self.frameTrainFreezeOption.setEnabled(False)
        if self.spinTrainFreeze is not None:
            self.spinTrainFreeze.setEnabled(False)
        if self.spinTrainLr0 is not None:
            self.spinTrainLr0.setEnabled(not running)
        if self.frameTrainRetrainRecipeOptions is not None:
            self.frameTrainRetrainRecipeOptions.setVisible(retrain_checked)
            self.frameTrainRetrainRecipeOptions.setEnabled((not running) and retrain_checked)
        if self.spinTrainReplayRatio is not None:
            self.spinTrainReplayRatio.setEnabled((not running) and retrain_checked)
        if self.spinTrainStage1Epochs is not None:
            self.spinTrainStage1Epochs.setEnabled((not running) and retrain_checked)
        if self.spinTrainStage2Epochs is not None:
            self.spinTrainStage2Epochs.setEnabled((not running) and retrain_checked)
        if self.spinTrainStage2LrFactor is not None:
            self.spinTrainStage2LrFactor.setEnabled((not running) and retrain_checked)
        if self.comboTrainUnfreezeMode is not None:
            self.comboTrainUnfreezeMode.setEnabled((not running) and retrain_checked)
        if self.spinTrainReplaySeed is not None:
            self.spinTrainReplaySeed.setEnabled((not running) and retrain_checked)
        if self.spinTrainEpochs is not None:
            self.spinTrainEpochs.setEnabled((not running) and (not retrain_checked))

    def _format_training_epoch_progress_line(
        self,
        epoch: int,
        total_epochs: int,
        batch_now: int,
        batch_total: int,
    ) -> str:
        """?먰룷??吏꾪뻾 ?곹솴??怨좎젙 ??吏꾪뻾諛???以?臾몄옄?대줈 援ъ꽦?⑸땲??"""
        ep = max(1, int(epoch))
        ep_total = max(1, int(total_epochs))
        bn = max(1, int(batch_now))
        bt = max(1, int(batch_total))
        if TqdmFormatter is not None:
            try:
                meter = TqdmFormatter.format_meter(
                    bn,
                    bt,
                    elapsed=0.0,
                    ncols=52,
                    bar_format="{percentage:6.2f}%|{bar:30}|",
                )
                return f"[{ep}/{ep_total}] {str(meter).strip()}"
            except Exception:
                pass
        ratio = max(0.0, min(1.0, float(bn) / float(bt)))
        bar_width = 30
        filled = int(round(ratio * float(bar_width)))
        filled = max(0, min(bar_width, filled))
        bar = ("#" * filled) + ("-" * (bar_width - filled))
        percent = ratio * 100.0
        return f"[{ep}/{ep_total}] [{bar}] {percent:6.2f}%"

    def _update_training_epoch_progress_log_line(
        self,
        epoch: int,
        total_epochs: int,
        batch_now: int,
        batch_total: int,
    ) -> None:
        """?숈뒿 濡쒓렇李쎌뿉???먰룷??吏꾪뻾 ??以꾩쓣 ?쒖옄由??낅뜲?댄듃?⑸땲??"""
        if self.textTrainLog is None:
            return
        key = (int(epoch), int(total_epochs), int(batch_now), int(batch_total))
        if key == self._train_progress_last_key:
            return
        self._train_progress_last_key = key
        line = self._format_training_epoch_progress_line(epoch, total_epochs, batch_now, batch_total)

        cursor = self.textTrainLog.textCursor()
        doc_end = max(0, int(self.textTrainLog.document().characterCount()) - 1)
        if self._train_progress_anchor_pos is None or not self._train_progress_line_text:
            cursor.movePosition(QTextCursor.MoveOperation.End)
            if self.textTrainLog.document().characterCount() > 1:
                cursor.insertText("\n")
            self._train_progress_anchor_pos = cursor.position()
            cursor.insertText(line)
        else:
            start = max(0, min(int(self._train_progress_anchor_pos), doc_end))
            end = max(start, min(start + len(self._train_progress_line_text), doc_end))
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertText(line)
            self._train_progress_anchor_pos = start
        self._train_progress_line_text = line
        self.textTrainLog.setTextCursor(cursor)
        sb = self.textTrainLog.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    def _finalize_training_epoch_progress_log_line(self) -> None:
        """吏꾪뻾 ??以꾩쓣 ?뺤젙?섍퀬 ?ㅼ쓬 ?쇰컲 濡쒓렇媛 ?댁뼱吏????덇쾶 以꾨컮轅?泥섎━?⑸땲??"""
        if self.textTrainLog is None:
            self._train_progress_anchor_pos = None
            self._train_progress_line_text = ""
            self._train_progress_last_key = None
            return
        if self._train_progress_anchor_pos is None or (not self._train_progress_line_text):
            self._train_progress_anchor_pos = None
            self._train_progress_line_text = ""
            self._train_progress_last_key = None
            return
        doc_end = max(0, int(self.textTrainLog.document().characterCount()) - 1)
        start = max(0, min(int(self._train_progress_anchor_pos), doc_end))
        end = max(start, min(start + len(self._train_progress_line_text), doc_end))
        cursor = self.textTrainLog.textCursor()
        cursor.setPosition(end)
        cursor.insertText("\n")
        self.textTrainLog.setTextCursor(cursor)
        self._train_progress_anchor_pos = None
        self._train_progress_line_text = ""
        self._train_progress_last_key = None

    def _append_train_log(self, text: str) -> None:
        """?숈뒿 濡쒓렇李쎌뿉 timestamp瑜?遺숈뿬 ??以?濡쒓렇瑜?異붽??⑸땲??"""
        if self.textTrainLog is None:
            return
        self._finalize_training_epoch_progress_log_line()
        stamp = datetime.now().strftime("%H:%M:%S")
        self.textTrainLog.append(f"[{stamp}] {text}")
        sb = self.textTrainLog.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    def _set_training_progress_summary(self, percent: int, epoch_now: int, epoch_total: int) -> None:
        """?숈뒿 吏꾪뻾瑜?諛붿? ?곗륫 EPOCH ?쇰꺼???숈씪??湲곗??쇰줈 媛깆떊?⑸땲??"""
        pct = max(0, min(100, int(percent)))
        total = max(1, int(epoch_total))
        current = max(0, min(int(epoch_now), total))
        if self.progressTrainTotal is not None:
            self.progressTrainTotal.setValue(pct)
        if self.labelTrainMetricNow is not None:
            self.labelTrainMetricNow.setText(f"progress {pct}%, {current} / {total}")

    def _set_training_dataset_path(self, data_yaml_path: Path) -> None:
        """?숈뒿 ?곗씠?곗뀑(data.yaml) 寃쎈줈瑜??대? ?곹깭? UI??諛섏쁺?⑸땲??"""
        self.training_data_yaml_path = Path(data_yaml_path)
        self._sync_training_dataset_display()

    def _is_retrain_mode_checked(self) -> bool:
        return bool(self.checkTrainRetrain is not None and self.checkTrainRetrain.isChecked())

    def _sync_training_dataset_display(self) -> None:
        """?⑥씪/?ㅼ쨷 YAML ?좏깮 ?곹깭??留욎떠 ?곗씠?곗뀑 ?쒖떆 UI瑜?媛깆떊?⑸땲??"""
        retrain_checked = self._is_retrain_mode_checked()
        if self.editTrainDataYaml is not None:
            self.editTrainDataYaml.setPlaceholderText("new data.yaml path" if retrain_checked else "data.yaml path")
            self.editTrainDataYaml.setText(str(self.training_data_yaml_path) if self.training_data_yaml_path is not None else "")
        if self.labelTrainDataset is not None:
            if retrain_checked:
                if self.training_data_yaml_path is not None:
                    self.labelTrainDataset.setText(f"new dataset: {self.training_data_yaml_path}")
                else:
                    self.labelTrainDataset.setText("new dataset: not selected")
            else:
                if self.training_data_yaml_path is None:
                    self.labelTrainDataset.setText("dataset: not selected")
                else:
                    self.labelTrainDataset.setText(f"dataset: {self.training_data_yaml_path}")

    def _refresh_training_yaml_list_widget(self) -> None:
        count = len(self.training_data_yaml_paths)
        if self.labelTrainDataYamlCount is not None:
            self.labelTrainDataYamlCount.setText(f"selected {count}")
        if self.listTrainDataYamls is not None:
            self.listTrainDataYamls.clear()
            for idx, path in enumerate(self.training_data_yaml_paths, start=1):
                path_text = str(path)
                item = QListWidgetItem(f"{idx}. {path_text}")
                item.setToolTip(path_text)
                item.setData(Qt.ItemDataRole.UserRole, path_text)
                self.listTrainDataYamls.addItem(item)
        self._sync_training_dataset_display()
        self._update_training_ui_state()

    def _add_training_data_yaml_paths(self, paths: Sequence[Path]) -> int:
        """?ㅼ쨷 ?ы븰??YAML 紐⑸줉??以묐났 ?놁씠 ?뚯씪 寃쎈줈瑜?異붽??⑸땲??"""
        added = 0
        seen = {str(path.resolve()).casefold() for path in self.training_data_yaml_paths}
        for raw in paths:
            path = Path(raw).resolve()
            key = str(path).casefold()
            if key in seen:
                continue
            if not path.is_file():
                continue
            self.training_data_yaml_paths.append(path)
            seen.add(key)
            added += 1
        if added > 0:
            self._refresh_training_yaml_list_widget()
        return added

    def _is_valid_training_data_yaml(self, path: Path) -> bool:
        candidate = Path(path).resolve()
        if not (candidate.is_file() and candidate.name.casefold() == "data.yaml"):
            return False
        allowed_roots = [
            DATASET_SAVE_DIR.resolve(),
            MERGED_DATASET_SAVE_DIR.resolve(),
        ]
        for root in allowed_roots:
            try:
                candidate.relative_to(root)
                return True
            except Exception:
                continue
        return False

    def _is_valid_retrain_merged_yaml(self, path: Path) -> bool:
        candidate = Path(path).resolve()
        if not (
            candidate.is_file()
            and candidate.name.casefold() == "data.yaml"
            and candidate.parent.name.casefold().startswith("merged_")
        ):
            return False
        allowed_roots = [
            MERGED_DATASET_SAVE_DIR.resolve(),
            DATASET_SAVE_DIR.resolve(),
        ]
        for root in allowed_roots:
            try:
                candidate.relative_to(root)
                return True
            except Exception:
                continue
        return False

    def _collect_training_yaml_search_roots(self) -> list[Path]:
        roots: list[Path] = [
            MERGED_DATASET_SAVE_DIR,
            DATASET_SAVE_DIR,
        ]
        unique: list[Path] = []
        seen: set[str] = set()
        for raw in roots:
            path = Path(raw).resolve()
            key = str(path).casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique

    def _open_training_yaml_picker_dialog(
        self,
        *,
        multi_select: bool,
        title: str,
        roots: Sequence[Path] | None = None,
        path_filter: Callable[[Path], bool] | None = None,
    ) -> list[Path]:
        dialog = TrainingYamlPickerDialog(
            roots=list(roots) if roots is not None else self._collect_training_yaml_search_roots(),
            title=title,
            multi_select=multi_select,
            path_filter=path_filter,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return []
        return dialog.selected_paths()

    def _on_pick_training_data_yaml(self) -> None:
        """?숈뒿??data.yaml ?뚯씪???좏깮??寃쎈줈瑜??ㅼ젙?⑸땲??"""
        selected_paths = self._open_training_yaml_picker_dialog(
            multi_select=False,
            title="Select retrain new data.yaml" if self._is_retrain_mode_checked() else "Select training data.yaml",
            roots=self._collect_training_yaml_search_roots(),
            path_filter=self._is_valid_training_data_yaml,
        )
        if not selected_paths:
            return
        path = Path(selected_paths[0]).resolve()
        if not path.is_file():
            QMessageBox.warning(self, "Training Dataset", f"File not found.\n{path}")
            return
        self._set_training_dataset_path(path)
        if self._is_retrain_mode_checked():
            self.training_data_yaml_paths = []
            self._refresh_training_yaml_list_widget()
            self._append_train_log(f"user action: retrain new data selected ({path})")
        else:
            self._append_train_log(f"user action: data selected ({path})")

    def _on_pick_training_data_yamls(self) -> None:
        """?ы븰?듭슜 ?ㅼ쨷 YAML ?뚯씪???좏깮??紐⑸줉??異붽??⑸땲??"""
        self._ensure_retrain_aliases_for_saved_datasets()
        selected_paths = self._open_training_yaml_picker_dialog(
            multi_select=True,
            title="Select retrain merged data.yaml",
            roots=self._collect_training_yaml_search_roots(),
            path_filter=self._is_valid_retrain_merged_yaml,
        )
        if not selected_paths:
            return
        valid_paths: list[Path] = []
        invalid_paths: list[Path] = []
        for path in selected_paths:
            if self._is_valid_retrain_merged_yaml(path):
                valid_paths.append(path)
            else:
                invalid_paths.append(path)
        if invalid_paths:
            invalid_text = "\n".join(str(path) for path in invalid_paths[:8])
            QMessageBox.warning(
                self,
                "Retrain Dataset",
                "Retrain only supports merged_*/data.yaml.\n\n"
                f"{invalid_text}",
            )
            return
        added = self._add_training_data_yaml_paths(valid_paths)
        if added <= 0:
            self._append_train_log("user action: retrain YAML add skipped (all duplicates)")
        else:
            self._append_train_log(f"user action: retrain YAML added ({added})")

    def _on_remove_selected_training_data_yamls(self) -> None:
        if self.listTrainDataYamls is None:
            return
        selected_items = self.listTrainDataYamls.selectedItems()
        if not selected_items:
            return
        remove_keys = {str(item.data(Qt.ItemDataRole.UserRole) or "").casefold() for item in selected_items}
        if not remove_keys:
            return
        self.training_data_yaml_paths = [
            path for path in self.training_data_yaml_paths if str(path.resolve()).casefold() not in remove_keys
        ]
        self._refresh_training_yaml_list_widget()
        self._append_train_log(f"user action: retrain YAML removed ({len(remove_keys)}媛?")

    def _on_clear_training_data_yamls(self) -> None:
        if not self.training_data_yaml_paths:
            return
        count = len(self.training_data_yaml_paths)
        self.training_data_yaml_paths = []
        self._refresh_training_yaml_list_widget()
        self._append_train_log(f"user action: retrain YAML cleared ({count}媛?")

    def _update_training_mode_ui(self) -> None:
        """?ы븰??泥댄겕 ?곹깭???곕씪 ?ㅼ쨷 YAML UI ?몄텧/?띿뒪?몃? 媛깆떊?⑸땲??"""
        retrain_checked = self._is_retrain_mode_checked()
        if self.training_data_yaml_paths:
            self.training_data_yaml_paths = []
            self._refresh_training_yaml_list_widget()
        if self.frameTrainRetrainYamlPanel is not None:
            self.frameTrainRetrainYamlPanel.setVisible(False)
        show_single_dataset_ui = True
        if self.labelTrainDataset is not None:
            self.labelTrainDataset.setVisible(show_single_dataset_ui)
        if self.editTrainDataYaml is not None:
            self.editTrainDataYaml.setVisible(show_single_dataset_ui)
        if self.btnTrainPickDataYaml is not None:
            self.btnTrainPickDataYaml.setVisible(show_single_dataset_ui)
            self.btnTrainPickDataYaml.setText("Select New Dataset" if retrain_checked else "Select Dataset")
        self._sync_training_dataset_display()
        self._refresh_training_model_combo()
        self._update_training_advanced_ui()
        self._update_training_ui_state()

    def _iter_best_model_candidates(self, root: Path) -> Sequence[Path]:
        """best.pt ?꾨낫瑜?鍮좊Ⅴ寃??섏쭛?⑸땲??"""
        if not root.is_dir():
            return []
        try:
            root_resolved = root.resolve()
        except Exception:
            root_resolved = root

        candidates: list[Path] = []
        if root_resolved == TRAIN_RUNS_DIR.resolve():
            # runs/train? ?쇰컲?곸쑝濡?<run>/weights/best.pt 援ъ“?대?濡??ш? rglob ????쒗븳???⑦꽩???ъ슜?⑸땲??
            candidates.extend(root.glob("*/weights/best.pt"))
            candidates.extend(root.glob("best.pt"))
            return candidates
        return list(root.rglob("best.pt"))

    def _is_supported_new_train_model_path(self, engine_key: str, model_path: Path) -> bool:
        engine_name = str(engine_key).strip().lower()
        if engine_name != "yolo":
            return True
        stem = str(model_path.stem).strip().lower()
        return not stem.endswith(("-seg", "-pose", "-obb", "-cls"))

    def _collect_local_training_models(self, engine_key: str) -> list[Path]:
        """?숈뒿 紐⑤뱶/?붿쭊??留욌뒗 濡쒖뺄 紐⑤뜽(.pt) 紐⑸줉???섏쭛?⑸땲??"""
        retrain_checked = self._is_retrain_mode_checked()
        collected: list[Path] = []
        seen: set[str] = set()

        if retrain_checked:
            roots = [
                TRAIN_RUNS_DIR,
                TRAIN_YOLO_MODELS_DIR,
                TRAIN_RTDETR_MODELS_DIR,
            ]
            for root in roots:
                if not root.is_dir():
                    continue
                for path in self._iter_best_model_candidates(root):
                    try:
                        resolved = path.resolve()
                    except Exception:
                        resolved = path
                    key = str(resolved).casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    collected.append(resolved)
        else:
            model_dir = TRAIN_RTDETR_MODELS_DIR if engine_key == "rtdetr" else TRAIN_YOLO_MODELS_DIR
            try:
                model_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            if model_dir.is_dir():
                for path in model_dir.glob("*.pt"):
                    if not self._is_supported_new_train_model_path(engine_key, path):
                        continue
                    try:
                        resolved = path.resolve()
                    except Exception:
                        resolved = path
                    key = str(resolved).casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    collected.append(resolved)
        collected.sort(key=lambda item: (item.name.casefold(), str(item).casefold()))
        return collected

    def _collect_training_model_sources(self, engine_key: str) -> list[tuple[str, str]]:
        retrain_checked = self._is_retrain_mode_checked()
        collected: list[tuple[str, str]] = []
        seen: set[str] = set()

        for path in self._collect_local_training_models(engine_key):
            source = str(path)
            key = source.casefold()
            if key in seen:
                continue
            display = path.name
            if path.parent.name.casefold() == "weights" and path.parent.parent is not None:
                display = f"{path.name} ({path.parent.parent.name}) [local]"
            elif retrain_checked:
                display = f"{path.name} ({path.parent.name}) [local]"
            else:
                display = f"{path.name} [local]"
            seen.add(key)
            collected.append((display, source))
        return collected

    def _refresh_training_model_combo(self) -> None:
        """?숈뒿 ?붿쭊 ?좏깮??留욎떠 紐⑤뜽 肄ㅻ낫瑜?媛깆떊?섍퀬 ?좏깮 紐⑤뜽??training_model_source濡??숆린?뷀빀?덈떎."""
        if self.comboTrainModel is None:
            return
        engine_key = str(self.comboTrainEngine.currentData() if self.comboTrainEngine is not None else "yolo").strip().lower()
        if engine_key not in {"yolo", "rtdetr"}:
            engine_key = "yolo"

        previous_source = str(self.training_model_source or "").strip()
        candidates = self._collect_training_model_sources(engine_key)

        self.comboTrainModel.blockSignals(True)
        self.comboTrainModel.clear()
        retrain_checked = self._is_retrain_mode_checked()
        seen_sources: set[str] = set()
        seen_model_names: set[str] = set()
        for display, source_text in candidates:
            source_name = Path(source_text).name
            self.comboTrainModel.addItem(display, source_text)
            seen_sources.add(source_text.casefold())
            seen_model_names.add(source_name.casefold())

        if (not retrain_checked) and engine_key == "rtdetr":
            # Team_code 湲곗? RT-DETR 湲곕낯 ?꾨━?? 濡쒖뺄 ?뚯씪???놁뼱??諛붾줈 ?숈뒿 媛??
            preset_models = [
                ("RT-DETR-L (preset)", "rtdetr-l.pt"),
                ("RT-DETR-X (preset)", "rtdetr-x.pt"),
            ]
            for label, source in preset_models:
                if Path(source).name.casefold() in seen_model_names:
                    continue
                self.comboTrainModel.addItem(label, source)
                seen_sources.add(source.casefold())
                seen_model_names.add(Path(source).name.casefold())
        if (not retrain_checked) and engine_key == "yolo":
            # YOLO 사전학습 모델 preset — 로칼 파일 없을 때 Ultralytics가 자동 다운로드
            preset_models = [
                ("YOLO11n (preset)", "yolo11n.pt"),
                ("YOLO11s (preset)", "yolo11s.pt"),
                ("YOLO11m (preset)", "yolo11m.pt"),
                ("YOLO11l (preset)", "yolo11l.pt"),
                ("YOLO11x (preset)", "yolo11x.pt"),
                ("YOLOv8n (preset)", "yolov8n.pt"),
                ("YOLOv8s (preset)", "yolov8s.pt"),
                ("YOLOv8m (preset)", "yolov8m.pt"),
                ("YOLOv8l (preset)", "yolov8l.pt"),
                ("YOLOv8x (preset)", "yolov8x.pt"),
            ]
            for label, source in preset_models:
                if Path(source).name.casefold() in seen_model_names:
                    continue
                self.comboTrainModel.addItem(label, source)
                seen_sources.add(source.casefold())
                seen_model_names.add(Path(source).name.casefold())

        if self.comboTrainModel.count() <= 0:
            if retrain_checked:
                empty_label = "No retrain best.pt found"
            else:
                empty_label = "No RT-DETR model found" if engine_key == "rtdetr" else "No YOLO model found"
            self.comboTrainModel.addItem(empty_label, "")

        target_index = -1
        if previous_source:
            target_index = self.comboTrainModel.findData(previous_source)
            if target_index < 0:
                prev_name = Path(previous_source).name.casefold()
                for idx in range(self.comboTrainModel.count()):
                    data_value = str(self.comboTrainModel.itemData(idx) or "")
                    if Path(data_value).name.casefold() == prev_name:
                        target_index = idx
                        break
        if target_index < 0:
            target_index = 0
        self.comboTrainModel.setCurrentIndex(target_index)
        self.comboTrainModel.blockSignals(False)

        selected = str(self.comboTrainModel.currentData() or "").strip()
        self.training_model_source = selected

    def _training_loss_chart_titles(self) -> tuple[str, str, str]:
        engine_key = str(self.training_engine_name or "yolo").strip().lower()
        if self.comboTrainEngine is not None:
            combo_key = str(self.comboTrainEngine.currentData() or engine_key).strip().lower()
            if combo_key in {"yolo", "rtdetr"}:
                engine_key = combo_key
        if engine_key == "rtdetr":
            return ("giou_loss", "cls_loss", "l1_loss")
        return ("box_loss", "cls_loss", "dfl_loss")

    def _update_training_metric_chart_titles(self) -> None:
        t1, t2, t3 = self._training_loss_chart_titles()
        if self.trainingBoxLossChart is not None:
            self.trainingBoxLossChart.set_title(t1)
        if self.trainingClsLossChart is not None:
            self.trainingClsLossChart.set_title(t2)
        if self.trainingDflLossChart is not None:
            self.trainingDflLossChart.set_title(t3)

    def _on_training_engine_changed(self) -> None:
        if self.comboTrainEngine is not None:
            engine_key = str(self.comboTrainEngine.currentData() or "yolo").strip().lower()
            if engine_key == "rtdetr":
                if self.spinTrainBatch is not None and int(self.spinTrainBatch.value()) <= 0:
                    self.spinTrainBatch.setValue(4)
            self.training_engine_name = engine_key if engine_key in {"yolo", "rtdetr"} else "yolo"
        self._update_training_metric_chart_titles()
        self._update_training_advanced_ui()
        self._refresh_training_model_combo()

    def _on_training_model_changed(self) -> None:
        if self.comboTrainModel is None:
            return
        selected = str(self.comboTrainModel.currentData() or "").strip()
        self.training_model_source = selected

    def _collect_retrain_yaml_paths_for_training(self) -> list[Path]:
        """?ы븰??紐⑤뱶?먯꽌 ?ъ슜??YAML 紐⑸줉??以묐났 ?놁씠 諛섑솚?⑸땲??"""
        if self.training_data_yaml_path is None:
            return []
        return [Path(self.training_data_yaml_path).resolve()]

    def _dataset_base_name_from_yaml_path(self, yaml_path: Path) -> str:
        """data.yaml 寃쎈줈?먯꽌 寃곌낵 ?대뜑 湲곕낯紐낆쓣 異붿텧?⑸땲??"""
        path = Path(yaml_path).resolve()
        parent_name = path.parent.name.strip()
        stem = path.stem.strip()
        if stem.casefold() == "data" and parent_name:
            return self._to_english_slug(parent_name, fallback="dataset")
        source = stem if stem else parent_name
        return self._to_english_slug(source, fallback="dataset")

    def _create_retrain_merged_alias_for_yaml(self, source_yaml_path: Path) -> Path | None:
        """?좉퇋?숈뒿 ?곗씠?곗뀑???ы븰???좏깮??merged_*/data.yaml?쇰줈 ?섑븨 ?앹꽦?⑸땲??"""
        source_yaml = Path(source_yaml_path).resolve()
        if not source_yaml.is_file():
            return None
        if self._is_valid_retrain_merged_yaml(source_yaml):
            return source_yaml
        if yaml is None:
            return None

        source_payload = _load_yaml_dict(source_yaml)
        if not source_payload:
            return None

        source_root = _resolve_dataset_root_from_yaml_payload(source_yaml, source_payload)
        train_entry = source_payload.get("train")
        valid_entry = source_payload.get("val", source_payload.get("valid"))
        if train_entry is None or valid_entry is None:
            return None

        class_names = _parse_names_from_yaml_payload(source_payload)
        if not class_names:
            class_names = self._read_class_names_from_data_yaml(source_yaml)
        class_names = _dedupe_preserve_order(class_names)
        if not class_names:
            class_names = ["class"]

        MERGED_DATASET_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        source_test_entry = source_payload.get("test")
        for existing_yaml in sorted(
            MERGED_DATASET_SAVE_DIR.glob("merged_*/data.yaml"),
            key=lambda item: item.stat().st_mtime if item.exists() else 0.0,
            reverse=True,
        ):
            existing_payload = _load_yaml_dict(existing_yaml)
            if not existing_payload:
                continue
            if _resolve_dataset_root_from_yaml_payload(existing_yaml, existing_payload) != source_root:
                continue
            if existing_payload.get("train") != train_entry:
                continue
            if existing_payload.get("val", existing_payload.get("valid")) != valid_entry:
                continue
            if existing_payload.get("test") != source_test_entry:
                continue
            existing_names = _dedupe_preserve_order(_parse_names_from_yaml_payload(existing_payload))
            if existing_names != class_names:
                continue
            return existing_yaml.resolve()

        merged_name = build_dataset_folder_name(class_names, kind="merged", created_at=datetime.now())
        merged_root = _build_unique_path(MERGED_DATASET_SAVE_DIR / merged_name)
        merged_root.mkdir(parents=True, exist_ok=True)

        alias_payload: dict[str, Any] = {
            "path": source_root.as_posix(),
            "train": train_entry,
            "val": valid_entry,
            "nc": int(len(class_names)),
            "names": {idx: name for idx, name in enumerate(class_names)},
        }
        if source_test_entry is not None:
            alias_payload["test"] = source_test_entry

        alias_yaml_path = merged_root / "data.yaml"
        alias_yaml_text = yaml.safe_dump(alias_payload, allow_unicode=True, sort_keys=False)
        alias_yaml_path.write_text(alias_yaml_text, encoding="utf-8")
        classes_text = "\n".join(f"{idx}: {name}" for idx, name in enumerate(class_names))
        (merged_root / "classes.txt").write_text(classes_text + ("\n" if classes_text else ""), encoding="utf-8")

        components = collect_original_components_from_yaml(source_yaml, fallback_class_names=class_names)
        write_provenance_readme(
            merged_root / "README.txt",
            created_at=datetime.now(),
            output_folder_name=str(merged_root.name),
            class_names=class_names,
            components=components,
        )
        logging.getLogger(__name__).info(
            "retrain merged alias created: source=%s alias=%s",
            source_yaml,
            alias_yaml_path,
        )
        return alias_yaml_path

    def _ensure_retrain_aliases_for_saved_datasets(self) -> None:
        """dataset_save_dir???쇰컲 ?곗씠?곗뀑?ㅼ쓣 ?ы븰?듭슜 merged YAML濡??숆린?뷀빀?덈떎."""
        if not DATASET_SAVE_DIR.is_dir():
            return
        for yaml_path in sorted(DATASET_SAVE_DIR.glob("*/data.yaml")):
            parent_name = str(yaml_path.parent.name).strip().casefold()
            if parent_name.startswith("merged_"):
                continue
            try:
                self._create_retrain_merged_alias_for_yaml(yaml_path)
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "failed to ensure retrain alias for dataset yaml (%s): %s",
                    yaml_path,
                    exc,
                )

    def _build_safe_training_run_name(self, run_project_dir: Path, base_name: str) -> str:
        """湲곕낯 run ?대쫫???대? 議댁옱?섎㈃ _v2, _v3... ?묐??щ? 遺숈뿬 異⑸룎???쇳빀?덈떎."""
        base = self._to_english_slug(str(base_name), fallback="train_run")
        if not base:
            base = "train_run"
        candidate = base
        serial = 2
        while (run_project_dir / candidate).exists():
            candidate = f"{base}_v{serial}"
            serial += 1
            if serial > 999:
                break
        return candidate

    def _resolve_retrain_source_data_yaml_from_model(self, model_source: str) -> Path | None:
        model_path = Path(str(model_source or "")).resolve()
        candidate_run_dirs: list[Path] = []
        if model_path.parent.name.casefold() == "weights" and model_path.parent.parent is not None:
            candidate_run_dirs.append(model_path.parent.parent.resolve())
        if model_path.parent is not None:
            candidate_run_dirs.append(model_path.parent.resolve())

        def _resolve_local_yaml(raw_path: object, base_dir: Path) -> Path | None:
            text = str(raw_path or "").strip()
            if not text:
                return None
            candidate = Path(text)
            if not candidate.is_absolute():
                candidate = (base_dir / candidate).resolve()
            else:
                candidate = candidate.resolve()
            return candidate if candidate.is_file() else None

        def _extract_yaml_from_readme(readme_path: Path, base_dir: Path) -> Path | None:
            payload = read_provenance_from_readme(readme_path)
            if not isinstance(payload, Mapping):
                return None

            raw_payload = payload.get("raw", {})
            if isinstance(raw_payload, Mapping):
                for key in ("data_yaml_path", "old_yaml_path", "merged_yaml_path"):
                    resolved = _resolve_local_yaml(raw_payload.get(key), base_dir)
                    if resolved is not None:
                        return resolved
                for key in ("components", "new_components", "old_components"):
                    component_list = raw_payload.get(key, [])
                    if isinstance(component_list, Sequence) and not isinstance(component_list, (str, bytes, bytearray)):
                        for item in component_list:
                            if not isinstance(item, Mapping):
                                continue
                            resolved = _resolve_local_yaml(item.get("data_yaml_path"), base_dir)
                            if resolved is not None:
                                return resolved

            components = payload.get("components", [])
            if isinstance(components, Sequence) and not isinstance(components, (str, bytes, bytearray)):
                for item in components:
                    if not isinstance(item, Mapping):
                        continue
                    resolved = _resolve_local_yaml(item.get("data_yaml_path"), base_dir)
                    if resolved is not None:
                        return resolved
            return None

        seen: set[str] = set()
        for run_dir in candidate_run_dirs:
            key = str(run_dir).casefold()
            if key in seen:
                continue
            seen.add(key)
            args_payload = self._read_training_args_from_run(run_dir)
            resolved_from_args = _resolve_local_yaml(args_payload.get("data"), run_dir)
            if resolved_from_args is not None:
                return resolved_from_args
            for readme_name in ("README.txt", "RETRAIN_README.txt"):
                resolved_from_readme = _extract_yaml_from_readme(run_dir / readme_name, run_dir)
                if resolved_from_readme is not None:
                    return resolved_from_readme
        return None

    def _begin_training_loading(self, message: str, sub_message: str = "Preparing training...") -> None:
        """?ы븰??蹂묓빀/?숈뒿 ?뚯씠?꾨씪???숈븞 濡쒕뵫 ?ㅻ쾭?덉씠瑜??쒖떆?⑸땲??"""
        if not self.training_loading_active:
            self._begin_loading(message)
            self.training_loading_active = True
        else:
            if self.loading_label is not None:
                self.loading_label.setText(str(message))
        if self.loading_sub_label is not None:
            self.loading_sub_label.setText(str(sub_message))

    def _update_training_loading(self, message: str, sub_message: str = "") -> None:
        if not self.training_loading_active:
            return
        if self.loading_label is not None:
            self.loading_label.setText(str(message))
        if sub_message and self.loading_sub_label is not None:
            self.loading_sub_label.setText(str(sub_message))

    def _end_training_loading(self) -> None:
        if not self.training_loading_active:
            return
        self.training_loading_active = False
        self._end_loading()

    def _mark_training_epoch_started(self) -> None:
        """?숈뒿???ㅼ젣 epoch 猷⑦봽??吏꾩엯?덉쓣 ??濡쒕뵫 ?ㅻ쾭?덉씠瑜?醫낅즺?⑸땲??"""
        if (not self.training_waiting_epoch_start) and (not self.training_loading_active):
            return
        self.training_waiting_epoch_start = False
        self._end_training_loading()

    def _start_training_worker(
        self,
        *,
        data_yaml_path: Path,
        model_source: str,
        train_engine: str,
        train_mode: str,
        task_name: str,
        epochs: int,
        imgsz: int,
        batch: int,
        patience: int,
        run_name: str | None = None,
        freeze: int | None = None,
        lr0: float = TRAIN_DEFAULT_LR0,
        retrain_recipe: Mapping[str, Any] | None = None,
        force_local: bool = False,
        extra_params: Mapping[str, Any] | None = None,
    ) -> None:
        """?숈뒿 ?뚯빱瑜??앹꽦??QThread?먯꽌 ?ㅽ뻾???쒖옉?⑸땲??"""
        self.training_data_yaml_path = Path(data_yaml_path).resolve()
        run_project_dir = TRAIN_RUNS_DIR
        if run_name:
            resolved_run_name = str(run_name).strip()
        else:
            resolved_run_name = f"{train_engine}_{train_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.training_started_at = datetime.now()
        self.training_args_snapshot = {
            "engine": str(train_engine),
            "mode": str(train_mode),
            "task": str(task_name),
            "model_source": str(model_source),
            "epochs": int(epochs),
            "imgsz": int(imgsz),
            "batch": int(batch),
            "patience": int(patience),
            "data_yaml_path": str(self.training_data_yaml_path),
            "freeze": None if freeze is None else int(max(0, freeze)),
            "lr0": float(max(0.000001, lr0)),
            "retrain_recipe": dict(retrain_recipe) if isinstance(retrain_recipe, Mapping) else None,
        }
        self.training_thread = QThread(self)
        self.training_worker = YoloTrainWorker(
            data_yaml_path=self.training_data_yaml_path,
            model_source=model_source,
            train_engine=train_engine,
            train_mode=train_mode,
            task_name=task_name,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            run_project_dir=run_project_dir,
            run_name=resolved_run_name,
            freeze=freeze,
            lr0=lr0,
            retrain_recipe=retrain_recipe,
            force_local=force_local,
            extra_params=extra_params,
        )
        self.training_worker.moveToThread(self.training_thread)
        self.training_thread.started.connect(self.training_worker.run)
        self.training_worker.log_message.connect(self._on_training_log_message)
        self.training_worker.terminal_output.connect(self._on_training_terminal_output)
        self.training_worker.metric_changed.connect(self._on_training_metric_changed)
        self.training_worker.batch_progress.connect(self._on_training_batch_progress)
        self.training_worker.finished.connect(self._on_training_finished)
        self.training_worker.failed.connect(self._on_training_failed)
        self.training_worker.finished.connect(self.training_thread.quit)
        self.training_worker.failed.connect(self.training_thread.quit)
        self.training_thread.finished.connect(self._on_training_thread_finished)
        self.training_thread.finished.connect(self.training_worker.deleteLater)
        self.training_thread.finished.connect(self.training_thread.deleteLater)

        self.is_training = True
        self.training_merge_active = False
        self._update_training_ui_state()
        self._update_button_state()
        freeze_text = "" if freeze is None else f"freeze={int(max(0, freeze))}, "
        self._append_train_log(
            f"start training: engine={train_engine}, mode={train_mode}, task={task_name}, "
            f"model={model_source}, epochs={epochs}, imgsz={imgsz}, batch={batch}, "
            f"{freeze_text}"
            f"lr0={float(max(0.000001, lr0)):.6f}, "
            f"patience={patience}, data={self.training_data_yaml_path}, run={resolved_run_name}"
        )
        if isinstance(retrain_recipe, Mapping):
            self._append_train_log(
                "retrain recipe: "
                f"replay_old={float(retrain_recipe.get('replay_ratio_old', TRAIN_REPLAY_RATIO_DEFAULT)):.2f}, "
                f"stage1_epochs={int(retrain_recipe.get('stage1_epochs', TRAIN_STAGE1_EPOCHS_DEFAULT))}, "
                f"stage2_epochs={int(retrain_recipe.get('stage2_epochs', TRAIN_STAGE2_EPOCHS_DEFAULT))}, "
                f"stage2_lr_factor={float(retrain_recipe.get('stage2_lr_factor', TRAIN_STAGE2_LR_FACTOR_DEFAULT)):.2f}, "
                f"unfreeze={str(retrain_recipe.get('unfreeze_mode', TRAIN_STAGE_UNFREEZE_NECK_ONLY))}, "
                f"seed={int(retrain_recipe.get('seed', TRAIN_RETRAIN_SEED_DEFAULT))}"
            )
        self._training_terminal_last_line = ""
        self.training_waiting_epoch_start = True
        self.training_thread.start()

    def _start_training_merge_worker(
        self,
        yaml_paths: Sequence[Path],
        *,
        old_yaml_path: Path,
        replay_ratio_old: float,
        seed: int,
    ) -> None:
        """?ы븰???ㅼ쨷 YAML 蹂묓빀 ?뚯빱瑜??쒖옉?⑸땲??"""
        merge_output_root = MERGED_DATASET_SAVE_DIR
        merge_output_root.mkdir(parents=True, exist_ok=True)
        self.training_merge_thread = QThread(self)
        self.training_merge_worker = ReplayDatasetMergeWorker(
            new_yaml_paths=yaml_paths,
            old_yaml_path=old_yaml_path,
            output_root=merge_output_root,
            replay_ratio_old=replay_ratio_old,
            seed=seed,
        )
        self.training_merge_worker.moveToThread(self.training_merge_thread)
        self.training_merge_thread.started.connect(self.training_merge_worker.run)
        self.training_merge_worker.progress.connect(self._on_training_merge_progress)
        self.training_merge_worker.finished.connect(self._on_training_merge_finished)
        self.training_merge_worker.failed.connect(self._on_training_merge_failed)
        self.training_merge_worker.finished.connect(self.training_merge_thread.quit)
        self.training_merge_worker.failed.connect(self.training_merge_thread.quit)
        self.training_merge_thread.finished.connect(self._on_training_merge_thread_finished)
        self.training_merge_thread.finished.connect(self.training_merge_worker.deleteLater)
        self.training_merge_thread.finished.connect(self.training_merge_thread.deleteLater)

        self.is_training = True
        self.training_merge_active = True
        self.training_waiting_epoch_start = False
        self._update_training_ui_state()
        self._update_button_state()
        self._append_train_log(
            f"start merge: new_yaml_count={len(yaml_paths)}, old_yaml={Path(old_yaml_path).resolve()}, "
            f"replay_old={float(replay_ratio_old):.2f}, seed={int(seed)}"
        )
        self.training_merge_thread.start()

    def on_start_training(self) -> None:
        """?좏깮???붿쭊(YOLO/RT-DETR)怨?紐⑤뱶(?좉퇋/?ы븰??濡??숈뒿 ?ㅻ젅?쒕? ?쒖옉?⑸땲??"""
        if self.is_training:
            return
        if self.is_processing or self.is_exporting or self.is_model_testing:
            QMessageBox.information(self, "Training", "Finish the current task first, then start training.")
            return
        _sync_ultralytics_datasets_dir()

        selected_engine = "yolo"
        if self.comboTrainEngine is not None:
            selected_engine = str(self.comboTrainEngine.currentData() or "yolo").strip().lower() or "yolo"
        self.training_engine_name = selected_engine if selected_engine in {"yolo", "rtdetr"} else "yolo"
        self._update_training_metric_chart_titles()
        self._refresh_training_model_combo()
        if self.comboTrainModel is not None:
            selected_model = str(self.comboTrainModel.currentData() or "").strip()
            self.training_model_source = selected_model
        model_source = str(self.training_model_source or "").strip()
        if not model_source:
            QMessageBox.warning(self, "Training", "Select a training model first.")
            return

        selected_mode = "new"
        if self.checkTrainRetrain is not None and self.checkTrainRetrain.isChecked():
            selected_mode = "retrain"
        elif self.checkTrainNew is not None and self.checkTrainNew.isChecked():
            selected_mode = "new"
        self.training_mode_name = selected_mode

        self.training_task_name = "detect"
        if self.training_mode_name == "retrain" and Path(model_source).name.casefold() != "best.pt":
            QMessageBox.warning(self, "Retrain", "Retrain mode requires selecting a best.pt model.")
            return
        if self.training_mode_name == "retrain" and self.training_engine_name != "yolo":
            QMessageBox.warning(self, "Retrain", "2-stage retrain currently supports YOLO Detect only.")
            return
        if self.training_engine_name == "rtdetr" and ("rtdetr" not in model_source.lower()):
            self._append_train_log("warning: RT-DETR works best with rtdetr-*.pt weights.")

        epochs = int(self.spinTrainEpochs.value()) if self.spinTrainEpochs is not None else 200
        imgsz = int(self.spinTrainImgsz.value()) if self.spinTrainImgsz is not None else 640
        batch = int(self.spinTrainBatch.value()) if self.spinTrainBatch is not None else -1
        patience = 50
        advanced_settings = self._current_training_advanced_settings()
        freeze: int | None = None
        lr0 = float(advanced_settings.get("lr0", TRAIN_DEFAULT_LR0))
        retrain_recipe: dict[str, Any] | None = None
        total_epochs_for_progress = int(epochs)
        if self.training_mode_name == "retrain":
            stage1_epochs = int(advanced_settings.get("stage1_epochs", TRAIN_STAGE1_EPOCHS_DEFAULT))
            stage2_epochs = int(advanced_settings.get("stage2_epochs", TRAIN_STAGE2_EPOCHS_DEFAULT))
            total_epochs_for_progress = max(1, stage1_epochs + stage2_epochs)
            retrain_recipe = {
                "stage1_epochs": stage1_epochs,
                "stage2_epochs": stage2_epochs,
                "stage2_lr_factor": float(advanced_settings.get("stage2_lr_factor", TRAIN_STAGE2_LR_FACTOR_DEFAULT)),
                "unfreeze_mode": str(advanced_settings.get("unfreeze_mode", TRAIN_STAGE_UNFREEZE_NECK_ONLY)),
                "replay_ratio_old": float(advanced_settings.get("replay_ratio_old", TRAIN_REPLAY_RATIO_DEFAULT)),
                "seed": int(advanced_settings.get("replay_seed", TRAIN_RETRAIN_SEED_DEFAULT)),
            }

        for chart in (
            self.trainingBoxLossChart,
            self.trainingClsLossChart,
            self.trainingDflLossChart,
            self.trainingAccChart,
        ):
            if chart is not None:
                chart.reset()
        if self.progressTrainTotal is not None:
            self.progressTrainTotal.setValue(0)
        self._set_training_progress_summary(0, 0, max(1, total_epochs_for_progress))
        self._train_progress_anchor_pos = None
        self._train_progress_line_text = ""
        self._train_progress_last_key = None
        self._pending_training_request = None

        if self.training_mode_name == "retrain":
            selected_yaml_paths = self._collect_retrain_yaml_paths_for_training()
            if not selected_yaml_paths:
                QMessageBox.warning(self, "Retrain", "Select a new data.yaml for retrain.")
                return
            for path in selected_yaml_paths:
                if (not path.is_file()) or (not self._is_valid_training_data_yaml(path)):
                    QMessageBox.warning(
                        self,
                        "Retrain",
                        "Retrain requires a valid new data.yaml.\n"
                        f"{path}",
                    )
                    return
                if _try_autofix_data_yaml_path(path):
                    self._append_train_log(f"data.yaml path auto-fixed: {path}")
                if _try_autofix_data_yaml_splits(path):
                    self._append_train_log(f"data.yaml split auto-fixed: {path}")
            old_data_yaml = self._resolve_retrain_source_data_yaml_from_model(model_source)
            if old_data_yaml is None or (not old_data_yaml.is_file()):
                QMessageBox.warning(
                    self,
                    "Retrain",
                    "Could not find previous training data.yaml from selected best.pt.\n"
                    "Select a model under runs/train/<run>/weights/best.pt.",
                )
                return
            if _try_autofix_data_yaml_path(old_data_yaml):
                self._append_train_log(f"old data.yaml path auto-fixed: {old_data_yaml}")
            if _try_autofix_data_yaml_splits(old_data_yaml):
                self._append_train_log(f"old data.yaml split auto-fixed: {old_data_yaml}")

            _pending_use_local = bool(
                self.checkTrainLocal is not None and self.checkTrainLocal.isChecked()
            )
            _pending_extra_params: dict[str, Any] = dict(getattr(self, "_yolo_extra_params", {}) or {})
            if _pending_use_local:
                _pending_local_device_str, _pending_local_device_desc = self._detect_local_device()
                if _pending_local_device_str == "cpu":
                    ret = QMessageBox.warning(
                        self,
                        "로컬 학습 — GPU 없음",
                        "CUDA GPU가 감지되지 않았습니다.\nCPU로 학습을 계속하면 매우 느릴 수 있습니다.\n\n계속하시겠습니까?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )
                    if ret != QMessageBox.StandardButton.Yes:
                        return
                _pending_extra_params["device"] = _pending_local_device_str
            self._pending_training_request = {
                "model_source": model_source,
                "train_engine": self.training_engine_name,
                "train_mode": self.training_mode_name,
                "task_name": self.training_task_name,
                "epochs": int(total_epochs_for_progress),
                "imgsz": int(imgsz),
                "batch": int(batch),
                "patience": int(patience),
                "freeze": None,
                "lr0": float(lr0),
                "retrain_recipe": dict(retrain_recipe or {}),
                "old_yaml_path": str(old_data_yaml),
                "force_local": _pending_use_local,
                "extra_params": _pending_extra_params if _pending_extra_params else None,
            }
            self._append_train_log(
                f"retrain base dataset: old_yaml={old_data_yaml}, new_yaml={selected_yaml_paths[0]}"
            )
            self._begin_training_loading("Validating", "Preparing replay dataset...")
            self._start_training_merge_worker(
                selected_yaml_paths,
                old_yaml_path=old_data_yaml,
                replay_ratio_old=float(retrain_recipe.get("replay_ratio_old", TRAIN_REPLAY_RATIO_DEFAULT)) if retrain_recipe else TRAIN_REPLAY_RATIO_DEFAULT,
                seed=int(retrain_recipe.get("seed", TRAIN_RETRAIN_SEED_DEFAULT)) if retrain_recipe else TRAIN_RETRAIN_SEED_DEFAULT,
            )
            return

        if self.training_data_yaml_path is None and self.last_export_dataset_root is not None:
            fallback = self.last_export_dataset_root / "data.yaml"
            if fallback.is_file():
                self._set_training_dataset_path(fallback)
        if self.training_data_yaml_path is None or (not self.training_data_yaml_path.is_file()):
            QMessageBox.warning(self, "Training", "Select a data.yaml file first.")
            return
        if _try_autofix_data_yaml_path(self.training_data_yaml_path):
            self._append_train_log(f"data.yaml path auto-fixed: {self.training_data_yaml_path}")
        if _try_autofix_data_yaml_splits(self.training_data_yaml_path):
            self._append_train_log(f"data.yaml split auto-fixed: {self.training_data_yaml_path}")
        if self.training_mode_name == "new":
            try:
                alias_yaml = self._create_retrain_merged_alias_for_yaml(self.training_data_yaml_path)
                if alias_yaml is not None:
                    self._append_train_log(f"retrain merged YAML prepared: {alias_yaml}")
            except Exception as exc:
                self._append_train_log(f"warning: failed to prepare retrain merged YAML ({exc})")

        # 로컬/서버 학습 분기 처리 및 고급 파라미터 수집
        use_local = bool(
            self.checkTrainLocal is not None and self.checkTrainLocal.isChecked()
        )
        extra_params_to_use: dict[str, Any] = dict(getattr(self, "_yolo_extra_params", {}) or {})
        local_device_str = ""
        if use_local:
            local_device_str, local_device_desc = self._detect_local_device()
            if local_device_str == "cpu":
                ret = QMessageBox.warning(
                    self,
                    "로컬 학습 — GPU 없음",
                    "CUDA GPU가 감지되지 않았습니다.\nCPU로 학습을 계속하면 매우 느릴 수 있습니다.\n\n계속하시겠습니까?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if ret != QMessageBox.StandardButton.Yes:
                    return
            extra_params_to_use["device"] = local_device_str
            if extra_params_to_use.get("device") is not None:
                self._append_train_log(f"로컬 학습 장치: {local_device_desc}")

        self._begin_training_loading("Training", "Preparing training...")
        self._start_training_worker(
            data_yaml_path=self.training_data_yaml_path,
            model_source=model_source,
            train_engine=self.training_engine_name,
            train_mode=self.training_mode_name,
            task_name=self.training_task_name,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            freeze=freeze,
            lr0=lr0,
            retrain_recipe=retrain_recipe,
            force_local=use_local,
            extra_params=extra_params_to_use if extra_params_to_use else None,
        )

    def on_stop_training(self) -> None:
        """?ㅽ뻾 以묒씤 ?숈뒿??以묒? ?붿껌??蹂대깄?덈떎(?꾩옱 epoch 醫낅즺 吏?먯뿉??諛섏쁺)."""
        if not self.is_training or self.training_worker is None:
            return
        self.training_worker.request_stop()
        self._append_train_log("stop requested")
        self._update_button_state()

    def _on_training_merge_progress(self, stage: str, done: int, total: int) -> None:
        total_safe = max(1, int(total))
        done_safe = max(0, min(int(done), total_safe))
        stage_text = str(stage).strip() or "Merging"
        self._update_training_loading(f"{stage_text}: {done_safe}/{total_safe}", "Preparing retrain dataset...")

    def _on_training_merge_finished(self, payload_obj: object) -> None:
        try:
            if not isinstance(payload_obj, Mapping):
                raise RuntimeError("蹂묓빀 寃곌낵 ?뺤떇???щ컮瑜댁? ?딆뒿?덈떎.")
            merged_yaml = Path(str(payload_obj.get("merged_yaml_path", ""))).resolve()
            merged_root = Path(str(payload_obj.get("merged_dataset_root", ""))).resolve()
            run_base_name = str(payload_obj.get("run_base_name", "")).strip()
            if (not merged_yaml.is_file()) or (not run_base_name):
                raise RuntimeError("蹂묓빀 data.yaml ?앹꽦???ㅽ뙣?덉뒿?덈떎.")
            if not merged_root.is_dir():
                raise RuntimeError("蹂묓빀 ?곗씠?곗뀑 ?대뜑媛 ?놁뒿?덈떎.")

            pending = dict(self._pending_training_request or {})
            model_source = str(pending.get("model_source", self.training_model_source))
            train_engine = str(pending.get("train_engine", self.training_engine_name))
            train_mode = str(pending.get("train_mode", self.training_mode_name))
            task_name = str(pending.get("task_name", self.training_task_name))
            epochs = int(pending.get("epochs", self.spinTrainEpochs.value() if self.spinTrainEpochs is not None else 200))
            imgsz = int(pending.get("imgsz", self.spinTrainImgsz.value() if self.spinTrainImgsz is not None else 640))
            batch = int(pending.get("batch", self.spinTrainBatch.value() if self.spinTrainBatch is not None else -1))
            patience = int(pending.get("patience", 50))
            freeze_value = pending.get("freeze", None)
            freeze = int(freeze_value) if freeze_value is not None else None
            lr0 = float(pending.get("lr0", TRAIN_DEFAULT_LR0))
            retrain_recipe = pending.get("retrain_recipe", {})
            pending_force_local = bool(pending.get("force_local", False))
            pending_extra_params = pending.get("extra_params", None)
            if not isinstance(retrain_recipe, Mapping):
                retrain_recipe = {}

            run_project_dir = TRAIN_RUNS_DIR
            safe_run_name = self._build_safe_training_run_name(run_project_dir, run_base_name)
            self._set_training_dataset_path(merged_yaml)
            self._append_train_log(
                "merge done: "
                f"root={merged_root}, yaml={merged_yaml}, run_base={safe_run_name}, "
                f"new_train={int(payload_obj.get('new_train_count', 0) or 0)}, "
                f"replay_old={int(payload_obj.get('old_replay_count', 0) or 0)}, "
                f"old_available={int(payload_obj.get('old_train_available_count', 0) or 0)}"
            )
            self.training_merge_active = False
            self._update_training_loading("Training", "蹂묓빀 ?꾨즺. ?숈뒿???쒖옉?⑸땲??..")
            self._start_training_worker(
                data_yaml_path=merged_yaml,
                model_source=model_source,
                train_engine=train_engine,
                train_mode=train_mode,
                task_name=task_name,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                patience=patience,
                run_name=safe_run_name,
                freeze=freeze,
                lr0=lr0,
                retrain_recipe={
                    **dict(retrain_recipe),
                    "merged_dataset_root": str(merged_root),
                    "merged_yaml_path": str(merged_yaml),
                    "old_eval_yaml_path": str(payload_obj.get("old_eval_yaml_path", "")),
                    "old_yaml_path": str(payload_obj.get("old_yaml_path", "")),
                    "new_yaml_paths": list(payload_obj.get("new_yaml_paths", [])) if isinstance(payload_obj.get("new_yaml_paths", []), Sequence) and not isinstance(payload_obj.get("new_yaml_paths", []), (str, bytes, bytearray)) else [],
                    "new_train_count": int(payload_obj.get("new_train_count", 0) or 0),
                    "old_train_available_count": int(payload_obj.get("old_train_available_count", 0) or 0),
                    "old_replay_count": int(payload_obj.get("old_replay_count", 0) or 0),
                    "merged_train_count": int(payload_obj.get("merged_train_count", 0) or 0),
                    "new_valid_count": int(payload_obj.get("new_valid_count", 0) or 0),
                    "new_test_count": int(payload_obj.get("new_test_count", 0) or 0),
                },
                force_local=pending_force_local,
                extra_params=pending_extra_params,
            )
            self._pending_training_request = None
        except Exception as exc:
            self._on_training_merge_failed(str(exc))

    def _on_training_merge_failed(self, message: str) -> None:
        self.training_merge_active = False
        self._pending_training_request = None
        self.training_waiting_epoch_start = False
        self.is_training = False
        self._append_train_log(f"error: retrain merge failed ({message})")
        self._end_training_loading()
        QMessageBox.critical(self, "?ы븰??蹂묓빀 ?ㅻ쪟", f"?ы븰???곗씠?곗뀑 蹂묓빀???ㅽ뙣?덉뒿?덈떎.\n{message}")
        self._update_training_ui_state()
        self._update_button_state()

    def _on_training_merge_thread_finished(self) -> None:
        self.training_merge_thread = None
        self.training_merge_worker = None
        if not self.is_training:
            self.training_merge_active = False
            self._update_training_ui_state()
            self._update_button_state()

    def _on_training_log_message(self, message: str) -> None:
        self._append_train_log(str(message))

    def _on_training_terminal_output(self, text: str) -> None:
        value = str(text)
        if not value:
            return
        self._update_training_progress_from_terminal(value)
        # Keep useful terminal lines in log; skip overly noisy tqdm progress updates.
        last_line = str(getattr(self, "_training_terminal_last_line", ""))
        for raw_line in value.replace("\r", "\n").splitlines():
            line = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", str(raw_line)).strip()
            if not line:
                continue
            upper = line.upper()
            if ("%|" in line and "IT/S" in upper) or ("TQDM" in upper):
                continue
            if line == last_line:
                continue
            last_line = line
            self._append_train_log(f"terminal: {line}")
        self._training_terminal_last_line = last_line

    def _update_training_progress_from_terminal(self, text: str) -> None:
        """?곕???異쒕젰?먯꽌 epoch/batch 吏꾪뻾 臾몄옄?댁쓣 ?뚯떛??吏꾪뻾瑜?諛붾? 蹂댁젙?⑸땲??"""
        if not text:
            return

        lines = str(text).replace("\r", "\n").splitlines()
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            all_pairs = re.findall(r"(\d+)\s*/\s*(\d+)", line)
            epoch_now = epoch_total = None
            batch_now = batch_total = None
            if all_pairs:
                try:
                    epoch_now = max(1, int(all_pairs[0][0]))
                    epoch_total = max(1, int(all_pairs[0][1]))
                except Exception:
                    epoch_now = epoch_total = None
                if len(all_pairs) >= 2:
                    try:
                        batch_now = max(1, int(all_pairs[-1][0]))
                        batch_total = max(1, int(all_pairs[-1][1]))
                    except Exception:
                        batch_now = batch_total = None

            if epoch_now is not None and epoch_total is not None:
                self._mark_training_epoch_started()
                percent = int(round((epoch_now / float(epoch_total)) * 100.0))
                current_percent = int(self.progressTrainTotal.value()) if self.progressTrainTotal is not None else 0
                self._set_training_progress_summary(
                    max(current_percent, percent),
                    int(epoch_now),
                    int(epoch_total),
                )

            percent_match = re.search(r"(\d{1,3}(?:\.\d+)?)%\|", line)
            if percent_match:
                try:
                    ep_percent_f = max(0.0, min(100.0, float(percent_match.group(1))))
                except Exception:
                    ep_percent_f = 0.0
                if (
                    epoch_now is not None
                    and epoch_total is not None
                    and (batch_now is None or batch_total is None)
                ):
                    batch_total = 100
                    batch_now = max(1, int(round(ep_percent_f)))

            if (
                epoch_now is not None
                and epoch_total is not None
                and batch_now is not None
                and batch_total is not None
            ):
                self._update_training_epoch_progress_log_line(
                    int(epoch_now),
                    int(epoch_total),
                    int(batch_now),
                    int(batch_total),
                )

    def _on_training_batch_progress(self, payload: object) -> None:
        if not isinstance(payload, Mapping):
            return
        self._mark_training_epoch_started()
        epoch_num = int(float(payload.get("epoch", 1)))
        total_epochs = max(1, int(float(payload.get("total_epochs", 1))))
        batch_now = int(float(payload.get("batch", 1)))
        num_batches = max(1, int(float(payload.get("num_batches", 1))))
        total_progress = float(payload.get("total_progress", 0.0))
        total_percent = int(round(max(0.0, min(1.0, total_progress)) * 100.0))
        self._set_training_progress_summary(total_percent, epoch_num, total_epochs)
        self._update_training_epoch_progress_log_line(epoch_num, total_epochs, batch_now, num_batches)
        if self.training_loading_active:
            self._update_training_loading(
                f"Training: epoch {epoch_num}/{total_epochs}, batch {batch_now}/{num_batches}",
                "Training in progress...",
            )

    def _on_training_metric_changed(self, metric_obj: object) -> None:
        if not isinstance(metric_obj, Mapping):
            return
        self._mark_training_epoch_started()
        epoch = int(float(metric_obj.get("epoch", 0)))
        total_epochs = max(1, int(float(metric_obj.get("total_epochs", 1))))
        box_loss = float(metric_obj.get("box_loss", 0.0))
        cls_loss = float(metric_obj.get("cls_loss", 0.0))
        dfl_loss = float(metric_obj.get("dfl_loss", 0.0))
        acc = float(metric_obj.get("accuracy", 0.0))

        epoch_percent = int(round((epoch / float(total_epochs)) * 100.0))
        if self.trainingBoxLossChart is not None:
            self.trainingBoxLossChart.add_point(epoch, box_loss)
        if self.trainingClsLossChart is not None:
            self.trainingClsLossChart.add_point(epoch, cls_loss)
        if self.trainingDflLossChart is not None:
            self.trainingDflLossChart.add_point(epoch, dfl_loss)
        if self.trainingAccChart is not None:
            self.trainingAccChart.add_point(epoch, acc)
        self._set_training_progress_summary(epoch_percent, epoch, total_epochs)

    def _read_class_names_from_data_yaml(self, data_yaml_path: Path | None) -> list[str]:
        if data_yaml_path is None:
            return []
        payload = _load_yaml_dict(Path(data_yaml_path))
        names = _parse_names_from_yaml_payload(payload)
        return _dedupe_preserve_order(names)

    def _training_model_label(self) -> str:
        return "RT-DETR" if str(self.training_engine_name).strip().lower() == "rtdetr" else "YOLO"

    def _build_training_output_folder_name(self, metrics: Mapping[str, Any]) -> str:
        class_names = self._read_class_names_from_data_yaml(self.training_data_yaml_path)
        sanitized = [sanitize_class_name(name) for name in class_names if str(name).strip()]
        class_part = "_".join(sanitized) if sanitized else "class"
        m50 = _metric_to_unit_interval(float(metrics.get("overall_map50", 0.0)))
        m50_95 = _metric_to_unit_interval(float(metrics.get("overall_map50_95", 0.0)))
        return f"{self._training_model_label()}_trained_{class_part}_m50({m50:.3f})_m50-95({m50_95:.3f})"

    def _rename_training_run_dir(self, save_dir: Path, metrics: Mapping[str, Any]) -> Path:
        """?숈뒿 寃곌낵 ?대뜑紐낆쓣 洹쒖튃??留욊쾶 蹂寃쏀빐 諛섑솚?⑸땲??"""
        if not save_dir.is_dir():
            return save_dir
        parent = save_dir.parent
        base_name = self._build_training_output_folder_name(metrics)
        target = _build_unique_path(parent / base_name)
        if target == save_dir:
            return save_dir
        try:
            save_dir.rename(target)
            logging.getLogger(__name__).info("training run renamed: old=%s new=%s", save_dir, target)
            return target
        except Exception as exc:
            logging.getLogger(__name__).warning("training run rename failed: %s", exc)
            return save_dir

    def _read_training_args_from_run(self, run_dir: Path) -> dict[str, Any]:
        args_path = Path(run_dir) / "args.yaml"
        payload = _load_yaml_dict(args_path)
        return dict(payload) if isinstance(payload, Mapping) else {}

    def _write_training_readme(
        self,
        *,
        run_dir: Path,
        metrics: Mapping[str, Any],
        ended_at: datetime,
    ) -> None:
        run_path = Path(run_dir).resolve()
        started_at = self.training_started_at or ended_at
        args_file = self._read_training_args_from_run(run_path)
        args_snapshot = dict(self.training_args_snapshot)

        data_yaml_raw = str(
            args_file.get(
                "data",
                args_snapshot.get(
                    "data_yaml_path",
                    str(self.training_data_yaml_path) if self.training_data_yaml_path is not None else "",
                ),
            )
        ).strip()
        data_yaml_path: Path | None = None
        if data_yaml_raw:
            candidate = Path(data_yaml_raw)
            if not candidate.is_absolute():
                candidate = (run_path / candidate).resolve()
            else:
                candidate = candidate.resolve()
            data_yaml_path = candidate
        if (data_yaml_path is None or (not data_yaml_path.is_file())) and self.training_data_yaml_path is not None:
            fallback_yaml = Path(self.training_data_yaml_path).resolve()
            if fallback_yaml.is_file():
                data_yaml_path = fallback_yaml

        class_names: list[str] = []
        provenance_components: list[dict[str, Any]] = []
        if data_yaml_path is not None and data_yaml_path.is_file():
            class_names = self._read_class_names_from_data_yaml(data_yaml_path)
            provenance_components = collect_original_components_from_yaml(data_yaml_path, fallback_class_names=class_names)

        best_pt = run_path / "weights" / "best.pt"
        best_pt_text = str(best_pt) if best_pt.is_file() else "(best.pt ?놁쓬)"
        map50 = _metric_to_unit_interval(float(metrics.get("overall_map50", 0.0)))
        map50_95 = _metric_to_unit_interval(float(metrics.get("overall_map50_95", 0.0)))
        best_epoch = int(metrics.get("best_epoch", 0) or 0)
        losses = metrics.get("losses", {})
        if not isinstance(losses, Mapping):
            losses = {}
        extra_losses = metrics.get("extra_losses", {})
        if not isinstance(extra_losses, Mapping):
            extra_losses = {}
        per_class = metrics.get("per_class", [])
        if not isinstance(per_class, Sequence) or isinstance(per_class, (str, bytes, bytearray)):
            per_class = []
        derivation = str(metrics.get("derivation", "?곗텧 諛⑹떇 ?뺣낫 ?놁쓬")).strip() or "?곗텧 諛⑹떇 ?뺣낫 ?놁쓬"

        data_yaml_text = str(data_yaml_path) if data_yaml_path is not None else "(data.yaml 寃쎈줈 ?뺣낫 ?놁쓬)"
        merged_from_names = [str(item.get("dataset_name", "dataset")).strip() for item in provenance_components]
        merged_from_names = [name for name in merged_from_names if name]

        def _safe_float(value: object, default: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:
                return float(default)

        lines: list[str] = []
        lines.append("?숈뒿 寃곌낵 ?앹꽦 ?뺣낫")
        lines.append("")
        lines.append("?숈뒿 ?붿빟")
        lines.append(f"- 紐⑤뜽 ?좏삎: {self._training_model_label()}")
        lines.append(f"- ?숈뒿 ?쒖옉 ?쒓컖: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- ?숈뒿 醫낅즺 ?쒓컖: {ended_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- ?ㅽ뻾 ?대뜑: {run_path}")
        if args_snapshot.get("retrain_recipe"):
            retrain_readme = run_path / "RETRAIN_README.txt"
            retrain_recipe_json = run_path / "train_recipe.json"
            if retrain_readme.is_file():
                lines.append(f"- ?ы븰???붿빟 ?뚯씪: {retrain_readme}")
            if retrain_recipe_json.is_file():
                lines.append(f"- ?ы븰??provenance JSON: {retrain_recipe_json}")
        lines.append("- ?숈뒿 ?몄옄")
        merged_args = {
            "epochs": args_file.get("epochs", args_snapshot.get("epochs")),
            "imgsz": args_file.get("imgsz", args_snapshot.get("imgsz")),
            "batch": args_file.get("batch", args_snapshot.get("batch")),
            "patience": args_file.get("patience", args_snapshot.get("patience")),
            "device": args_file.get("device", "auto"),
            "model": args_file.get("model", args_snapshot.get("model_source")),
            "task": args_file.get("task", args_snapshot.get("task")),
            "mode": args_file.get("mode", args_snapshot.get("mode")),
        }
        for key, value in merged_args.items():
            lines.append(f"  - {key}: {value}")
        lines.append("")
        lines.append("?ъ슜 ?곗씠?곗뀑")
        lines.append(f"- data.yaml 寃쎈줈: {data_yaml_text}")
        lines.append(f"- 理쒖쥌 ?대옒??紐⑸줉: {', '.join(class_names) if class_names else '-'}")
        lines.append(f"- 蹂묓빀 ?먮낯: {' + '.join(merged_from_names) if merged_from_names else '-'}")
        lines.append("- 援ъ꽦 ?곗씠?곗뀑")
        if provenance_components:
            for idx, component in enumerate(provenance_components, start=1):
                split_counts = component.get("split_counts", {})
                train = split_counts.get("train", {"images": 0, "labels": 0}) if isinstance(split_counts, Mapping) else {"images": 0, "labels": 0}
                valid = split_counts.get("valid", {"images": 0, "labels": 0}) if isinstance(split_counts, Mapping) else {"images": 0, "labels": 0}
                test = split_counts.get("test", {"images": 0, "labels": 0}) if isinstance(split_counts, Mapping) else {"images": 0, "labels": 0}
                comp_classes = component.get("class_names", [])
                class_text = ", ".join(comp_classes) if isinstance(comp_classes, Sequence) and not isinstance(comp_classes, (str, bytes, bytearray)) else "-"
                lines.append(f"  {idx}) ?곗씠?곗뀑 ?대뜑紐? {component.get('dataset_name', 'dataset')}")
                lines.append(f"     - data.yaml 寃쎈줈: {component.get('data_yaml_path', '(寃쎈줈 ?뺣낫 ?놁쓬)')}")
                lines.append(f"     - ?대옒??紐⑸줉: {class_text}")
                lines.append(
                    "     - 遺꾪븷蹂??대?吏/?쇰꺼 ?? "
                    f"train {int(train.get('images', 0))}/{int(train.get('labels', 0))}, "
                    f"valid {int(valid.get('images', 0))}/{int(valid.get('labels', 0))}, "
                    f"test {int(test.get('images', 0))}/{int(test.get('labels', 0))}"
                )
        else:
            lines.append("  - 蹂묓빀/?먮낯 援ъ꽦 ?뺣낫瑜??뺤씤?????놁뼱 ?몃? 紐⑸줉???앸왂?⑸땲??")
        lines.append("")
        lines.append("理쒖쟻 泥댄겕?ъ씤???깅뒫")
        lines.append(f"- best.pt 寃쎈줈: {best_pt_text}")
        lines.append(f"- 理쒖쟻 epoch: {best_epoch}")
        lines.append(f"- ?꾩껜 mAP50: {map50:.3f}")
        lines.append(f"- ?꾩껜 mAP50-95: {map50_95:.3f}")
        lines.append("- ?대옒?ㅻ퀎 ?깅뒫")
        if per_class:
            for item in per_class:
                if not isinstance(item, Mapping):
                    continue
                class_name = str(item.get("class_name", "class")).strip() or "class"
                if class_name.isdigit() and class_names:
                    class_idx = int(class_name)
                    if 0 <= class_idx < len(class_names):
                        class_name = class_names[class_idx]
                pc_m50 = _metric_to_unit_interval(_safe_float(item.get("map50", 0.0), 0.0))
                pc_m95 = _metric_to_unit_interval(_safe_float(item.get("map50_95", 0.0), 0.0))
                lines.append(f"  - {class_name}: mAP50={pc_m50:.3f}, mAP50-95={pc_m95:.3f}")
        else:
            lines.append("  - 寃곌낵 ?뚯씪?먯꽌 ?대옒?ㅻ퀎 ?깅뒫???쒓났?섏? ?딆븘 ?앸왂")
        lines.append("")
        lines.append("理쒖쟻 泥댄겕?ъ씤??Loss")
        lines.append(f"- train/box_loss: {_safe_float(losses.get('train_box_loss', 0.0), 0.0):.6f}")
        lines.append(f"- train/cls_loss: {_safe_float(losses.get('train_cls_loss', 0.0), 0.0):.6f}")
        lines.append(f"- train/dfl_loss: {_safe_float(losses.get('train_dfl_loss', 0.0), 0.0):.6f}")
        lines.append(f"- val/box_loss: {_safe_float(losses.get('val_box_loss', 0.0), 0.0):.6f}")
        lines.append(f"- val/cls_loss: {_safe_float(losses.get('val_cls_loss', 0.0), 0.0):.6f}")
        lines.append(f"- val/dfl_loss: {_safe_float(losses.get('val_dfl_loss', 0.0), 0.0):.6f}")
        if extra_losses:
            lines.append("- 異붽? loss ??ぉ")
            for loss_name, loss_value in sorted(extra_losses.items(), key=lambda item: str(item[0]).casefold()):
                lines.append(f"  - {loss_name}: {_safe_float(loss_value, 0.0):.6f}")
        lines.append(f"- ?곗텧 諛⑹떇: {derivation}")
        lines.append("")
        lines.append(PROVENANCE_JSON_BEGIN)
        provenance_payload = {
            "created_at": ended_at.isoformat(timespec="seconds"),
            "output_folder_name": run_path.name,
            "class_names": class_names,
            "merged_from_names": merged_from_names,
            "components": provenance_components,
        }
        lines.append(json.dumps(provenance_payload, ensure_ascii=False, indent=2))
        lines.append(PROVENANCE_JSON_END)

        readme_path = run_path / "README.txt"
        readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logging.getLogger(__name__).info("training README written: %s", readme_path)

    def _on_training_finished(self, result_obj: object) -> None:
        self.training_waiting_epoch_start = False
        self._end_training_loading()
        save_dir = ""
        stopped = False
        skip_rename = False
        final_metric: Mapping[str, Any] = {}
        retrain_summary: Mapping[str, Any] = {}
        if isinstance(result_obj, Mapping):
            save_dir = str(result_obj.get("save_dir", ""))
            stopped = bool(result_obj.get("stopped", False))
            skip_rename = bool(result_obj.get("skip_rename", False))
            metric_obj = result_obj.get("final_metric", {})
            if isinstance(metric_obj, Mapping):
                final_metric = metric_obj
            retrain_obj = result_obj.get("retrain_summary", {})
            if isinstance(retrain_obj, Mapping):
                retrain_summary = retrain_obj
        if self.progressTrainTotal is not None:
            self.progressTrainTotal.setValue(100 if not stopped else self.progressTrainTotal.value())
        snapshot_total_epochs = int(
            self.training_args_snapshot.get(
                "epochs",
                self.spinTrainEpochs.value() if self.spinTrainEpochs is not None else 1,
            )
        )
        total_epochs = max(1, int(float(final_metric.get("total_epochs", snapshot_total_epochs))))
        if stopped:
            epoch_now = int(float(final_metric.get("epoch", 0)))
            percent_now = int(self.progressTrainTotal.value()) if self.progressTrainTotal is not None else 0
        else:
            epoch_now = total_epochs
            percent_now = 100
        self._set_training_progress_summary(percent_now, epoch_now, total_epochs)
        renamed_path = save_dir
        if (not stopped) and save_dir:
            save_dir_path = Path(save_dir).resolve()
            metric_summary = extract_training_metrics_and_losses(save_dir_path)
            if float(metric_summary.get("overall_map50", 0.0)) <= 0.0:
                metric_summary["overall_map50"] = _metric_to_unit_interval(float(final_metric.get("map50", final_metric.get("accuracy", 0.0))))
            if float(metric_summary.get("overall_map50_95", 0.0)) <= 0.0:
                metric_summary["overall_map50_95"] = _metric_to_unit_interval(float(final_metric.get("map50_95", 0.0)))
            if int(metric_summary.get("best_epoch", 0) or 0) <= 0:
                metric_summary["best_epoch"] = int(float(final_metric.get("epoch", total_epochs)))

            renamed = save_dir_path if skip_rename else self._rename_training_run_dir(save_dir_path, metric_summary)
            renamed_path = str(renamed)
            if renamed_path != str(save_dir_path):
                self._append_train_log(f"run folder renamed: {save_dir_path.name} -> {Path(renamed_path).name}")
            try:
                self._write_training_readme(
                    run_dir=Path(renamed_path),
                    metrics=metric_summary,
                    ended_at=datetime.now(),
                )
            except Exception as exc:
                self._append_train_log(f"warning: failed to write training README ({exc})")
            self._append_train_log(
                "training metric summary: "
                f"mAP50={float(metric_summary.get('overall_map50', 0.0)):.3f}, "
                f"mAP50-95={float(metric_summary.get('overall_map50_95', 0.0)):.3f}, "
                f"best_epoch={int(metric_summary.get('best_epoch', 0) or 0)}"
            )
            if retrain_summary:
                stage1 = retrain_summary.get("stage1", {})
                stage2 = retrain_summary.get("stage2", {})
                if isinstance(stage1, Mapping) and isinstance(stage2, Mapping):
                    stage1_new = stage1.get("metrics_new", {})
                    stage1_old = stage1.get("metrics_old", {})
                    stage2_new = stage2.get("metrics_new", {})
                    stage2_old = stage2.get("metrics_old", {})
                    self._append_train_log(
                        "retrain validation summary: "
                        f"stage1 new mAP50={float(stage1_new.get('map50', 0.0)):.3f}, "
                        f"stage1 old mAP50={float(stage1_old.get('map50', 0.0)):.3f}, "
                        f"stage2 new mAP50={float(stage2_new.get('map50', 0.0)):.3f}, "
                        f"stage2 old mAP50={float(stage2_old.get('map50', 0.0)):.3f}"
                    )
        if stopped:
            self._append_train_log("finish (stopped)")
        else:
            self._append_train_log(f"finish training: save_dir={renamed_path}")
        if renamed_path:
            dialog_title = "Training Stopped" if stopped else "Training Finished"
            dialog_body = "Saved result path" if stopped else "Training output path"
            QMessageBox.information(self, dialog_title, f"{dialog_body}:\n{renamed_path}")

    def _on_training_failed(self, message: str) -> None:
        self.training_waiting_epoch_start = False
        self._end_training_loading()
        self._append_train_log(f"error: training failed ({message})")
        QMessageBox.critical(self, "Training Error", f"Training failed.\n{message}")

    def _on_training_thread_finished(self) -> None:
        self.is_training = False
        self.training_merge_active = False
        self.training_worker = None
        self.training_thread = None
        self.training_waiting_epoch_start = False
        self.training_started_at = None
        self.training_args_snapshot = {}
        self._pending_training_request = None
        self._end_training_loading()
        self._update_training_ui_state()
        self._update_button_state()


