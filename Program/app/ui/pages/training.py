from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Feature 1 & 2 위젯 추가를 위한 임포트 (QCheckBox, QPushButton 이미 포함)


def _build_training_page_page(
    self: Any,
    *,
    metric_chart_widget_cls: type[Any],
    train_default_freeze: int,
    train_default_lr0: float,
    train_replay_ratio_default: float,
    train_stage1_epochs_default: int,
    train_stage2_epochs_default: int,
    train_stage2_lr_factor_default: float,
    train_retrain_seed_default: int,
    train_stage_unfreeze_neck_only: str,
    train_stage_unfreeze_backbone_last: str,
) -> QWidget:
    """Build the training page UI."""
    page = QWidget(self)
    root_layout = QHBoxLayout(page)
    root_layout.setContentsMargins(0, 0, 0, 0)
    root_layout.setSpacing(14)

    left_col = QWidget(page)
    left_layout = QVBoxLayout(left_col)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(14)

    summary = QFrame(left_col)
    summary.setProperty("pageCard", True)
    summary_layout = QVBoxLayout(summary)
    summary_layout.setContentsMargins(18, 16, 18, 16)
    summary_layout.setSpacing(8)

    title = QLabel("Model Training", summary)
    title.setObjectName("labelPageCardTitle")
    summary_layout.addWidget(title)

    mode_row = QHBoxLayout()
    mode_row.setSpacing(8)
    mode_label = QLabel("Mode", summary)
    self.checkTrainNew = QCheckBox("New Train", summary)
    self.checkTrainNew.setObjectName("checkTrainNew")
    self.checkTrainRetrain = QCheckBox("Retrain", summary)
    self.checkTrainRetrain.setObjectName("checkTrainRetrain")
    self.checkTrainNew.setChecked(True)
    self.checkTrainNew.toggled.connect(self._on_training_mode_checkbox_toggled)
    self.checkTrainRetrain.toggled.connect(self._on_training_mode_checkbox_toggled)
    mode_row.addWidget(mode_label, 0)
    mode_row.addWidget(self.checkTrainNew, 0)
    mode_row.addWidget(self.checkTrainRetrain, 0)
    mode_row.addStretch(1)
    summary_layout.addLayout(mode_row)

    self.labelTrainDataset = QLabel("dataset: not selected", summary)
    summary_layout.addWidget(self.labelTrainDataset)

    path_row = QHBoxLayout()
    self.editTrainDataYaml = QLineEdit(summary)
    self.editTrainDataYaml.setReadOnly(True)
    self.editTrainDataYaml.setPlaceholderText("data.yaml path")
    self.btnTrainPickDataYaml = QPushButton("Select Dataset", summary)
    self.btnTrainPickDataYaml.clicked.connect(self._on_pick_training_data_yaml)
    path_row.addWidget(self.editTrainDataYaml, 1)
    path_row.addWidget(self.btnTrainPickDataYaml, 0)
    summary_layout.addLayout(path_row)

    self.frameTrainRetrainYamlPanel = QFrame(summary)
    retrain_yaml_layout = QVBoxLayout(self.frameTrainRetrainYamlPanel)
    retrain_yaml_layout.setContentsMargins(0, 0, 0, 0)
    retrain_yaml_layout.setSpacing(6)

    retrain_title_row = QHBoxLayout()
    retrain_title_row.setSpacing(8)
    retrain_title = QLabel("Retrain YAML List", self.frameTrainRetrainYamlPanel)
    retrain_title_row.addWidget(retrain_title, 0)
    self.labelTrainDataYamlCount = QLabel("selected 0", self.frameTrainRetrainYamlPanel)
    self.labelTrainDataYamlCount.setObjectName("labelTrainDataYamlCount")
    retrain_title_row.addWidget(self.labelTrainDataYamlCount, 0)
    retrain_title_row.addStretch(1)
    self.btnTrainAddDataYamls = QPushButton("Add YAML", self.frameTrainRetrainYamlPanel)
    self.btnTrainAddDataYamls.clicked.connect(self._on_pick_training_data_yamls)
    self.btnTrainRemoveDataYaml = QPushButton("Remove", self.frameTrainRetrainYamlPanel)
    self.btnTrainRemoveDataYaml.clicked.connect(self._on_remove_selected_training_data_yamls)
    self.btnTrainClearDataYamls = QPushButton("Clear", self.frameTrainRetrainYamlPanel)
    self.btnTrainClearDataYamls.clicked.connect(self._on_clear_training_data_yamls)
    retrain_title_row.addWidget(self.btnTrainAddDataYamls, 0)
    retrain_title_row.addWidget(self.btnTrainRemoveDataYaml, 0)
    retrain_title_row.addWidget(self.btnTrainClearDataYamls, 0)
    retrain_yaml_layout.addLayout(retrain_title_row)

    self.listTrainDataYamls = QListWidget(self.frameTrainRetrainYamlPanel)
    self.listTrainDataYamls.setSelectionMode(QListView.SelectionMode.ExtendedSelection)
    self.listTrainDataYamls.setMinimumHeight(84)
    self.listTrainDataYamls.setMaximumHeight(112)
    retrain_yaml_layout.addWidget(self.listTrainDataYamls, 1)
    self.frameTrainRetrainYamlPanel.hide()
    summary_layout.addWidget(self.frameTrainRetrainYamlPanel, 0)

    engine_row = QHBoxLayout()
    engine_row.setSpacing(8)
    engine_label = QLabel("Engine", summary)
    self.comboTrainEngine = QComboBox(summary)
    self.comboTrainEngine.addItem("YOLO", "yolo")
    self.comboTrainEngine.addItem("RT-DETR", "rtdetr")
    engine_row.addWidget(engine_label, 0)
    engine_row.addWidget(self.comboTrainEngine, 1)
    summary_layout.addLayout(engine_row)

    model_combo_row = QHBoxLayout()
    model_combo_row.setSpacing(8)
    model_combo_label = QLabel("Model", summary)
    self.comboTrainModel = QComboBox(summary)
    model_combo_row.addWidget(model_combo_label, 0)
    model_combo_row.addWidget(self.comboTrainModel, 1)
    summary_layout.addLayout(model_combo_row)

    option_row = QHBoxLayout()
    option_row.setSpacing(10)

    option_epoch_col = QVBoxLayout()
    option_epoch_col.setSpacing(4)
    option_imgsz_col = QVBoxLayout()
    option_imgsz_col.setSpacing(4)
    option_batch_col = QVBoxLayout()
    option_batch_col.setSpacing(4)

    epoch_text = QLabel("EPOCHS", summary)
    imgsz_text = QLabel("IMAGE SIZE", summary)
    batch_text = QLabel("BATCH", summary)

    self.spinTrainEpochs = QSpinBox(summary)
    self.spinTrainEpochs.setRange(1, 1000)
    self.spinTrainEpochs.setValue(200)

    self.spinTrainImgsz = QSpinBox(summary)
    self.spinTrainImgsz.setRange(320, 1920)
    self.spinTrainImgsz.setSingleStep(32)
    self.spinTrainImgsz.setValue(640)

    self.spinTrainBatch = QSpinBox(summary)
    self.spinTrainBatch.setRange(-1, 128)
    self.spinTrainBatch.setValue(-1)

    option_epoch_col.addWidget(epoch_text)
    option_epoch_col.addWidget(self.spinTrainEpochs)
    option_imgsz_col.addWidget(imgsz_text)
    option_imgsz_col.addWidget(self.spinTrainImgsz)
    option_batch_col.addWidget(batch_text)
    option_batch_col.addWidget(self.spinTrainBatch)

    option_row.addLayout(option_epoch_col, 1)
    option_row.addLayout(option_imgsz_col, 1)
    option_row.addLayout(option_batch_col, 1)
    option_row.addStretch(1)
    summary_layout.addLayout(option_row)

    self.frameTrainAdvancedOptions = QFrame(summary)
    advanced_layout = QHBoxLayout(self.frameTrainAdvancedOptions)
    advanced_layout.setContentsMargins(12, 8, 12, 8)
    advanced_layout.setSpacing(8)
    self.frameTrainFreezeOption = QFrame(self.frameTrainAdvancedOptions)
    freeze_layout = QHBoxLayout(self.frameTrainFreezeOption)
    freeze_layout.setContentsMargins(0, 0, 0, 0)
    freeze_layout.setSpacing(8)
    freeze_layout.addWidget(QLabel("FREEZE", self.frameTrainFreezeOption), 0)
    self.spinTrainFreeze = QSpinBox(self.frameTrainFreezeOption)
    self.spinTrainFreeze.setRange(0, 200)
    self.spinTrainFreeze.setValue(train_default_freeze)
    self.spinTrainFreeze.valueChanged.connect(self._persist_training_ui_settings)
    freeze_layout.addWidget(self.spinTrainFreeze, 0)
    advanced_layout.addWidget(QLabel("LR0", self.frameTrainAdvancedOptions), 0)
    self.spinTrainLr0 = QDoubleSpinBox(self.frameTrainAdvancedOptions)
    self.spinTrainLr0.setRange(0.000001, 1.0)
    self.spinTrainLr0.setDecimals(6)
    self.spinTrainLr0.setSingleStep(0.0005)
    self.spinTrainLr0.setValue(train_default_lr0)
    self.spinTrainLr0.valueChanged.connect(self._persist_training_ui_settings)
    advanced_layout.addWidget(self.spinTrainLr0, 0)
    advanced_layout.addStretch(1)
    summary_layout.addWidget(self.frameTrainAdvancedOptions, 0)

    self.frameTrainRetrainRecipeOptions = QFrame(summary)
    retrain_recipe_layout = QHBoxLayout(self.frameTrainRetrainRecipeOptions)
    retrain_recipe_layout.setContentsMargins(12, 8, 12, 8)
    retrain_recipe_layout.setSpacing(8)
    retrain_recipe_layout.addWidget(QLabel("REPLAY OLD", self.frameTrainRetrainRecipeOptions), 0)
    self.spinTrainReplayRatio = QDoubleSpinBox(self.frameTrainRetrainRecipeOptions)
    self.spinTrainReplayRatio.setRange(0.05, 0.30)
    self.spinTrainReplayRatio.setDecimals(2)
    self.spinTrainReplayRatio.setSingleStep(0.01)
    self.spinTrainReplayRatio.setValue(train_replay_ratio_default)
    self.spinTrainReplayRatio.valueChanged.connect(self._persist_training_ui_settings)
    retrain_recipe_layout.addWidget(self.spinTrainReplayRatio, 0)
    retrain_recipe_layout.addWidget(QLabel("STAGE1 EPOCHS", self.frameTrainRetrainRecipeOptions), 0)
    self.spinTrainStage1Epochs = QSpinBox(self.frameTrainRetrainRecipeOptions)
    self.spinTrainStage1Epochs.setRange(5, 20)
    self.spinTrainStage1Epochs.setValue(train_stage1_epochs_default)
    self.spinTrainStage1Epochs.valueChanged.connect(self._persist_training_ui_settings)
    retrain_recipe_layout.addWidget(self.spinTrainStage1Epochs, 0)
    retrain_recipe_layout.addWidget(QLabel("STAGE2 EPOCHS", self.frameTrainRetrainRecipeOptions), 0)
    self.spinTrainStage2Epochs = QSpinBox(self.frameTrainRetrainRecipeOptions)
    self.spinTrainStage2Epochs.setRange(20, 80)
    self.spinTrainStage2Epochs.setValue(train_stage2_epochs_default)
    self.spinTrainStage2Epochs.valueChanged.connect(self._persist_training_ui_settings)
    retrain_recipe_layout.addWidget(self.spinTrainStage2Epochs, 0)
    retrain_recipe_layout.addWidget(QLabel("STAGE2 LR x", self.frameTrainRetrainRecipeOptions), 0)
    self.spinTrainStage2LrFactor = QDoubleSpinBox(self.frameTrainRetrainRecipeOptions)
    self.spinTrainStage2LrFactor.setRange(0.05, 1.0)
    self.spinTrainStage2LrFactor.setDecimals(2)
    self.spinTrainStage2LrFactor.setSingleStep(0.05)
    self.spinTrainStage2LrFactor.setValue(train_stage2_lr_factor_default)
    self.spinTrainStage2LrFactor.valueChanged.connect(self._persist_training_ui_settings)
    retrain_recipe_layout.addWidget(self.spinTrainStage2LrFactor, 0)
    retrain_recipe_layout.addWidget(QLabel("UNFREEZE", self.frameTrainRetrainRecipeOptions), 0)
    self.comboTrainUnfreezeMode = QComboBox(self.frameTrainRetrainRecipeOptions)
    self.comboTrainUnfreezeMode.addItem("neck_only", train_stage_unfreeze_neck_only)
    self.comboTrainUnfreezeMode.addItem("backbone_last", train_stage_unfreeze_backbone_last)
    self.comboTrainUnfreezeMode.currentIndexChanged.connect(self._persist_training_ui_settings)
    retrain_recipe_layout.addWidget(self.comboTrainUnfreezeMode, 0)
    retrain_recipe_layout.addWidget(QLabel("SEED", self.frameTrainRetrainRecipeOptions), 0)
    self.spinTrainReplaySeed = QSpinBox(self.frameTrainRetrainRecipeOptions)
    self.spinTrainReplaySeed.setRange(0, 999999)
    self.spinTrainReplaySeed.setValue(train_retrain_seed_default)
    self.spinTrainReplaySeed.valueChanged.connect(self._persist_training_ui_settings)
    retrain_recipe_layout.addWidget(self.spinTrainReplaySeed, 0)
    retrain_recipe_layout.addStretch(1)
    summary_layout.addWidget(self.frameTrainRetrainRecipeOptions, 0)

    progress_row = QHBoxLayout()
    progress_row.setSpacing(8)
    self.progressTrainTotal = QProgressBar(summary)
    self.progressTrainTotal.setRange(0, 100)
    self.progressTrainTotal.setValue(0)
    self.progressTrainTotal.setMinimumHeight(26)
    self.progressTrainTotal.setTextVisible(False)
    progress_row.addWidget(self.progressTrainTotal, 1)
    self.labelTrainMetricNow = QLabel("progress 0%, 0 / 0", summary)
    self.labelTrainMetricNow.setWordWrap(False)
    self.labelTrainMetricNow.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
    self.labelTrainMetricNow.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    progress_row.addWidget(self.labelTrainMetricNow, 0)
    summary_layout.addLayout(progress_row)

    controls = QHBoxLayout()
    controls.setSpacing(8)
    self.btnTrainStart = QPushButton("Start Training", summary)
    self.btnTrainStart.clicked.connect(self.on_start_training)
    self.btnTrainStop = QPushButton("Stop Training", summary)
    self.btnTrainStop.clicked.connect(self.on_stop_training)
    # Feature 2: 고급 파라미터 버튼
    self.btnTrainAdvancedParams = QPushButton("고급 파라미터 ⚙", summary)
    self.btnTrainAdvancedParams.setObjectName("btnTrainAdvancedParams")
    self.btnTrainAdvancedParams.setToolTip("YOLO 학습 하이퍼파라미터를 상세 설정합니다.")
    self.btnTrainAdvancedParams.clicked.connect(self._on_open_yolo_params_dialog)
    controls.addWidget(self.btnTrainStart)
    controls.addWidget(self.btnTrainStop)
    controls.addWidget(self.btnTrainAdvancedParams)
    controls.addStretch(1)
    summary_layout.addLayout(controls)

    # Feature 1: 로컬/서버 학습 전환 행
    local_train_row = QHBoxLayout()
    local_train_row.setSpacing(8)
    self.checkTrainLocal = QCheckBox("로컬 PC에서 학습", summary)
    self.checkTrainLocal.setObjectName("checkTrainLocal")
    self.checkTrainLocal.setToolTip(
        "체크 시 원격 GPU 서버 대신 이 PC에서 직접 YOLO 학습을 실행합니다.\n"
        "CUDA GPU가 없으면 CPU로 학습하며 매우 느릴 수 있습니다."
    )
    self.checkTrainLocal.toggled.connect(self._on_train_local_toggled)
    local_train_row.addWidget(self.checkTrainLocal, 0)
    self.labelTrainLocalDevice = QLabel("", summary)
    self.labelTrainLocalDevice.setObjectName("labelTrainLocalDevice")
    local_train_row.addWidget(self.labelTrainLocalDevice, 0)
    local_train_row.addStretch(1)
    summary_layout.addLayout(local_train_row)

    left_layout.addWidget(summary, 0)

    logs = QFrame(left_col)
    logs.setProperty("pageCard", True)
    logs_layout = QVBoxLayout(logs)
    logs_layout.setContentsMargins(18, 16, 18, 16)
    logs_layout.setSpacing(8)
    logs_title = QLabel("Training Log", logs)
    logs_title.setObjectName("labelPageCardTitle")
    logs_layout.addWidget(logs_title)
    self.textTrainLog = QTextEdit(logs)
    self.textTrainLog.setReadOnly(True)
    self.textTrainLog.setPlainText("[info] training log will appear here in real time.")
    logs_layout.addWidget(self.textTrainLog)
    logs.setMinimumHeight(240)
    left_layout.addWidget(logs, 1)

    right_col = QFrame(page)
    right_col.setProperty("pageCard", True)
    right_col.setMinimumWidth(380)
    right_layout = QVBoxLayout(right_col)
    right_layout.setContentsMargins(12, 12, 12, 12)
    right_layout.setSpacing(10)
    chart_title = QLabel("Live Metrics", right_col)
    chart_title.setObjectName("labelPageCardTitle")
    chart_title.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    right_layout.addWidget(chart_title, 0, Qt.AlignmentFlag.AlignTop)

    chart_container = QWidget(right_col)
    chart_layout = QVBoxLayout(chart_container)
    chart_layout.setContentsMargins(0, 0, 0, 0)
    chart_layout.setSpacing(8)
    self.trainingBoxLossChart = metric_chart_widget_cls("box_loss", QColor(255, 173, 93), parent=chart_container)
    self.trainingClsLossChart = metric_chart_widget_cls("cls_loss", QColor(145, 206, 255), parent=chart_container)
    self.trainingDflLossChart = metric_chart_widget_cls("dfl_loss", QColor(208, 171, 255), parent=chart_container)
    self.trainingAccChart = metric_chart_widget_cls(
        "accuracy (mAP50)",
        QColor(90, 223, 183),
        y_min=0.0,
        y_max=1.0,
        parent=chart_container,
    )
    chart_layout.addWidget(self.trainingBoxLossChart, 1)
    chart_layout.addWidget(self.trainingClsLossChart, 1)
    chart_layout.addWidget(self.trainingDflLossChart, 1)
    chart_layout.addWidget(self.trainingAccChart, 1)
    right_layout.addWidget(chart_container, 1)
    right_layout.addStretch(0)

    root_layout.addWidget(left_col, 3)
    root_layout.addWidget(right_col, 1)

    self._refresh_training_model_combo()
    self._update_training_metric_chart_titles()
    self._update_training_ui_state()
    return page
