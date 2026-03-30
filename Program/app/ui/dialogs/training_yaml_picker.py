from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class TrainingYamlPickerDialog(QDialog):
    """Dialog for choosing one or more training YAML files from known roots."""

    def __init__(
        self,
        *,
        roots: Sequence[Path],
        title: str,
        multi_select: bool = True,
        path_filter: Callable[[Path], bool] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        unique_roots: list[Path] = []
        seen_roots: set[str] = set()
        for raw_root in roots:
            root = Path(raw_root).expanduser().resolve()
            key = str(root).casefold()
            if key in seen_roots:
                continue
            seen_roots.add(key)
            unique_roots.append(root)
        self.roots = unique_roots
        self.multi_select = bool(multi_select)
        self.path_filter = path_filter
        self.yaml_list = QListWidget(self)
        self.yaml_list.setSelectionMode(
            QListView.SelectionMode.ExtendedSelection if self.multi_select else QListView.SelectionMode.SingleSelection
        )

        self.setWindowTitle(str(title))
        self.resize(860, 520)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(8)

        roots_text = "\n".join(f"- {root}" for root in self.roots) if self.roots else "- (no root)"
        info = QLabel(
            (
                "The list below is built automatically from the selected roots.\n"
                "Use drag, Shift, or Ctrl to select multiple YAML files.\n"
                f"Search roots:\n{roots_text}"
            ),
            self,
        )
        info.setWordWrap(True)
        root_layout.addWidget(info)
        root_layout.addWidget(self.yaml_list, 1)

        button_row = QHBoxLayout()
        button_row.setSpacing(6)
        btn_refresh = QPushButton("Refresh", self)
        btn_refresh.clicked.connect(self.refresh_yaml_list)
        button_row.addWidget(btn_refresh, 0)

        if self.multi_select:
            btn_select_all = QPushButton("Select All", self)
            btn_select_all.clicked.connect(self._select_all)
            button_row.addWidget(btn_select_all, 0)

        btn_clear = QPushButton("Clear", self)
        btn_clear.clicked.connect(self.yaml_list.clearSelection)
        button_row.addWidget(btn_clear, 0)
        button_row.addStretch(1)

        btn_confirm = QPushButton("Confirm", self)
        btn_confirm.clicked.connect(self._accept_if_selected)
        btn_cancel = QPushButton("Cancel", self)
        btn_cancel.clicked.connect(self.reject)
        button_row.addWidget(btn_confirm, 0)
        button_row.addWidget(btn_cancel, 0)
        root_layout.addLayout(button_row)

        self.refresh_yaml_list()

    def refresh_yaml_list(self) -> None:
        self.yaml_list.clear()
        files = self._scan_yaml_files()
        if not files:
            empty_item = QListWidgetItem("No YAML file found.")
            empty_item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.yaml_list.addItem(empty_item)
            return

        for yaml_path in files:
            item = QListWidgetItem(self._display_name(yaml_path))
            item.setToolTip(str(yaml_path))
            item.setData(Qt.ItemDataRole.UserRole, str(yaml_path))
            self.yaml_list.addItem(item)

    def selected_paths(self) -> list[Path]:
        paths: list[Path] = []
        for item in self.yaml_list.selectedItems():
            raw = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
            if not raw:
                continue
            path = Path(raw).resolve()
            if path.is_file():
                paths.append(path)
        return paths

    def _scan_yaml_files(self) -> list[Path]:
        collected: list[Path] = []
        seen: set[str] = set()
        for root in self.roots:
            if not root.is_dir():
                continue
            for pattern in ("*.yaml", "*.yml"):
                for path in root.rglob(pattern):
                    if not path.is_file():
                        continue
                    resolved = path.resolve()
                    if self.path_filter is not None:
                        try:
                            if not bool(self.path_filter(resolved)):
                                continue
                        except Exception:
                            continue
                    key = str(resolved).casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    collected.append(resolved)
        collected.sort(key=lambda item: str(item).casefold())
        return collected

    def _display_name(self, yaml_path: Path) -> str:
        for root in self.roots:
            try:
                relative = yaml_path.relative_to(root).as_posix()
                return f"[{root.name}] {relative}"
            except Exception:
                continue
        return str(yaml_path)

    def _select_all(self) -> None:
        for index in range(self.yaml_list.count()):
            item = self.yaml_list.item(index)
            if item is None:
                continue
            if not bool(item.flags() & Qt.ItemFlag.ItemIsSelectable):
                continue
            item.setSelected(True)

    def _accept_if_selected(self) -> None:
        if not self.selected_paths():
            QMessageBox.warning(self, "YAML Selection", "Select at least one YAML file.")
            return
        self.accept()
