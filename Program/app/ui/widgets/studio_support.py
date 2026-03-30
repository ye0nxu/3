from __future__ import annotations

import math
from typing import Any

import numpy as np
from PyQt6.QtCore import QPoint, QRect, QSettings, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QLabel, QSizePolicy, QWidget

class HoverPreviewLabel(QLabel):
    """마우스 진입/이탈 시그널을 발생시켜 호버 미리보기를 제어하는 라벨 위젯입니다."""
    hover_entered = pyqtSignal()
    hover_left = pyqtSignal()

    def enterEvent(self, event) -> None:  # type: ignore[override]
        """마우스 진입 이벤트를 받아 hover_entered 시그널을 발생시키고 기본 동작을 호출합니다."""
        self.hover_entered.emit()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        """마우스 이탈 이벤트를 받아 hover_left 시그널을 발생시키고 기본 동작을 호출합니다."""
        self.hover_left.emit()
        super().leaveEvent(event)

class RoiSelectionLabel(QLabel):
    """마우스 드래그로 ROI 사각형을 선택할 수 있는 라벨 위젯입니다."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dragging = False
        self._drag_start = QPoint()
        self._drag_current = QPoint()
        self._selection_rect: QRect | None = None
        self.setMouseTracking(True)

    def set_selection_rect(self, rect: QRect | None) -> None:
        """외부에서 지정한 ROI 사각형을 현재 선택 상태로 반영합니다."""
        if rect is None:
            self._selection_rect = None
        else:
            normalized = QRect(rect).normalized()
            if normalized.width() >= 2 and normalized.height() >= 2:
                self._selection_rect = self._clamp_rect(normalized)
            else:
                self._selection_rect = None
        self.update()

    def selected_rect(self) -> QRect | None:
        """현재 선택된 ROI 사각형을 반환합니다."""
        if self._selection_rect is None:
            return None
        return QRect(self._selection_rect)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        """드래그 시작 지점을 기록해 ROI 선택을 시작합니다."""
        if event.button() == Qt.MouseButton.LeftButton:
            point = self._clamp_point(event.position().toPoint())
            self._drag_start = point
            self._drag_current = point
            self._dragging = True
            self._selection_rect = QRect(point, point)
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        """드래그 중 현재 포인터 위치를 반영해 ROI 사각형을 갱신합니다."""
        if self._dragging:
            self._drag_current = self._clamp_point(event.position().toPoint())
            self._selection_rect = self._clamp_rect(QRect(self._drag_start, self._drag_current).normalized())
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        """드래그 종료 시 ROI 사각형을 확정하고 최소 크기 미만이면 해제합니다."""
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._drag_current = self._clamp_point(event.position().toPoint())
            rect = self._clamp_rect(QRect(self._drag_start, self._drag_current).normalized())
            if rect.width() < 2 or rect.height() < 2:
                self._selection_rect = None
            else:
                self._selection_rect = rect
            self._dragging = False
            self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        """현재 선택된 ROI 사각형을 반투명 오버레이와 외곽선으로 렌더링합니다."""
        super().paintEvent(event)
        if self._selection_rect is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRect(self._selection_rect)

        shade = QColor(6, 10, 18, 90)
        full = self.rect()
        painter.fillRect(QRect(full.left(), full.top(), full.width(), max(0, rect.top() - full.top())), shade)
        painter.fillRect(QRect(full.left(), rect.bottom(), full.width(), max(0, full.bottom() - rect.bottom())), shade)
        painter.fillRect(QRect(full.left(), rect.top(), max(0, rect.left() - full.left()), rect.height()), shade)
        painter.fillRect(QRect(rect.right(), rect.top(), max(0, full.right() - rect.right()), rect.height()), shade)

        pen = QPen(QColor(86, 226, 255, 230), 2)
        painter.setPen(pen)
        painter.drawRect(rect)

    def _clamp_point(self, point: QPoint) -> QPoint:
        """라벨 범위를 벗어나지 않도록 좌표를 보정해 반환합니다."""
        x = max(0, min(point.x(), max(0, self.width() - 1)))
        y = max(0, min(point.y(), max(0, self.height() - 1)))
        return QPoint(x, y)

    def _clamp_rect(self, rect: QRect) -> QRect:
        """라벨 범위로 ROI 사각형을 보정해 반환합니다."""
        left = max(0, min(rect.left(), max(0, self.width() - 1)))
        top = max(0, min(rect.top(), max(0, self.height() - 1)))
        right = max(0, min(rect.right(), max(0, self.width() - 1)))
        bottom = max(0, min(rect.bottom(), max(0, self.height() - 1)))
        return QRect(QPoint(left, top), QPoint(right, bottom)).normalized()

class FuturisticSpinner(QWidget):
    """원형 궤도와 네온 트레일을 회전시켜 로딩 진행 중 상태를 직관적으로 표시하는 커스텀 스피너입니다."""

    def __init__(self, parent: QWidget | None = None, size: int = 76) -> None:
        """스피너 크기와 애니메이션 타이머를 초기화합니다."""
        super().__init__(parent)
        side = max(44, int(size))
        self.setFixedSize(side, side)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._angle = 0.0
        self._pulse_phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._on_tick)

    def start(self) -> None:
        """스피너 회전 애니메이션을 시작합니다."""
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        """스피너 회전 애니메이션을 정지합니다."""
        if self._timer.isActive():
            self._timer.stop()

    def _on_tick(self) -> None:
        """한 프레임씩 회전 각도와 펄스 위상을 갱신한 뒤 다시 그리기를 요청합니다."""
        self._angle = (self._angle + 7.0) % 360.0
        self._pulse_phase = (self._pulse_phase + 0.08) % 1.0
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        """원형 베이스 링, 회전 트레일, 중앙 펄스 코어를 렌더링합니다."""
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        side = min(self.width(), self.height())
        margin = max(5, side // 12)
        diameter = max(10, side - (2 * margin))
        x = (self.width() - diameter) // 2
        y = (self.height() - diameter) // 2

        base_pen = QPen(QColor(120, 150, 190, 60), max(2, side // 18))
        base_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(base_pen)
        painter.drawArc(x, y, diameter, diameter, 0, 360 * 16)

        segments = 16
        span_deg = 22.0
        line_width = max(3, side // 17)
        for idx in range(segments):
            trail_ratio = 1.0 - (idx / float(segments))
            alpha = int(30 + 225 * (trail_ratio**1.9))
            red = int(35 + 25 * (1.0 - trail_ratio))
            green = int(155 + 70 * trail_ratio)
            blue = int(255 - 50 * (1.0 - trail_ratio))
            pen = QPen(QColor(red, green, blue, alpha), line_width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)

            angle = self._angle - (idx * (360.0 / float(segments)))
            start = int((90.0 - angle - (span_deg * 0.5)) * 16.0)
            span = int(span_deg * 16.0)
            painter.drawArc(x, y, diameter, diameter, start, span)

        pulse_ratio = 0.5 + (0.5 * math.sin(self._pulse_phase * math.tau))
        core_radius = max(4, int((diameter * 0.12) + (diameter * 0.045 * pulse_ratio)))
        cx = self.width() // 2
        cy = self.height() // 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(72, 219, 251, int(130 + (90 * pulse_ratio))))
        painter.drawEllipse(cx - core_radius, cy - core_radius, core_radius * 2, core_radius * 2)

class MetricChartWidget(QWidget):
    """단일 지표(epoch-x / metric-y)를 축과 그리드가 보이도록 렌더링하는 위젯입니다."""

    def __init__(
        self,
        title: str,
        line_color: QColor,
        y_min: float | None = None,
        y_max: float | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.title = str(title)
        self.line_color = QColor(line_color)
        self.fixed_y_min = y_min
        self.fixed_y_max = y_max
        self._points: list[tuple[int, float]] = []

    def _is_light_theme(self) -> bool:
        theme = str(QSettings("SL_TEAM", "AutoLabelStudio").value("theme", "dark")).strip().lower()
        return theme == "light"

    def reset(self) -> None:
        self._points = []
        self.update()

    def set_title(self, title: str) -> None:
        self.title = str(title)
        self.update()

    def add_point(self, epoch: int, value: float | None) -> None:
        if value is None:
            return
        ep = max(1, int(epoch))
        val = float(value)
        if not np.isfinite(val):
            return
        self._points.append((ep, val))
        if len(self._points) > 400:
            self._points = self._points[-400:]
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        is_light = self._is_light_theme()
        outer_fill = QColor("#FFFFFF") if is_light else QColor(14, 20, 34, 34)
        chart_fill = QColor("#FFFFFF") if is_light else QColor(8, 14, 25, 18)
        border_color = QColor("#D7DFEA") if is_light else QColor(125, 142, 168, 130)
        title_color = QColor("#0F172A") if is_light else QColor(180, 196, 221, 200)
        grid_color = QColor("#CBD5E1") if is_light else QColor(140, 152, 176, 90)
        axis_color = QColor("#94A3B8") if is_light else QColor(180, 190, 212, 140)
        empty_color = QColor("#64748B") if is_light else QColor(145, 160, 185, 180)
        tick_color = QColor("#64748B") if is_light else QColor(185, 195, 220, 185)
        footer_color = QColor("#64748B") if is_light else QColor(170, 180, 200, 180)

        outer = self.rect().adjusted(6, 6, -6, -6)
        painter.fillRect(outer, outer_fill)
        painter.setPen(QPen(border_color, 1))
        painter.drawRect(outer)

        chart = outer.adjusted(42, 18, -10, -22)
        if chart.width() <= 10 or chart.height() <= 10:
            return

        painter.fillRect(chart, chart_fill)
        painter.setPen(QPen(title_color, 1))
        painter.drawText(outer.left() + 8, outer.top() + 13, self.title)

        painter.setPen(QPen(grid_color, 1))
        for i in range(1, 4):
            y = chart.top() + int((i / 4.0) * chart.height())
            painter.drawLine(chart.left(), y, chart.right(), y)
        for i in range(1, 5):
            x = chart.left() + int((i / 5.0) * chart.width())
            painter.drawLine(x, chart.top(), x, chart.bottom())

        painter.setPen(QPen(axis_color, 1))
        painter.drawLine(chart.left(), chart.bottom(), chart.right(), chart.bottom())
        painter.drawLine(chart.left(), chart.top(), chart.left(), chart.bottom())

        if len(self._points) < 1:
            painter.setPen(QPen(empty_color, 1))
            painter.drawText(chart, int(Qt.AlignmentFlag.AlignCenter), "데이터 없음")
            return

        epochs = [p[0] for p in self._points]
        vals = [p[1] for p in self._points]
        ep_min, ep_max = min(epochs), max(epochs)
        if ep_max <= ep_min:
            ep_max = ep_min + 1

        if self.fixed_y_min is None:
            y_min = float(min(vals))
        else:
            y_min = float(self.fixed_y_min)
        if self.fixed_y_max is None:
            y_max = float(max(vals))
        else:
            y_max = float(self.fixed_y_max)
        if y_max <= y_min:
            y_max = y_min + 1e-6

        def _x(ep: float) -> float:
            return chart.left() + ((ep - ep_min) / float(ep_max - ep_min)) * chart.width()

        def _y(v: float) -> float:
            ratio = (v - y_min) / float(y_max - y_min)
            return chart.bottom() - (ratio * chart.height())

        poly: list[QPoint] = []
        for ep, val in self._points:
            poly.append(QPoint(int(round(_x(ep))), int(round(_y(val)))))

        painter.setPen(QPen(self.line_color, 2))
        for idx in range(1, len(poly)):
            painter.drawLine(poly[idx - 1], poly[idx])

        painter.setPen(QPen(tick_color, 1))
        y_ticks = 4
        for i in range(0, y_ticks + 1):
            ratio = i / float(y_ticks)
            y_px = chart.bottom() - int(round(ratio * chart.height()))
            y_val = y_min + (ratio * (y_max - y_min))
            painter.drawText(outer.left() + 2, y_px + 4, f"{y_val:.3f}")

        x_ticks = 4
        for i in range(0, x_ticks + 1):
            ratio = i / float(x_ticks)
            x_px = chart.left() + int(round(ratio * chart.width()))
            ep_val = int(round(ep_min + (ratio * (ep_max - ep_min))))
            painter.drawText(x_px - 10, chart.bottom() + 15, str(ep_val))

        painter.setPen(QPen(footer_color, 1))
        painter.drawText(outer.left() + 8, outer.bottom() - 6, f"x(epoch): {ep_min}..{ep_max}")
        painter.drawText(outer.right() - 150, outer.bottom() - 6, f"y: {y_min:.4f}..{y_max:.4f}")
