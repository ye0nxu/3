from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

from core.dataset import BoxAnnotation
from app.studio.runtime import cv2

def _show_frame_ops(self, frame: np.ndarray, boxes: Sequence[Any]) -> None:
    """대상 프레임 또는 대화상자를 사용자 화면에 표시합니다."""
    if self.labelVideoPlaceholder is None:
        return

    draw_frame = self._draw_boxes(frame, boxes)
    self.last_preview_frame = draw_frame
    pixmap = self._frame_to_pixmap(draw_frame)
    self.labelVideoPlaceholder.setPixmap(pixmap)
    self.labelVideoPlaceholder.setText("")

def _draw_boxes_ops(self, frame: np.ndarray, boxes: Sequence[Any]) -> np.ndarray:
    """프레임 이미지 위에 박스/폴리곤/클래스 라벨을 오버레이해 시각화 결과를 반환합니다."""
    canvas = frame.copy()
    for box in boxes:
        xyxy = self._extract_box_xyxy(box)
        if xyxy is None:
            continue
        x1, y1, x2, y2, class_id, track_id = xyxy
        status = self._extract_box_status(box)
        polygon = self._extract_polygon_points(box)
        color = self._box_color_for_status(status)
        # box dict에 class_name이 직접 담겨 있으면 우선 사용 (model test 경로).
        if isinstance(box, dict) and str(box.get("class_name", "")).strip():
            class_name = str(box["class_name"]).strip()
        else:
            class_name = self._class_name(class_id)
        label = class_name
        if track_id is not None:
            label = f"{label}#{track_id}"
        if status in {"keep", "hold", "drop"}:
            label = f"{label} ({status})"
        if isinstance(box, dict):
            drop_reason = str(box.get("drop_reason", "")).strip().lower()
            if status == "drop" and drop_reason:
                label = f"{label}/{drop_reason}"

        draw_polygon = polygon if polygon is not None and len(polygon) >= 3 else [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        ]
        pts = np.array(draw_polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 255, 240),
            2,
            cv2.LINE_AA,
        )
    return canvas

def _extract_box_status_ops(self, box: Any) -> str | None:
    """원본 데이터 구조에서 필요한 필드만 추출해 표준 형식으로 반환합니다."""
    if isinstance(box, dict):
        status = str(box.get("status", "")).strip().lower()
        if status:
            return status
    return None

def _extract_polygon_points_ops(self, box: Any) -> list[tuple[int, int]] | None:
    """원본 데이터 구조에서 필요한 필드만 추출해 표준 형식으로 반환"""
    if not isinstance(box, dict):
        return None
    raw = box.get("polygon")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None
    points: list[tuple[int, int]] = []
    for pt in raw:
        try:
            x = int(round(float(pt[0])))
            y = int(round(float(pt[1])))
        except Exception:
            continue
        points.append((x, y))
    if len(points) < 3:
        return None
    return points

def _box_color_for_status_ops(self, status: str | None) -> tuple[int, int, int]:
    """박스 상태(keep/hold/drop 등)에 대응하는 표시 색상을 반환"""
    key = str(status or "").strip().lower()
    if key == "keep":
        return (80, 240, 120)
    if key == "hold":
        return (0, 165, 255)
    if key == "drop":
        return (50, 50, 255)
    return (120, 220, 240)

def _extract_box_xyxy_ops(self, box: Any) -> tuple[int, int, int, int, int, int | None] | None:
    """원본 데이터 구조에서 필요한 필드만 추출해 표준 형식으로 반환"""
    if isinstance(box, BoxAnnotation):
        return (
            int(round(box.x1)),
            int(round(box.y1)),
            int(round(box.x2)),
            int(round(box.y2)),
            int(box.class_id),
            int(box.track_id) if box.track_id is not None else None,
        )

    if isinstance(box, dict):
        class_id = int(box.get("class_id", box.get("cls", 0)))
        track_id = box.get("track_id")
        if {"x1", "y1", "x2", "y2"}.issubset(box):
            return (
                int(round(float(box["x1"]))),
                int(round(float(box["y1"]))),
                int(round(float(box["x2"]))),
                int(round(float(box["y2"]))),
                class_id,
                int(track_id) if track_id is not None else None,
            )
    return None

def _class_name_ops(self, class_id: int) -> str:
    """클래스 ID를 현재 결과/활성 클래스 목록에서 사용자 표시용 이름으로 변환해 반환"""
    if not self.active_class_names:
        return f"클래스_{class_id}"
    if 0 <= class_id < len(self.active_class_names):
        return str(self.active_class_names[class_id]).strip()
    return f"클래스_{class_id}"

def _frame_to_pixmap_ops(
    self,
    frame: np.ndarray,
    max_width: int | None = None,
    max_height: int | None = None,
) -> QPixmap:
    """OpenCV BGR 프레임을 Qt Pixmap으로 변환하고 최대 크기에 맞춰 스케일링"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    qimage = QImage(
        rgb.data,
        width,
        height,
        width * 3,
        QImage.Format.Format_RGB888,
    ).copy()
    pixmap = QPixmap.fromImage(qimage)

    if max_width is not None and max_height is not None:
        return pixmap.scaled(
            max_width,
            max_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    if self.labelVideoPlaceholder is not None:
        target_size = self.labelVideoPlaceholder.size()
        if target_size.width() > 0 and target_size.height() > 0:
            pixmap = pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
    return pixmap

def _read_video_frame_raw_ops(self, frame_index: int) -> np.ndarray | None:
    """원본 VideoCapture에서 지정 프레임을 읽어 그대로 반환합니다."""
    if self.video_cap is None:
        return None
    target_index = max(0, int(frame_index))
    if (
        self._video_frame_cache_index == target_index
        and isinstance(self._video_frame_cache_frame, np.ndarray)
    ):
        return self._video_frame_cache_frame
    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
    ok, frame = self.video_cap.read()
    if not ok or frame is None:
        return None
    self._video_frame_cache_index = target_index
    self._video_frame_cache_frame = frame
    return frame

def _read_video_frame_ops(self, frame_index: int) -> np.ndarray | None:
    """지정 프레임을 읽고 현재 선택 ROI를 적용한 화면/처리용 프레임을 반환합니다."""
    frame = self._read_video_frame_raw(frame_index)
    if frame is None:
        return None
    return self._apply_video_roi_to_frame(frame)

def _ensure_preview_image_cap_ops(self) -> cv2.VideoCapture | None:
    """미리보기 팝업용 전용 VideoCapture를 준비해 반환합니다."""
    if self.video_path is None:
        return None
    target_path = str(self.video_path)
    if (
        self.preview_image_cap is not None
        and self.preview_image_cap.isOpened()
        and self.preview_image_cap_path == target_path
    ):
        return self.preview_image_cap
    self._release_preview_image_cap()
    cap = cv2.VideoCapture(target_path)
    if not cap.isOpened():
        cap.release()
        return None
    self.preview_image_cap = cap
    self.preview_image_cap_path = target_path
    return cap

def _release_preview_image_cap_ops(self) -> None:
    """미리보기 팝업 전용 VideoCapture를 해제합니다."""
    if self.preview_image_cap is not None:
        self.preview_image_cap.release()
        self.preview_image_cap = None
    self.preview_image_cap_path = ""
    self._clear_thumbnail_source_cache()

def _clear_video_frame_cache_ops(self) -> None:
    """랜덤 접근용 최근 프레임 캐시를 초기화합니다."""
    self._video_frame_cache_index = -1
    self._video_frame_cache_frame = None

def _clear_thumbnail_source_cache_ops(self) -> None:
    """썸네일/팝업 원본 프레임 캐시를 비웁니다."""
    self._thumb_source_cache.clear()

def _remember_thumbnail_source_cache_ops(self, cache_key: tuple[int, str], frame: np.ndarray) -> None:
    """썸네일 소스 프레임을 LRU 캐시에 저장하고 용량 상한을 유지합니다."""
    self._thumb_source_cache[cache_key] = frame
    self._thumb_source_cache.move_to_end(cache_key, last=True)
    while len(self._thumb_source_cache) > max(8, int(self._thumb_source_cache_limit)):
        self._thumb_source_cache.popitem(last=False)

def _update_time_label_ops(self, frame_index: int) -> None:
    """현재 데이터와 상태를 기준으로 UI 표시값 또는 내부 상태를 동기화"""
    if self.labelTime is None:
        return
    fps = float(self.video_meta.get("fps", 0.0))
    total_frames = int(self.video_meta.get("frame_count", 0))
    if fps <= 0 or total_frames <= 0:
        self.labelTime.setText("00:00:00.0")
        return

    current_sec = max(0.0, float(frame_index) / fps)
    total_sec = max(0.0, float(total_frames - 1) / fps)
    self.labelTime.setText(f"{self._format_hms(current_sec)} / {self._format_hms(total_sec)}")

def _format_hms_ops(self, sec: float) -> str:
    """내부 값을 사용자 표시용 문자열 형식으로 변환해 반환"""
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60.0
    return f"{h:02d}:{m:02d}:{s:04.1f}"

def _format_eta_ops(self, sec: float) -> str:
    """내부 값을 사용자 표시용 문자열 형식으로 변환해 반환"""
    total_sec = max(0, int(round(sec)))
    m, s = divmod(total_sec, 60)
    return f"{m:02d}:{s:02d}"
