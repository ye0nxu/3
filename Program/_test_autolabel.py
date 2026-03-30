"""
TrackObject CONF_LOW + IOU ID 매칭 수정 후 전체 파이프라인 재현 테스트
"""
import sys, os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

MODEL_PATH  = Path("C:/Users/jhjh5/KDT_10_SL/99.CV_project/YOLO_trained_tail_light_m50(0.324)_m50-95(0.145)/weights/best.pt")
VIDEO_PATH  = Path("C:/Users/jhjh5/KDT_10_SL/99.CV_project/LIM_PART/video/Car_Lamp/lamp_test1.mp4")
CLASS_NAMES = ["tail_light"]
YOLO_CONF   = 0.10
YOLO_IOU    = 0.45
YOLO_IMGSZ  = 320
MAX_FRAMES  = 100
BORDER_MARGIN = 5

def norm_cls(name):
    return str(name).strip().casefold().replace("_", " ")

def assign_ids_by_iou(boxes, track_history, frame_idx, iou_threshold=0.3, max_gap=5):
    next_id = max(track_history.keys(), default=0) + 1
    assigned, used = [], set()
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        best_id, best_iou = None, iou_threshold
        for tid, tracker in track_history.items():
            if tid in used: continue
            if frame_idx - int(getattr(tracker, "last_seen_frame", -999)) > max_gap: continue
            buf = getattr(tracker, "buffer", [])
            lb = buf[-1].get("bbox") if buf and isinstance(buf[-1], dict) else None
            if lb is None: continue
            ix1, iy1 = max(x1, lb[0]), max(y1, lb[1])
            ix2, iy2 = min(x2, lb[2]), min(y2, lb[3])
            inter = max(0, ix2-ix1)*max(0, iy2-iy1)
            if inter <= 0: continue
            iou = inter / max((x2-x1)*(y2-y1)+(lb[2]-lb[0])*(lb[3]-lb[1])-inter, 1e-6)
            if iou > best_iou:
                best_iou, best_id = iou, tid
        if best_id is None:
            best_id = next_id; next_id += 1
        used.add(best_id); assigned.append(best_id)
    return assigned

def main():
    from app.studio.runtime import UltralyticsYOLO, cv2
    from core.tracking import TrackObject

    model = UltralyticsYOLO(str(MODEL_PATH))
    model_names = getattr(model, "names", {})
    print(f"모델 클래스: {model_names}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    allowed_keys = {norm_cls(n): n for n in CLASS_NAMES}
    all_class_mode = not CLASS_NAMES
    track_history = {}

    keep_cnt = hold_cnt = drop_cnt = border_cnt = 0

    for frame_idx in range(MAX_FRAMES):
        ok, frame = cap.read()
        if not ok: break

        results = model.track(frame, persist=True, verbose=False,
                              tracker="bytetrack.yaml",
                              conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ)

        if not results or results[0].boxes is None or len(results[0].boxes.xyxy) == 0:
            continue

        boxes   = results[0].boxes.xyxy.cpu().numpy()
        confs   = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
        else:
            ids = assign_ids_by_iou(boxes, track_history, frame_idx)

        for i, track_id in enumerate(ids):
            x1, y1, x2, y2 = [float(v) for v in boxes[i]]
            score    = float(confs[i])
            cls_name = str(model_names.get(int(classes[i]), "?"))

            if not all_class_mode and allowed_keys.get(norm_cls(cls_name)) is None:
                continue
            if x1 < BORDER_MARGIN or y1 < BORDER_MARGIN or x2 > w-BORDER_MARGIN or y2 > h-BORDER_MARGIN:
                border_cnt += 1; continue

            tracker = track_history.get(int(track_id))
            if tracker is None:
                tracker = TrackObject(int(track_id), frame_idx, conf_low=YOLO_CONF)
                track_history[int(track_id)] = tracker

            bbox = [x1, y1, x2, y2]
            action, reason = tracker.process(bbox, score, None, frame_idx, (h, w))

            if action in {"KEEP", "KEEP_BUFFER"}:
                keep_cnt += 1
            elif action in {"HOLD", "HOLD_BUFFER"}:
                hold_cnt += 1
            elif action in {"DROP"}:
                drop_cnt += 1
            # BUFFERING, SKIP은 카운트 안 함

    cap.release()

    print(f"\n========== {MAX_FRAMES}프레임 결과 ==========")
    print(f"BORDER 스킵  : {border_cnt}")
    print(f"KEEP         : {keep_cnt}")
    print(f"HOLD         : {hold_cnt}")
    print(f"DROP         : {drop_cnt}")
    if keep_cnt + hold_cnt > 0:
        print("\n라벨링 저장 정상 동작합니다.")
    else:
        print("\n[경고] KEEP/HOLD 없음 - 저장 불가")

if __name__ == "__main__":
    main()
