# text_detection/__init__.py
import cv2
import numpy as np
from pathlib import Path
from paddleocr import PaddleOCR

_DET = None

def get_detector(
    text_detection_model_name="PP-OCRv4_server_det",
    text_det_thresh=0.3,
    text_det_box_thresh=0.3,
    text_det_unclip_ratio=None,
    use_doc_unwarping=False,
    lang="en",
):
    """
    Create or return a cached PaddleOCR detector instance (detection-only).
    """
    global _DET
    if _DET is None:
        _DET = PaddleOCR(
            text_detection_model_name=text_detection_model_name,
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
            text_det_unclip_ratio=text_det_unclip_ratio,
            use_doc_unwarping=use_doc_unwarping,
            use_doc_orientation_classify=False,
            use_textline_orientation=False,
            lang=lang
        )
    return _DET

def _crop_region(img, box):
    """
    Crop axis-aligned bounding rectangle using the 4-point box.
    """
    pts = np.array(box).astype("int32")
    x, y, w, h = cv2.boundingRect(pts)
    return img[y:y+h, x:x+w].copy()

def detect_text_regions(
    img_color,
    detector=None,
    save_crops_dir=None,
    min_score=0.0
):
    """
    Runs detection and returns list of detections:
      [ { 'box': [[x,y],...], 'score': float, 'crop': ndarray }, ... ]
    Works with PaddleOCR 2.x predict() output (dict) and older shapes too.
    """
    det = detector or get_detector()
    # call predict() (2.x)
    raw = det.predict(img_color)

    # normalizing detection boxes from possible shapes
    boxes = []
    # Common new-format: raw is dict with "det": [boxes], maybe "det_score"
    if isinstance(raw, dict):
        boxes = raw.get("det", []) or raw.get("bboxes", []) or []
        # If det contains pair [box, score] handle later
    elif isinstance(raw, list):
        # older outputs could be list of line entries, flatten if necessary
        # try to interpret typical older shape: list of [box, (text, score)]
        candidate = raw
        # If nested (one page)
        if len(candidate) == 1 and isinstance(candidate[0], list):
            candidate = candidate[0]
        for e in candidate:
            try:
                box = e[0]
                boxes.append(box)
            except Exception:
                continue

    detections = []
    for i, item in enumerate(boxes):
        # item might be box or [box, score] depending on format
        if isinstance(item, (list, tuple)) and len(item) >= 4 and isinstance(item[0], (list, tuple)):
            # could be [box1, box2, ...] or [box, score]
            # detect whether item[0] is a point
            first = item[0]
            if isinstance(first, (list, tuple)) and len(first) == 2:
                # it's a box (4 points) -> use it
                box = item if len(item) == 4 else item[0:4]
                score = None
            else:
                # fallback
                box = item
                score = None
        else:
            box = item
            score = None

        crop = _crop_region(img_color, box)
        # try to get a score if available in raw/det dict under det_score or det_scores
        score = None
        if isinstance(raw, dict):
            # raw may store matching det_score array
            det_scores = raw.get("det_score") or raw.get("det_scores") or raw.get("scores")
            if det_scores and len(det_scores) > i:
                try:
                    score = float(det_scores[i])
                except Exception:
                    score = None

        detections.append({
            "box": box,
            "score": score if score is not None else 0.0,
            "crop": crop
        })

        if save_crops_dir:
            Path(save_crops_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(save_crops_dir) / f"crop_{i:03d}.jpg"), crop)

    return detections

