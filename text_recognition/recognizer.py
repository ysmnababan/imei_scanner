# text_recognition/__init__.py
import cv2
import numpy as np
from pathlib import Path
from paddleocr import PaddleOCR
import pytesseract

_RECOG = None

def get_recognizer(
    text_recognition_model_name="en_PP-OCRv4_server_rec",
    text_rec_score_thresh=0.3,
    return_word_box=False,
    lang="en"
):
    """
    Return a cached PaddleOCR recognition-only instance.
    """
    global _RECOG
    if _RECOG is None:
        _RECOG = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_recognition_model_name=text_recognition_model_name,
            text_rec_score_thresh=text_rec_score_thresh,
            return_word_box=return_word_box,
            lang=lang
        )
    return _RECOG

def recognize_crops(
    detections,
    recognizer=None,
    fallback_tesseract=True,
    save_recrops_dir=None
):
    """
    Input: detections = [{ 'box':..., 'crop': ndarray, ... }, ...]
    Output: [{ 'text': str, 'score': float, 'box':..., 'crop': ndarray }, ...]
    Works with recognizer.predict() output dict keys: rec_text, rec_score, maybe rec_bbox.
    """
    rec = recognizer or get_recognizer()
    results = []
    for i, d in enumerate(detections):
        crop = d["crop"]
        text = ""
        score = 0.0

        try:
            raw = rec.predict(crop)
            # Common new-format: dict with rec_text (list) and rec_score (list)
            if isinstance(raw, dict):
                texts = raw.get("rec_text") or raw.get("texts") or []
                scores = raw.get("rec_score") or raw.get("rec_scores") or raw.get("scores") or []
                if texts:
                    # take the first (most-likely) recognized string
                    text = texts[0] if isinstance(texts, (list, tuple)) else str(texts)
                    try:
                        score = float(scores[0]) if scores else 0.0
                    except Exception:
                        score = 0.0
            elif isinstance(raw, list):
                # older shape: list of entries [box, (text, score)]
                candidate = raw
                if len(candidate) == 1 and isinstance(candidate[0], list):
                    candidate = candidate[0]
                parts = []
                scs = []
                for entry in candidate:
                    try:
                        item = entry[1]
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            parts.append(str(item[0]))
                            scs.append(float(item[1]))
                    except Exception:
                        continue
                if parts:
                    text = " ".join(parts)
                    score = max(scs) if scs else 0.0
        except Exception:
            # swallow and fallback below
            text = ""
            score = 0.0

        # Optional fallback to tesseract for low-confidence/noisy crops
        if fallback_tesseract and (not text.strip() or score < 0.35):
            try:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
                _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                t = pytesseract.image_to_string(th, config="--oem 1 --psm 6")
                if t and t.strip():
                    text = t.strip()
                    score = max(score, 0.4)
            except Exception:
                pass

        results.append({
            "text": text,
            "score": float(score),
            "box": d.get("box"),
            "crop": crop
        })

        if save_recrops_dir:
            Path(save_recrops_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(save_recrops_dir) / f"recrop_{i:03d}.jpg"), crop)

    return results
