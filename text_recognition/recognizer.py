from paddleocr import TextRecognition

_RECOG = None

def set_recognizer(
    text_recognition_model_name="en_PP-OCRv3_mobile_rec",
):
    """
    Return a cached PaddleOCR recognition-only instance.
    """
    global _RECOG
    if _RECOG is None:
        _RECOG = TextRecognition(model_name=text_recognition_model_name,enable_hpi=True) 

def recognize_crops(
    regions,
    save_recrops_dir=None
):
    """
    Input: detections = [{ 'box':..., 'crop': ndarray, ... }, ...]
    Output: [{ 'text': str, 'score': float, 'box':..., 'crop': ndarray }, ...]
    Works with recognizer.predict() output dict keys: rec_text, rec_score, maybe rec_bbox.
    """
    crops_ndarrays = [r["crop"] for r in regions]
    print("crops ndarrays", len(crops_ndarrays))
    batch_output = _RECOG.predict(input=crops_ndarrays, batch_size=8)  # adjust batch_size
    recognized = []
    # `batch_output` is expected to be a list-like of results, one per input
    for i, res in enumerate(batch_output):
        # normalize result extraction depending on your API shape:
        # For paddlex TextRecognition result objects:
        recognized.append({"text": res["rec_text"], "score": float(res["rec_score"])})
    # attach results back to regions
    for r, rec in zip(regions, recognized):
        r["rec_text"] = rec["text"]
        r["rec_score"] = rec["score"]

    return regions
