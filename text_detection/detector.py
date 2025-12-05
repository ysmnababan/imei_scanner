import cv2
import numpy as np
from pathlib import Path
from paddleocr import TextDetection
import os 

_DET = None

def set_detector(text_detection_model_name="PP-OCRv5_server_det",):
    """
    Create or return a cached PaddleOCR detector instance (detection-only).
    """
    global _DET
    if _DET is None:
        _DET = TextDetection(model_name=text_detection_model_name)

def sort_quad_points(pts):
    """
    Accept 4 points and return in order: tl, tr, br, bl
    pts: (4,2) array-like
    """
    pts = np.array(pts, dtype=np.float32)
    # sum and diff method
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).squeeze()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def quad_to_rect_size(quad):
    """
    Given quad ordered tl,tr,br,bl return (width, height) for destination rectangle
    """
    tl, tr, br, bl = quad
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    # ensure at least 1
    return max(1, maxWidth), max(1, maxHeight)

def perspective_crop(image, quad, pad_pixels=4, out_size=None):
    """
    Warp quadrilateral region to a straight rectangle.
    - image: BGR ndarray
    - quad: (4,2) points in any order
    - pad_pixels: expand rectangle by this amount (both horizontal and vertical) before warp
    - out_size: (w,h) override; if None compute from quad
    Returns the warped crop (RGB/BGR same as input).
    """
    quad = sort_quad_points(quad)
    w, h = quad_to_rect_size(quad)

    # apply padding in the local rectangle space: we expand w/h by pad_pixels each side
    w += pad_pixels * 2
    h += pad_pixels * 2

    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    # shift source quad so padding maps correctly: we need to move source corners outward along edges
    # A simple approach: compute rectangle center and extend quad points away from center proportionally
    center = quad.mean(axis=0)
    vecs = quad - center
    # factor to expand so dest rectangle (w,h) fits proportionally to source bounding box
    src_w, src_h = quad_to_rect_size(quad)
    scale_x = (w) / max(1, src_w)
    scale_y = (h) / max(1, src_h)
    scale = max(scale_x, scale_y)
    quad_expanded = center + vecs * scale

    M = cv2.getPerspectiveTransform(quad_expanded, dst)
    warped = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped

def minarea_rotated_crop(image, quad, pad=4):
    """
    Alternative: take minAreaRect of the quadrilateral and rotate the crop to horizontal.
    Returns the rotated crop.
    """
    quad = np.array(quad, dtype=np.int32)
    rect = cv2.minAreaRect(quad)    # ((cx,cy),(w,h),angle)
    (cx, cy), (w, h), angle = rect
    # Make sure width is the longer side (we expect w >= h)
    if w < h:
        w, h = h, w
        angle += 90.0

    # get rotation matrix
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    H, W = image.shape[:2]
    rotated = cv2.warpAffine(image, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # crop axis-aligned bbox around rotated rect center
    x1 = int(cx - w/2) - pad
    y1 = int(cy - h/2) - pad
    x2 = int(cx + w/2) + pad
    y2 = int(cy + h/2) + pad

    # clamp
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)

    crop = rotated[y1:y2, x1:x2]
    return crop

def save_debug_crop(img, poly, score, out_dir, idx, method="persp", is_debug_save=False):
    os.makedirs(out_dir, exist_ok=True)
    if method == "persp":
        crop = perspective_crop(img, poly, pad_pixels=6)
    else:
        crop = minarea_rotated_crop(img, poly, pad=6)
    # optional contrast/threshold for debug save
    if is_debug_save:
        return "", crop
    save_path = os.path.join(out_dir, f"crop_{idx:03d}_{method}_{score:.2f}.jpg")
    cv2.imwrite(save_path, crop)
    return save_path, crop

def sort_polygons_and_scores(polys, scores):
    # combine polys and scores so they stay paired
    combined = list(zip(polys, scores))

    # sort by top-most (y) then left-most (x)
    combined_sorted = sorted(
        combined,
        key=lambda item: (
            min(pt[1] for pt in item[0]),  # sort by Y (top)
            min(pt[0] for pt in item[0])   # then X (left)
        )
    )

    # unzip back
    sorted_polys, sorted_scores = zip(*combined_sorted)

    return list(sorted_polys), list(sorted_scores)

def detect_text_regions(
    img,
    save_crops_dir=None,
    is_debug_save=False
):
    """
    Runs detection and returns list of detections:
      [ { 'box': [[x,y],...], 'score': float, 'crop': ndarray }, ... ]
    Works with PaddleOCR 2.x predict() output (dict) and older shapes too.
    """
    
    # Output folder
    os.makedirs(save_crops_dir, exist_ok=True)

    result=_DET.predict(img, batch_size=1)[0]

    # FOR DEBUG
    # result.print()
    result.save_to_img(save_path=os.path.join(save_crops_dir, "with_bounding_box.jpg"))
    # result.save_to_json(save_path="./outputs/res.json")

    polys, scores = sort_polygons_and_scores(result["dt_polys"], result["dt_scores"])
    
    # Collect region dictionaries (no reloading)
    regions = []
    for i, (poly, score) in enumerate(zip(polys, scores)):
        # Use perspective crop (returns file path and the actual crop ndarray)
        save_path, crop_img = save_debug_crop(img, poly, score, save_crops_dir, i, method="persp", is_debug_save=is_debug_save)
        # Optionally also create rotated crop variant by calling save_debug_crop(..., method="rot")
        regions.append({
            "crop": crop_img,    # BGR numpy array
            "poly": poly,        # polygon (4x2)
            "score": float(score),
            "save_path": save_path
        })

    print(f"Collected {len(regions)} crops in memory (no disk reload).")
    return regions

