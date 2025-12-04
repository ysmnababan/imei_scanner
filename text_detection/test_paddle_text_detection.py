from paddleocr import TextDetection
import cv2
import numpy as np
import os 

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

def save_debug_crop(img, poly, score, out_dir, idx, method="persp"):
    os.makedirs(out_dir, exist_ok=True)
    if method == "persp":
        crop = perspective_crop(img, poly, pad_pixels=6)
    else:
        crop = minarea_rotated_crop(img, poly, pad=6)
    # optional contrast/threshold for debug save
    save_path = os.path.join(out_dir, f"crop_{idx:03d}_{method}_{score:.2f}.jpg")
    cv2.imwrite(save_path, crop)
    return save_path

def sort_polygons(polys):
    return sorted(polys, key=lambda p: (min(y[1] for y in p), min(x[0] for x in p)))

def crop_polygon(img, polygon):
    pts = np.array(polygon, dtype=np.float32)

    # Compute bounding rectangle
    x, y, w, h = cv2.boundingRect(pts)
    crop = img[y:y+h, x:x+w]

    return crop

path= "./../assets/sample3.jpg"
img = cv2.imread(path)

# Output folder
crop_dir = "./outputs/crops"
os.makedirs(crop_dir, exist_ok=True)

model = TextDetection(model_name="PP-OCRv5_server_det")
result= model.predict(path, batch_size=1)[0]

# if needed
result.print()
result.save_to_img(save_path="./outputs/")
result.save_to_json(save_path="./outputs/res.json")

polys = sort_polygons(result['dt_polys'])
scores = result['dt_scores']
# img= result['input_img']

for i, (poly, score) in enumerate(zip(polys, scores)):
    # best: perspective crop
    save_p = save_debug_crop(img, poly, score, crop_dir, i, method="persp")
    # optional rotated crop as fallback
    # save_r = save_debug_crop(img, poly, score, crop_dir, i, method="rot")
    print("Saved:", save_p, )