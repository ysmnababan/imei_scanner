import cv2
import numpy as np
from pathlib import Path
import imutils

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img

def deskew_image(gray):
    # compute skew angle and rotate to deskew
    coords = np.column_stack(np.where(gray < 255))
    if coords.size == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    print("angle ", angle)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    rotated = imutils.rotate_bound(gray, angle)
    return rotated

def preprocess_image(image_path, save_dir=None, target_width=1400, only_color=True):
    img = load_image(image_path)
    if only_color:
        return img, None, None
    # convert to color and gray
    img_color = img.copy()
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # scale up to target width if smaller
    h, w = gray.shape
    if w < target_width:
        scale = target_width / w
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        img_color = cv2.resize(img_color, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)

    # denoise
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # deskew
    gray = deskew_image(gray)

    # adaptive threshold for sharp edges (useful for OCR)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)

    # optional morphological closing to fix gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(save_dir)/"pre_gray.jpg"), gray)
        cv2.imwrite(str(Path(save_dir)/"pre_thresh.jpg"), thresh)
        cv2.imwrite(str(Path(save_dir)/"pre_color.jpg"), img_color)

    return img_color, gray, thresh
