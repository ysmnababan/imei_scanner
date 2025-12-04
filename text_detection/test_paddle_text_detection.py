from paddleocr import TextDetection
import cv2
import numpy as np
import os 

def sort_polygons(polys):
    return sorted(polys, key=lambda p: (min(y[1] for y in p), min(x[0] for x in p)))

def crop_polygon(img, polygon):
    pts = np.array(polygon, dtype=np.float32)

    # Compute bounding rectangle
    x, y, w, h = cv2.boundingRect(pts)
    crop = img[y:y+h, x:x+w]

    return crop

path= "./../assets/sample4.jpg"
# img = cv2.imread(path)

# Output folder
crop_dir = "./outputs/crop"
os.makedirs(crop_dir, exist_ok=True)

model = TextDetection(model_name="PP-OCRv5_server_det")
result= model.predict(path, batch_size=1)[0]

# if needed
result.print()
result.save_to_img(save_path="./outputs/")
result.save_to_json(save_path="./outputs/res.json")

polys = sort_polygons(result['dt_polys'])
scores = result['dt_scores']
img= result['input_img']

for idx, (poly, score) in enumerate(zip(polys, scores)):
    crop = crop_polygon(img, poly)
    # Save with OpenCV
    save_path = os.path.join(crop_dir, f"crop_{idx:03d}.jpg")
    cv2.imwrite(save_path, crop)

    print(f"Saved: {save_path} (score={score:.2f})")