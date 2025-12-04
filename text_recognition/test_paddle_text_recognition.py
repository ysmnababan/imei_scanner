from paddleocr import TextRecognition
import os 

path= "./../text_detection/outputs/crops"
# Collect all image files in the directory
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

img_list = [
    os.path.join(path, f)
    for f in os.listdir(path)
    if f.lower().endswith(valid_ext)
]

if len(img_list) == 0:
    raise RuntimeError(f"No image files found in: {path}")

print("Number of crops:", len(img_list))

model = TextRecognition(model_name="en_PP-OCRv5_mobile_rec")
output = model.predict(input=img_list, batch_size=1)
for i, res in enumerate(output):
    print(f"--- Result {i} ---")
    res.print()

    # save image (auto-names files inside output folder)
    res.save_to_img(save_path="./outputs/")

    # save json with unique name
    # json_path = f"./outputs/crop_{i}.json"
    # res.save_to_json(save_path=json_path)