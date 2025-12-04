from paddleocr import TextDetection

path= "./../assets/sample4.jpg"
model = TextDetection(model_name="PP-OCRv5_server_det")
output = model.predict(path, batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./outputs/")
    res.save_to_json(save_path="./outputs/res.json")