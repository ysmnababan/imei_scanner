# Phone Detail Scanner OCR App
OCR app for scanning Phone detail like IMEI, Serial Number, Model Number using Tesseract OCR.

Folders:
- preprocessing/
- text_detection/
- text_recognition/
- post_processing/
- main.py

## Install (Linux/CPU example)
1. Create venv:
   python -m venv venv
   source venv/bin/activate

2. Install paddlepaddle (CPU) (follow official instructions for your platform):
   python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

3. Install rest:
   pip install -r requirements.txt

## Usage
python main.py --image /path/to/image.jpg --outdir ./outputs

Outputs:
- saved intermediate images in outdir (gray, thresh, crops)
- printed extracted fields and raw OCR text

## Notes
- For better accuracy, capture photos at higher DPI or apply super-resolution.
- If you plan to deploy, consider running inference on GPU and caching models.
