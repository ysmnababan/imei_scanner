import time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import os
from pathlib import Path
import shutil
import logging

from preprocessing.preprocess import preprocess_image
from text_detection.detector import set_detector, detect_text_regions
from text_recognition.recognizer import set_recognizer, recognize_crops
from post_processing import extractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ocr_pipeline")

app = FastAPI()

OUTPUT_DIR = "./outputs"


def ensure_dir(p: str):
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

def inspect_var(x):
    print("Type:", type(x))
    print("Dir:", dir(x))

    # For objects with fields
    if hasattr(x, "__dict__"):
        print("Fields (__dict__):", x.__dict__)

    # For dicts
    if isinstance(x, dict):
        print("Keys:", list(x.keys()))
    for i, item in enumerate(x):
        print(f"Item {i}:")
        print("  Type:", type(item))
        print("  Value:", item)

        if hasattr(item, "__dict__"):      # custom object
            print("  Fields:", item.__dict__)

        if isinstance(item, dict):
            print("  Keys:", list(item.keys()))

def find_imei_info(img_path: str, outdir: str, is_debug_save:bool):
    ensure_dir(outdir)
    
    t0 = time.time()
    logger.info("Preprocessing image...")
    img_color, gray, thresh = preprocess_image(img_path, save_dir=outdir, only_color=True)
    print(f"[TIMER] preprocess_image: {time.time() - t0:.3f}s")
    
    t0 = time.time()
    logger.info("Detecting text regions...")
    regions = detect_text_regions(img_color, save_crops_dir=os.path.join(outdir, "crops"), is_debug_save=is_debug_save)
    print(f"[TIMER] detections: {time.time() - t0:.3f}s")

    t0 = time.time()
    logger.info("Running recognition on cropped regions...")
    rec_results = recognize_crops(regions, save_recrops_dir=os.path.join(outdir, "recognized"))
    print(f"[TIMER] recognitions: {time.time() - t0:.3f}s")

    t0 = time.time()
    logger.info("Assembling full text...")
    full_text = "\n".join(
        r["rec_text"]
        for r in rec_results
        if isinstance(r.get("rec_text"), str) and r["rec_text"].strip()
    )

    logger.info("Extracting target block...")
    extracted = extractor.extract_info(rec_results)
    print(f"[TIMER] extraction: {time.time() - t0:.3f}s")

    return extracted


def find_total_price(img_path: str, outdir: str, is_debug_save:bool):
    ensure_dir(outdir)
    
    t0 = time.time()
    logger.info("Preprocessing image...")
    img_color, gray, thresh = preprocess_image(img_path, save_dir=outdir, only_color=True)
    print(f"[TIMER] preprocess_image: {time.time() - t0:.3f}s")
    
    t0 = time.time()
    logger.info("Detecting text regions...")
    regions = detect_text_regions(img_color, save_crops_dir=os.path.join(outdir, "crops"), is_debug_save=is_debug_save)
    print(f"[TIMER] detections: {time.time() - t0:.3f}s")

    t0 = time.time()
    logger.info("Running recognition on cropped regions...")
    rec_results = recognize_crops(regions, save_recrops_dir=os.path.join(outdir, "recognized"))
    print(f"[TIMER] recognitions: {time.time() - t0:.3f}s")

    t0 = time.time()
    logger.info("Assembling full text...")
    full_text = "\n".join(
        r["rec_text"]
        for r in rec_results
        if isinstance(r.get("rec_text"), str) and r["rec_text"].strip()
    )

    logger.info("Extracting target block...")
    extracted = extractor.extract_total_amount(rec_results)
    print(f"[TIMER] extraction: {time.time() - t0:.3f}s")

    return extracted

@app.post("/imei")
async def ocr_endpoint(
    file: UploadFile = File(...),
    is_debug_save = Form(False)
    ):
    try:
        # save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # run pipeline
        extracted = find_imei_info(temp_path, OUTPUT_DIR, is_debug_save=is_debug_save)

        # cleanup uploaded file
        os.remove(temp_path)

        return JSONResponse(extracted)

    except Exception as e:
        logger.exception("Error running OCR pipeline")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/invoice")
async def ocr_endpoint(
    file: UploadFile = File(...),
    is_debug_save = Form(False)
    ):
    try:
        # save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # run pipeline
        extracted = find_total_price(temp_path, OUTPUT_DIR, is_debug_save=is_debug_save)

        # cleanup uploaded file
        os.remove(temp_path)

        return JSONResponse(extracted)

    except Exception as e:
        logger.exception("Error running OCR pipeline")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "OCR Service is running! POST /ocr to process an image."}


if __name__ == "__main__":
    # Load your models ONCE at startup
    set_detector()
    set_recognizer()
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
