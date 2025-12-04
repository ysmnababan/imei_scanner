import argparse
import logging
import os
from pathlib import Path

from preprocessing.preprocess import preprocess_image
from text_detection.detector import set_detector, detect_text_regions 
from text_recognition.recognizer import set_recognizer, recognize_crops
from post_processing import extractor
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ocr_pipeline")

def ensure_dir(p: str):
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    # Delete all files and folders inside the directory
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

def run_pipeline(img_path: str, outdir: str):
    ensure_dir(outdir)
    logger.info("Preprocessing image...")
    img_color, gray, thresh = preprocess_image(img_path, save_dir=outdir,only_color=True)

    logger.info("Detecting text regions...")
    regions = detect_text_regions(img_color, save_crops_dir=os.path.join(outdir, "crops"))

    logger.info("Running recognition on cropped regions...")
    rec_results = recognize_crops(regions, save_recrops_dir=os.path.join(outdir, "recognized"))

    logger.info("Assembling full text and running post-processing...")
    full_text = "\n".join(
        r["rec_text"]
        for r in rec_results
        if isinstance(r.get("rec_text"), str) and r["rec_text"].strip()
    )
    extracted = extractor.extract_product_block(full_text)

    # Print results
    print("="*40)
    print("FULL OCR TEXT:\n")
    print(full_text)
    print("="*40)
    print("EXTRACTED TARGET BLOCK:\n")
    print(extracted or "No block matched.")
    print("="*40)

if __name__ == "__main__":
    set_detector()
    set_recognizer()
    parser = argparse.ArgumentParser(description="Modular OCR pipeline (preprocess -> detect -> recognize -> postprocess)")
    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument("--outdir", "-o", default="./outputs", help="Output directory to save intermediate images")
    args = parser.parse_args()
    run_pipeline(args.image, args.outdir)
