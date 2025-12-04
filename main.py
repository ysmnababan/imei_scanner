import argparse
import logging
import os
from pathlib import Path

from preprocessing.preprocess import preprocess_image
from text_detection.detector import get_detector, detect_text_regions
from text_recognition.recognizer import get_recognizer, recognize_crops
from post_processing import extractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ocr_pipeline")

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def run_pipeline(img_path: str, outdir: str):
    ensure_dir(outdir)
    logger.info("Preprocessing image...")
    img_color, gray, thresh = preprocess_image(img_path, save_dir=outdir)

    logger.info("Detecting text regions...")
    detector = get_detector(
        text_detection_model_name="PP-OCRv5_server_det",
        text_det_thresh=0.2,
        text_det_box_thresh=0.3,
        lang="en"
    )
    detections = detect_text_regions(img_color, detector=detector, save_crops_dir=os.path.join(outdir, "crops"))

    logger.info("Running recognition on cropped regions...")
    recognizer = get_recognizer(
        text_recognition_model_name="PP-OCRv5_server_rec",
        text_rec_score_thresh=0.3,
        lang="en"
    )
    rec_results = recognize_crops(detections, recognizer=recognizer, save_recrops_dir=os.path.join(outdir, "recrops"))

    logger.info("Assembling full text and running post-processing...")
    full_text = "\n".join([r["text"] for r in rec_results if r["text"].strip()])
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
    parser = argparse.ArgumentParser(description="Modular OCR pipeline (preprocess -> detect -> recognize -> postprocess)")
    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument("--outdir", "-o", default="./outputs", help="Output directory to save intermediate images")
    args = parser.parse_args()
    run_pipeline(args.image, args.outdir)
