
# Phone Detail Scanner OCR App

OCR application for scanning **phone details** such as **IMEI, Serial Number, Model Number** using **PaddleOCR**.

---

## Project Structure

```
preprocessing/       # Image preprocessing (resize, denoise, etc.)
text_detection/      # Detect text regions in images
text_recognition/    # Recognize text from detected regions
post_processing/     # Clean and extract relevant information (IMEI, Serial)
main.py              # Starts API server
```

---

## Installation

### 1. Python Virtual Environment (Linux / CPU example)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install PaddlePaddle (CPU version)
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu

# Install PaddleOCR
python -m pip install paddleocr

# Enable high performance inference (CPU)
paddleocr install_hpi_deps cpu

# Install other dependencies
pip install -r requirements.txt
```

> **Note:** For better OCR accuracy, use higher-resolution images. For production, GPU inference is recommended for faster performance.

---

## Usage

This app exposes an **API** for OCR tasks. You can run it via Python CLI or Docker — both work the same way.

### 1. Using Python CLI

Start the API server:

```bash
python main.py
```

The API will run on `http://localhost:8000`.

Available endpoints:

* **Scan Invoice / Image**:
  `POST http://localhost:8000/invoice`
* **Scan Phone IMEI / Serial**:
  `POST http://localhost:8000/imei`

You can use `curl` or any HTTP client to send images:

```bash
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/imei
```

---

### 2. Using Docker

Build and run the container:

```bash
# Build Docker image
docker build -t ocr-service .

# Run container (example on port 8000)
docker run -e PORT=8000 -p 8000:8000 ocr-service
```

The API will be available at `http://localhost:8000`.

Endpoints are the same:

* `/invoice`
* `/imei`

Example `curl` request:

```bash
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/imei
```

---

## Notes & Recommendations

* Use high-resolution images for better OCR accuracy.
* Apply preprocessing (denoise, contrast adjustment, super-resolution) if necessary.
* For production:

  * Use GPU for faster inference.
  * Cache OCR models to speed up responses.
  * Secure API endpoints if exposed publicly.
