FROM python:3.12-slim

# Install OS dependencies required by PaddleOCR HPI
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    ffmpeg \
    wget \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PaddlePaddle (CPU) from official mirror
RUN python -m pip install --no-cache-dir paddlepaddle==3.2.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Install PaddleOCR
RUN python -m pip install --no-cache-dir paddleocr

# Install high performance inference deps
RUN paddleocr install_hpi_deps cpu

# Copy project
COPY . .

# Install your libraries
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
