FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
# CPU-only image: torch CPU wheel keeps the image small; swap for the cu124
# index if GPU inference is needed in the container.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY app.py .

ENV MODEL_DIR=/app/model
EXPOSE 8000 8501

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
