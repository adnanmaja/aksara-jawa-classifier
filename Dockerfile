# FOR AZURE
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY . /app
COPY ONNX_MODELS/ /app/ONNX_MODELS/

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "api:app"]