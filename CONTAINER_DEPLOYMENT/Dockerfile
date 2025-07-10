# FOR AZURE
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "flask", "run", "--host", "0.0.0.0", "--port", "8000"]