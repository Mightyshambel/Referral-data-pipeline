FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
CMD ["python", "scripts/your_script.py", "--data_dir", "data", "--out_dir", "/out"]
