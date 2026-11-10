# Run `python -m ci.train` first: the image serves artifacts/, it does not train.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ci ci
COPY artifacts artifacts
EXPOSE 8000
CMD ["uvicorn", "ci.api:app", "--host", "0.0.0.0", "--port", "8000"]
