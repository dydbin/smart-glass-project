FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY apps/inference-server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/inference-server /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["celery", "-A", "src.queue.tasks", "worker", "--loglevel=info", "-P", "solo"]
