import os


class Settings:
    celery_broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    storage_region = os.getenv("STORAGE_REGION", os.getenv("AWS_REGION", "ap-northeast-2"))
