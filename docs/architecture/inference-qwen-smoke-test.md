# Inference Qwen Smoke Test

## 목적

`apps/inference-server`에 추가한 Qwen VLM adapter를 Docker 기준으로 재현 가능하게 검증합니다.
기본 smoke 모델은 `qwen2.5-vl-3b`이며, 더 무거운 검증이 필요하면 `MODEL_KEY=qwen2.5-vl-7b`로 실행합니다.

검증 범위:

- inference-server 이미지 빌드
- readiness endpoint 확인
- 샘플 이미지 1장을 Object Storage에 업로드
- worker 내부에서 `process_vision_inference()` 실행
- 결과가 `VlmInferenceResult` 계약을 만족하는지 검증

## 전제 조건

- Docker Desktop WSL integration 활성화
- `.env` 파일 존재
- 아래 환경변수 설정
  - `STORAGE_ACCESS_KEY_ID`
  - `STORAGE_SECRET_ACCESS_KEY`
  - `STORAGE_BUCKET_NAME`
  - `STORAGE_REGION`
  - `STORAGE_ENDPOINT_URL`

## 실행 방법

저장소 루트에서 아래 스크립트를 실행합니다.

```bash
bash scripts/smoke-inference-qwen.sh
```

7B로 실행하려면:

```bash
MODEL_KEY=qwen2.5-vl-7b bash scripts/smoke-inference-qwen.sh
```

## 내부 동작

스크립트는 아래 순서로 진행됩니다.

1. `infra/docker/inference-server.Dockerfile` 기준으로 `inference-worker`, `inference-api` 이미지를 빌드
2. `redis`, `inference-api`를 기동
3. `GET /health/ready`를 컨테이너 내부에서 확인
4. `apps/inference-server/sample_data/key_1.jpg`를 Object Storage에 업로드
5. worker 컨테이너에서 `apps/inference-server/scripts/smoke_qwen_e2e.py` 실행
6. `VlmInferenceResult` JSON 출력 및 필수 키 검증

## 참고

- smoke test는 Celery broker round-trip 대신 worker 프로세스 내부에서 `process_vision_inference()`를 직접 호출합니다.
- 목적은 모델 로딩, Object Storage 접근, 계약 직렬화, Qwen adapter 연결을 빠르게 검증하는 것입니다.
- 이후 실제 비동기 큐 round-trip 검증이 필요하면 별도 통합 테스트를 추가하는 것이 좋습니다.
