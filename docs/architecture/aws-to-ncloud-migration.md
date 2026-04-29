# AWS -> 네이버클라우드 서비스로 전환

## 배경

- 기존 inference-server는 AWS S3를 전제로 object storage를 사용하고 있었습니다.
- 프로젝트 진행 중 스토리지 사용처를 AWS에서 네이버클라우드 Object Storage로 전환해야 하는 요구가 생겼습니다.
- 네이버클라우드 Object Storage는 S3 호환 API를 제공하므로, SDK를 전면 교체하기보다 스토리지 설정 계약을 일반화하는 방향으로 대응했습니다.

## 목표

- inference-server가 AWS 전용 환경변수 없이 네이버클라우드 Object Storage를 사용할 수 있어야 함
- 기존 S3 기반 구현 경계를 크게 흔들지 않고 운영 리스크를 최소화할 것
- 로컬/컨테이너/API 경유 실행 경로에서 실제 업로드 및 조회가 가능해야 함
- 문서와 예제 설정도 새 계약에 맞게 정리할 것

## 적용 방향

- `boto3`는 유지
- 스토리지 설정만 AWS 전용 명명에서 S3 호환 object storage 명명으로 일반화
- 기본 환경변수를 `STORAGE_*` 기준으로 통일
- 기존 `AWS_*` 변수는 fallback 으로 남겨 급격한 회귀를 방지

## 주요 변경 사항

### 1. 스토리지 설정 계약 변경

기존:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_S3_BUCKET_NAME`
- `AWS_REGION`

변경 후:

- `STORAGE_ACCESS_KEY_ID`
- `STORAGE_SECRET_ACCESS_KEY`
- `STORAGE_BUCKET_NAME`
- `STORAGE_REGION`
- `STORAGE_ENDPOINT_URL`
- `STORAGE_ADDRESSING_STYLE`

네이버클라우드 기본값:

- `STORAGE_REGION=kr-standard`
- `STORAGE_ENDPOINT_URL=https://kr.object.ncloudstorage.com`
- `STORAGE_ADDRESSING_STYLE=path`

### 2. inference-server 스토리지 클라이언트 일반화

- `apps/inference-server/src/storage/s3.py`
- `boto3.client("s3")`는 유지
- endpoint URL, region, bucket, addressing style 을 환경변수로 주입 가능하도록 변경
- 예외 메시지를 AWS/S3 특정 용어보다 일반적인 storage 표현으로 정리

### 3. 헬스체크 및 설정 반영

- `apps/inference-server/src/health/checks.py`
- readiness/check_storage 가 `STORAGE_*` 기준으로 설정 누락을 판단하도록 수정
- deep probe 시 object storage bucket 접근을 점검하도록 유지

### 4. 실행 스크립트 및 compose 반영

- `infra/compose/docker-compose.local.yml`
- `scripts/smoke-inference-qwen.sh`
- `scripts/run-qwen-via-api.sh`
- `scripts/run-qwen-local-image.sh`
- object storage 기준 안내 문구로 정리
- local compose 에 `STORAGE_REGION`, `STORAGE_ENDPOINT_URL`, `STORAGE_ADDRESSING_STYLE` 기본값 반영
- `infra/compose/docker-compose.dev.yml`, `infra/compose/docker-compose.prod.yml`에도 동일한 storage 계약을 반영해 환경 차이를 줄임

### 5. 예제 설정 및 문서 정리

- `.env.example`를 `STORAGE_*` 기준으로 수정
- S3/AWS 전제 문구가 남아 있던 architecture/sprint/setup 문서를 object storage 기준으로 정리
- `README.md`의 "AWS 관련 민감한 값" 문구를 object storage 기준으로 수정

## 검증 결과

### 1. 단위 테스트

- `conda run -n smartglass python -m unittest tests.test_storage_s3 tests.test_health_checks tests.test_task_contract`
- 결과: 통과

### 2. Docker smoke test

- `bash scripts/smoke-inference-qwen.sh`
- 결과: 성공
- 확인 내용:
  - inference image build
  - readiness 확인
  - worker 내부에서 sample image 업로드 및 추론
  - `VlmInferenceResult` 정상 반환

### 3. API 경유 end-to-end 테스트

- `SKIP_BUILD=1 bash scripts/run-qwen-via-api.sh apps/inference-server/sample_data/key_1.jpg`
- 결과: 성공
- 확인 내용:
  - inference-api 컨테이너에서 object storage 업로드
  - task enqueue
  - worker 가 object storage 에서 이미지 조회
  - Qwen 7B 추론 완료
  - API polling 결과 `status=success`

### 4. 호스트 conda 환경 직접 검증

- DNS 수정 이후 아래 흐름 재검증
  - `probe_bucket_access()`
  - `write_object()`
  - `read_object()`
- 결과: 성공
- 확인 예시:
  - bucket: `sgp-inference-storage-v1`
  - content_type: `text/plain`
  - body: `smartglass-object-storage-probe`

## 운영 이슈 및 대응

### WSL DNS 이슈

- 초기에는 호스트 WSL 환경에서 `kr.object.ncloudstorage.com` DNS 해석이 실패했습니다.
- 같은 시점에 Docker 컨테이너 내부 경로는 정상 동작했기 때문에, 스토리지 권한 문제보다 호스트 DNS 문제로 판단했습니다.

재현 증상:

- `getent hosts kr.object.ncloudstorage.com` 실패
- `socket.getaddrinfo('kr.object.ncloudstorage.com', 443)` 실패
- `/etc/resolv.conf` 가 WSL 자동 생성 DNS 를 가리킴

조치:

- `/etc/wsl.conf` 에 `generateResolvConf = false` 설정
- `/etc/resolv.conf` 를 수동 DNS 로 재작성
- WSL 재시작 후 이름 해석 재확인

결과:

- 호스트에서도 `kr.object.ncloudstorage.com` 이름 해석 성공
- 이후 `conda run -n smartglass` 기준 object storage probe/put/get 성공

## 왜 이렇게 설계했는가

- 네이버 Object Storage는 S3 호환 API 이므로, `boto3` 자체를 버리는 것은 실익보다 리스크가 큼
- 현재 구조에서 중요한 것은 SDK 교체가 아니라 "AWS 전용 설정 계약 제거"였음
- `STORAGE_*` 계약으로 일반화하면 이후 다른 S3 호환 스토리지로도 이동이 쉬움
- 기존 `AWS_*` fallback 유지로, 중간 전환 과정에서 설정 회귀를 줄일 수 있음

## 남은 과제

- 현재 `infra/compose/docker-compose.dev.yml`, `infra/compose/docker-compose.prod.yml`는 골격 수준이라, 실제 inference-server 서비스가 들어갈 때 같은 `STORAGE_*` 계약을 그대로 유지해야 함
- 배포 환경에서 object storage credential rotation 방식과 secret 주입 경로를 정리할 필요가 있음
- 필요하면 storage 설정 키 이름을 `packages/shared-config` 수준으로 끌어올려 서비스 간 공통화할 수 있음

## 변경 파일 요약

- `.env.example`
- `README.md`
- `apps/inference-server/src/storage/s3.py`
- `apps/inference-server/src/health/checks.py`
- `apps/inference-server/src/utils/config.py`
- `apps/inference-server/tests/test_storage_s3.py`
- `apps/inference-server/tests/test_health_checks.py`
- `apps/inference-server/tests/test_task_contract.py`
- `infra/compose/docker-compose.local.yml`
- `infra/compose/docker-compose.dev.yml`
- `infra/compose/docker-compose.prod.yml`
- `scripts/smoke-inference-qwen.sh`
- `scripts/run-qwen-via-api.sh`
- `scripts/run-qwen-local-image.sh`
- 관련 architecture/setup/sprint 문서

## 결론

- inference-server 는 AWS 전용 S3 설정에서 벗어나 네이버클라우드 Object Storage 기반으로 동작하도록 전환되었습니다.
- 로컬 단위 테스트, Docker smoke test, API 경유 end-to-end, 호스트 conda direct probe 까지 모두 성공했습니다.
- 현재 기준으로 개발/테스트 경로에서는 네이버 Object Storage 전환이 완료된 상태로 판단할 수 있습니다.
