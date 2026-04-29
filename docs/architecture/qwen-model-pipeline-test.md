# Qwen 모델 파이프라인 테스트

## 문서 목적

동료가 올린 `AI/` 기반 Qwen 실험 코드를 현재 저장소 구조에 맞춰 `apps/inference-server`에 편입 가능한 형태로 재구성했고, Docker 기준으로 바로 재현 가능한 테스트 경로를 정리한다.

이 문서는 아래 두 가지 용도로 사용한다.

- 팀 내부 검증 공유
- 노션 문서 `qwen 모델 파이프라인 테스트` 초안

## 배경

초기 PR은 `AI/` 디렉터리 아래에 실험 코드가 들어가 있었고, 현재 서비스 구조인 `apps/inference-server`와 직접 연결되지는 않은 상태였다.

그래서 이번 작업에서는 `AI/` 코드를 그대로 옮기지 않고, 아래 방향으로 분리 편입했다.

- Qwen VLM adapter: `apps/inference-server/src/models/qwen_vlm.py`
- JSON 후처리/정규화: `apps/inference-server/src/contracts/qwen_parser.py`
- 실행 엔트리포인트 연결: `apps/inference-server/src/queue/tasks.py`
- Docker smoke / benchmark 스크립트: `scripts/`

## 이번 작업에서 정리한 내용

### 1. Qwen 모델 adapter 편입

- `qwen2.5-vl-3b`
- `qwen2.5-vl-7b`

두 모델을 `apps/inference-server` 내부에서 공통 방식으로 로딩할 수 있게 정리했다.

추가한 운영 관점 안전장치:

- `4bit` 양자화
- 이미지 자동 다운스케일
- `max_pixels` 제한
- `7B` 실패 시 `3B` fallback

### 2. 후처리 로직 선별 이식

`AI/pipeline.py`에서 그대로 가져오지 않고, 현재 구조에 맞는 최소 안전 로직만 이식했다.

- object name dedup
- nearby candidate 수집
- noisy / scene-like token 제거
- tag 재구성
- 선택적 object review / missing-object check

### 3. Docker 실행 경로 정리

동료가 별도 로컬 스크립트 수정 없이 바로 실행할 수 있도록 아래 스크립트를 추가했다.

- 1장 smoke test: `scripts/smoke-inference-qwen.sh`
- 2장 이상 batch 비교: `scripts/benchmark-qwen-models.sh`
- 원하는 로컬 이미지 1장 테스트: `scripts/run-qwen-local-image.sh`

## 동료가 바로 재현하는 방법

### 전제 조건

- Docker Desktop 실행
- WSL integration 활성화
- 루트에 `.env` 파일 존재
- `.env.example`를 복사해서 `.env` 생성

필수 env:

- `STORAGE_ACCESS_KEY_ID`
- `STORAGE_SECRET_ACCESS_KEY`
- `STORAGE_BUCKET_NAME`
- `STORAGE_REGION`
- `STORAGE_ENDPOINT_URL`
- `HF_TOKEN`

권장 env:

- `VISION_QWEN_FALLBACK_MODEL=qwen2.5-vl-3b`
- `VISION_QWEN_ENABLE_OBJECT_REVIEW=1`

### 1. 1장 smoke test

기본은 `3B`:

```bash
make smoke-inference-qwen
```

`7B`로 확인:

```bash
MODEL_KEY=qwen2.5-vl-7b bash scripts/smoke-inference-qwen.sh
```

검증 범위:

- inference image build
- readiness probe
- sample image object storage upload
- worker 내부 `process_vision_inference()` 호출
- `VlmInferenceResult` contract 검증

### 2. batch benchmark

현재는 2장 기준으로 비교한 상태다.

```bash
LIMIT=2 bash scripts/benchmark-qwen-models.sh
```

결과 파일:

- `output/qwen-benchmark-report.json`

### 3. 원하는 로컬 사진 테스트

원하는 사진으로 바로 테스트하려면:

```bash
MODEL_KEY=qwen2.5-vl-7b bash scripts/run-qwen-local-image.sh /absolute/or/relative/path/to/image.jpg
```

또는:

```bash
make run-qwen-local-image IMAGE=/absolute/or/relative/path/to/image.jpg
```

기본 모델은 `qwen2.5-vl-7b`이며, 필요하면 `MODEL_KEY=qwen2.5-vl-3b`로 바꿔서 실행할 수 있다.

반복 테스트 시에는 worker를 계속 켜둔 상태에서 `docker compose exec`로 실행하므로, 기본값은 재빌드를 생략한다.

코드 변경 직후에만 강제로 다시 빌드하려면:

```bash
SKIP_BUILD=0 MODEL_KEY=qwen2.5-vl-7b bash scripts/run-qwen-local-image.sh /absolute/or/relative/path/to/image.jpg
```

## 현재 검증 결과

### 1. 7B 단일 이미지 smoke

카카오톡 이미지 포함해서 `qwen2.5-vl-7b`는 Docker 환경에서 실제 추론 성공까지 확인했다.

추가로 적용한 안정화:

- 큰 이미지 입력 시 자동 축소
- `7B` 추론에서 OOM 방지용 픽셀 제한

### 2. 2장 benchmark 결과

비교 샘플:

- `key_1.jpg`
- `wallet_1.jpg`

요약:

- `qwen2.5-vl-3b`
  - `successCount: 2/2`
  - `avgLatencySec: 13.8559`
  - `avgObjectCount: 0.5`
  - `avgTagCount: 4`

- `qwen2.5-vl-7b`
  - `successCount: 2/2`
  - `avgLatencySec: 13.6103`
  - `avgObjectCount: 4.5`
  - `avgTagCount: 7.5`

해석:

- 현재 2장 기준으로는 `7B`가 `3B`보다 훨씬 풍부한 object/tag를 생성한다.
- 다만 `wallet_1.jpg` 같은 샘플에서는 `7B`도 hallucination이 있다.
- 즉, 인프라/서빙/계약은 검증됐지만 모델 품질은 추가 튜닝이 필요하다.

## 이번 단계 결론

현재 상태는 아래처럼 정리할 수 있다.

- `AI/` 실험 코드를 서비스 구조에 직접 편입 가능한 최소 골격으로 옮김
- Docker에서 팀원이 바로 재현 가능한 실행 경로 확보
- `7B`는 동작 확인 완료
- `7B -> 3B fallback`도 코드 레벨로 반영 완료
- 품질은 아직 운영 기본값으로 확정할 단계는 아님

## 남은 작업

- batch benchmark를 5~10장으로 확대
- hallucination 유형 분류
- object review / missing-object prompt 조정
- 도메인별 alias 규칙 보강
- 최종 운영 기본 모델 결정

## 파이프라인 병목 1차 점검

Qwen multi-stage pipeline은 이제 `pipelineOutput.pipeline_meta`에 단계별 진단값을 포함한다.

추가된 주요 필드:

- `stage_timings_sec`: 단계별 누적 소요 시간
- `stage_call_counts`: 단계별 호출 횟수
- `stage_total_sec`: 기록된 단계 시간의 합계
- `slowest_stage`: 해당 실행에서 가장 오래 걸린 단계
- `candidate_counts`: object 후보가 단계별로 얼마나 늘거나 줄었는지 보여주는 카운트
- `image_pixels`: 원본/처리 이미지 픽셀 수와 다운스케일 여부

1차 병목 판단 기준:

- `object_detail` 시간이 크고 호출 횟수가 많으면 object 수가 latency를 밀어 올리는 상황이다.
- `object_list` 또는 `scene_summary`가 가장 느리면 기본 generation 자체가 병목일 가능성이 높다.
- `deduplicate` 또는 `missing_object_check`가 커지면 후처리용 VLM 재확인 prompt가 병목 후보가 된다.
- `image_pixels.downscaled=true`이면 입력 이미지 크기가 latency와 VRAM 사용량에 영향을 준 것으로 해석한다.

`benchmark_qwen_models.py`는 각 case에 `pipelineDiagnostics`를 남기고, 모델별 summary에 `avgStageTimingsSec`와 `slowestStageCounts`를 포함한다. 따라서 실제 GPU benchmark를 돌린 뒤에는 총 latency뿐 아니라 어느 단계가 반복적으로 가장 느린지도 같이 비교할 수 있다.

## PR에 같이 적으면 좋은 요약

이번 PR은 `AI/` 아래에 있던 Qwen 실험 코드를 직접 편입한 것이 아니라, `apps/inference-server` 구조에 맞는 adapter / parser / task 연결 / Docker smoke test 경로를 추가한 작업이다.

팀원이 바로 확인할 수 있도록 Docker 기준 smoke 및 benchmark 스크립트를 같이 정리했고, 현재는 2장 기준 benchmark와 7B single-image smoke까지 검증한 상태다.

품질은 아직 추가 평가가 필요하지만, 최소한 다음 단계의 협업 검증은 바로 가능한 상태다.
