# 웨어러블 VLM 파이프라인 — 구현 플랜

> XIAO ESP32S3 Sense 기반 안경 카메라 → 이미지 전처리 → VLM 캡션 생성 → JSON 정제
> 목적: 5분마다 자동 촬영 후 "내 지갑 어딨어?" 같은 질의에 위치 응답

---

## 0. 하드웨어 스펙 (XIAO ESP32S3 Sense)

| 항목 | 기본 스펙 | 실사용 권장 |
|---|---|---|
| 카메라 | OV2640 (최대 1600×1200) / OV3660 (최대 2048×1536) | 640×480 (VGA) 권장 |
| 메모리 | 8MB PSRAM + 8MB Flash | SD카드 병행 또는 WiFi 전송 |
| 통신 | 2.4GHz WiFi + BLE 5.0 | 이미지 → 서버 WiFi 전송에 활용 |
| 촬영 주기 | 5분 1회 | burst 5장 연속 촬영 후 선별 |
| 폼팩터 | 21×17.5mm | 안경 프레임 내 탑재 |

> VLM 입력 해상도는 어차피 448×448 수준으로 리사이즈하므로, 카메라에서 최대 해상도로 캡처할 필요 없음.
> 전송 속도 / 저장 용량을 고려해 640×480 권장.

---

## 1단계 — 이미지 전처리

### 1-1. 입력 규격 및 정규화

| 처리 항목 | 내용 |
|---|---|
| 리사이즈 목표 | 448×448 (정방) 또는 672×448 (와이드, Qwen2.5-VL 권장) |
| 색공간 변환 | BGR → RGB |
| 정규화 | mean=[0.485, 0.456, 0.406] / std=[0.229, 0.224, 0.225] (ImageNet 표준) |
| 포맷 | float32 또는 uint8 그대로 VLM에 전달 (모델별 상이) |

---

### 1-2. 흔들림 보정 방법 비교

안경 웨어러블 특성상 5분 간격 촬영 → 각 시점에 burst 5장을 짧게 찍는 구조.
방법마다 트레이드오프가 다르므로 직접 실험 권장. 아래 순서대로 올라가는 것이 효율적.

| 방법 | 원리 | 장점 | 단점 | 난이도 |
|---|---|---|---|---|
| **A. Laplacian 분산 선택** | 5장 중 선명도(Laplacian variance) 가장 높은 1장만 선택 | 구현 단순, 처리 빠름 (~10ms), OpenCV 몇 줄 | 5장 전부 흐릴 경우 그냥 가장 덜 흐린 것 선택. 합성 없음 | 낮음 |
| **B. ECC 정렬 + 평균 합성** | 5장을 ECC(Enhanced Correlation Coefficient)로 픽셀 align 후 평균 | 노이즈 감소, SNR 향상, 부분 흐림 보완 | 장면 내 움직이는 물체에 약함. 처리 시간 수 초 | 중간 |
| **C. 선택적 합성 (A+B 혼합)** | Laplacian으로 상위 2~3장 선별 후, 그것만 ECC 정렬 + 평균 | B의 모션 오염 리스크를 줄이면서 품질 향상 | threshold 파라미터 튜닝 필요 | 중간 |
| **D. Fourier Burst Accumulation** | 5장의 Fourier 스펙트럼을 크기 가중 평균 → 역변환 | 카메라 흔들림에 수학적으로 최적화된 알고리즘. 스마트폰 탑재 목적으로 설계됨 | 구현 복잡, 사전 정렬 필요, 정지 장면 전제 | 높음 |
| **E. 딥러닝 Deblur (NAFNet 등)** | 단일 이미지 deblurring 모델 사용 (NAFNet, DeblurGAN-v2 등) | 가장 강력한 blur 제거 가능 | 추가 VRAM/시간 소모, VLM 전에 별도 모델 추론 필요 | 높음 |

**권장 실험 순서:** A → C → D → E

안경 특성상 대부분 정적 장면 촬영이므로 방법 A 또는 C로 충분할 가능성이 높음.

---

## 2단계 — VLM 캡션 생성

### 2-1. 모델 선택지 (8GB VRAM 기준)

| 모델 | 정밀도 | 예상 VRAM | 물체인식 | 추론 속도 | 비고 |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B | BF16 | ~16GB | 최상 | — | 8GB 불가 |
| Qwen2.5-VL-7B | INT8 (GPTQ) | ~9GB | 최상에 근접 | 보통 | 8GB 빠듯, 시도 가능 |
| **Qwen2.5-VL-7B** | **INT4 (AWQ)** | **~5~6GB** | **상** | **빠름** | **8GB 여유있음 ✓ 1순위** |
| Qwen2.5-VL-7B | INT4 (GPTQ) | ~5~6GB | 상 | AWQ보다 느림 | AWQ 대안 |
| Qwen2.5-VL-7B | GGUF Q4_K_M | ~5GB | 상 | CPU+GPU 혼합 가능 | llama.cpp 기반, VRAM 오프로드 유연 |
| Qwen2.5-VL-7B | GGUF Q8_0 | ~8GB | AWQ 이상 | 느림 | 8GB 경계선 |
| InternVL2-8B | INT4 (AWQ) | ~6GB | 상 | 빠름 | Qwen 대안, 실험 가치 있음 |
| LLaVA-1.6-Mistral-7B | INT4 | ~5GB | 중상 | 빠름 | 범용, 물체인식 특화 아님 |
| PaliGemma2-10B | INT4 | ~7~8GB | 상 | 보통 | OCR/캡셔닝 특화, VRAM 빠듯 |
| Qwen2.5-VL-3B | BF16 | ~7GB | 중상 | 매우 빠름 | VRAM 여유 없을 때 fallback |

**추천 실험 순서:** Qwen2.5-VL-7B AWQ-INT4 → GGUF Q4_K_M → InternVL2-8B AWQ

> GGUF는 VRAM 부족 시 RAM으로 일부 오프로드 가능해서 하드웨어 여건에 따라 더 유연하게 쓸 수 있음.

---

### 2-2. 추론 속도 최적화 옵션

| 옵션 | 효과 | 비고 |
|---|---|---|
| max_new_tokens 제한 | 출력 길이 제한 → 시간 단축 | 256~512 토큰이면 충분 |
| flash_attention_2 | 메모리/속도 동시 개선 | Ampere 이상 GPU (RTX 30xx~) 필요 |
| 입력 해상도 축소 | 비전 토큰 수 감소 → 직접 영향 | 448px vs 672px 비교 필요 |
| vLLM 서빙 | throughput 향상 | batch=1 단일 이미지라면 효과 제한적 |
| torch.compile | 2회차부터 빠름 | 5분 주기 특성상 효과 제한적 |

---

### 2-3. 프롬프트 엔지니어링

이 프로젝트의 목적은 **"나중에 물건을 찾기 위한 기억 보조"** 이므로,
프롬프트를 단순 이미지 설명이 아닌 **물체 식별 + 위치 맥락 추출** 에 최적화해야 한다.

---

#### System Prompt (영어 출력용)

```
You are a memory-aid assistant built into smart glasses.
Your job is to carefully analyze images captured every 5 minutes
and help the user recall where objects are located later.

When analyzing an image, prioritize the following:

1. OBJECT IDENTIFICATION
   - Detect every distinct portable object visible (bags, wallets, keys,
     phones, glasses, chargers, books, clothes, food items, etc.)
   - For each object, describe its identifying visual features:
     color, material, brand (if visible), size estimate, shape, condition
   - Rate your confidence in the identification (high / medium / low)

2. SPATIAL LOCATION
   - Describe where each object is in the frame:
     frame region (top-left / top-center / top-right /
                   center-left / center / center-right /
                   bottom-left / bottom-center / bottom-right)
   - What surface the object is resting on (desk, floor, shelf, sofa, etc.)
   - Approximate depth (foreground / mid-ground / background)

3. CONTEXTUAL SURROUNDINGS
   - List all nearby objects within close proximity of each target object
   - Describe the environment type (living room, kitchen, office, bedroom,
     outdoor, etc.)
   - Note any distinctive environmental features that would help locate
     the object later (next to the window, near the TV, on top of the laptop, etc.)

4. SCENE SUMMARY
   - Write a single sentence summarizing the overall scene in a way
     that would help someone recall this location later.

Be exhaustive and specific. Assume the user will ask "where is my [object]?"
six hours later and needs enough detail to find it.
Output a structured description, one object per paragraph.
```

---

#### User Prompt (영어 출력용)

```
Analyze this image and describe all visible objects in detail.

For each object you find, provide:
- Object name and category
- Visual identifying features (color, material, brand, shape)
- Location in frame (use 9-region grid: top/center/bottom + left/center/right)
- What it is placed on or near
- List of nearby objects within arm's reach
- Any unique characteristics that distinguish this specific object

Also describe:
- The type of space or room visible
- Overall scene context that would help locate objects later

Focus especially on everyday portable items a person might want to find later:
wallets, keys, phones, bags, glasses, chargers, clothing items, food, drinks.
```

---

#### System Prompt (한국어 출력 직접 요청 버전)

```
당신은 스마트 안경에 내장된 기억 보조 AI입니다.
5분마다 자동으로 촬영된 이미지를 분석하여,
사용자가 나중에 "내 지갑 어딨어?", "열쇠 못 찾겠어" 같은 질문을 했을 때
정확한 위치를 알려줄 수 있도록 상세한 정보를 추출합니다.

이미지를 분석할 때 다음 순서로 진행하세요:

1. 물체 식별 (최우선)
   - 이미지에서 보이는 모든 이동 가능한 물체를 식별하세요
     (지갑, 열쇠, 가방, 핸드폰, 충전기, 안경, 책, 옷, 음식, 음료 등 포함)
   - 각 물체의 식별 특징을 구체적으로 묘사하세요:
     색상, 재질, 브랜드(보이는 경우), 크기 추정, 형태, 상태
   - 인식 신뢰도를 평가하세요 (높음 / 보통 / 낮음)

2. 공간적 위치
   - 각 물체가 화면 어느 위치에 있는지 9분할 기준으로 표현하세요
     (좌상단 / 중앙상단 / 우상단 / 좌중앙 / 정중앙 / 우중앙 /
      좌하단 / 중앙하단 / 우하단)
   - 물체가 놓인 표면을 명시하세요 (책상, 바닥, 선반, 소파, 침대 등)
   - 원근감 표현 (전경 / 중경 / 배경)

3. 주변 맥락
   - 각 물체 근처에 있는 다른 물체들을 나열하세요
   - 공간 유형을 추정하세요 (거실, 주방, 사무실, 침실, 실외 등)
   - 위치 기억에 도움이 되는 특징적 환경 묘사
     (창문 옆, TV 앞, 노트북 위 등)

4. 장면 요약
   - 나중에 이 위치를 떠올릴 수 있도록 한 문장으로 전체 장면을 요약하세요

반드시 한국어로 출력하세요.
모든 답변을 물체별로 단락을 나눠 구조적으로 작성하세요.
```

---

#### User Prompt (한국어 출력 직접 요청 버전)

```
이 이미지를 분석하고 보이는 모든 물체를 상세하게 설명해주세요.

각 물체에 대해 다음을 제공해주세요:
- 물체 이름과 종류
- 시각적 식별 특징 (색상, 재질, 브랜드, 형태)
- 화면 내 위치 (9분할 격자 사용)
- 어디에 놓여있는지 (표면 및 주변 환경)
- 주변 한 팔 거리 내 다른 물체 목록
- 이 물체를 다른 유사 물체와 구별할 수 있는 특징

추가로:
- 촬영 공간의 종류 (방 유형 추정)
- 나중에 물체 위치를 기억하는 데 도움이 될 전반적인 장면 맥락

특히 일상적으로 찾게 되는 물체에 집중하세요:
지갑, 열쇠, 핸드폰, 가방, 안경, 충전기, 의류, 음식, 음료

반드시 한국어로 답변하세요.
```

---

#### 영어 캡션 → 한국어 변환 후처리 프롬프트

영어로 캡션을 먼저 생성한 뒤, 별도로 한국어 변환하는 방식.
(영어 캡션을 보존하면서 한국어도 확보하는 하이브리드 전략)

```
다음은 이미지에서 추출된 영어 장면 묘사입니다.
이것을 한국어로 자연스럽게 번역하되, 아래 규칙을 따르세요:

규칙:
1. 물체 이름은 한국어 일반 명칭으로 번역 (예: wallet → 지갑, charger → 충전기)
2. 위치 표현은 한국식으로 자연스럽게 변환 (예: center-left → 왼쪽 중앙)
3. 공간 표현은 한국 주거 문화에 맞게 변환 (예: couch → 소파)
4. 브랜드명, 고유명사는 그대로 유지
5. 번역 과정에서 정보 누락 없이 완전하게 번역

원문:
{english_caption}

한국어 번역:
```

---

#### 프롬프트 전략 선택 가이드

| 상황 | 권장 전략 |
|---|---|
| 추론 속도 우선 | 영어로 캡션 생성 → JSON 정제 시 한국어 변환 (2단계 분리) |
| 품질 우선 | 처음부터 한국어 출력 요청 (system prompt에 한국어 명시) |
| 영문 캡션 보존 필요 | 영어 캡션 생성 후 별도 번역 프롬프트로 후처리 |
| 검색 정확도 우선 | 영어 캡션 + 한국어 번역 둘 다 JSON에 저장 (raw_description_en + raw_description_ko) |

> Qwen2.5-VL은 한국어를 포함한 29개 언어를 직접 지원하므로, 처음부터 한국어 출력을 요청해도 품질 저하가 크지 않음.
> 다만 영어로 먼저 생성하고 번역하는 방식이 일반적으로 물체 명칭 인식 정확도가 약간 더 높다는 보고가 있으므로, 두 방식을 실험해서 비교 권장.

---

## 3단계 — 정제 및 JSON 스키마

### 3-1. JSON 스키마 구조

```json
{
  "capture_id": "uuid-v4",
  "timestamp": "2026-04-03T14:30:00+09:00",
  "image_path": "captures/2026-04-03/img_1430.jpg",
  "sharpness_score": 142.3,
  "scene_summary": "나무 책상 위에 노트북, 지갑, 커피잔이 놓여 있는 실내 공간",
  "location_context": "거실 책상 위",
  "objects": [
    {
      "object_id": 1,
      "name": "지갑",
      "name_en": "wallet",
      "confidence": 0.92,
      "position": {
        "depth_hint": "near",
        "surface": "나무 책상 위"
      },
      "visual_features": {
        "color": "갈색",
        "material": "가죽",
        "brand": null,
        "shape": "직사각형"
      },
      "nearby_objects": ["노트북", "커피잔", "마우스", "볼펜"],
      "raw_description_en": "Brown leather wallet lying flat on a wooden desk, located in the center-left of the frame, near a laptop and a coffee cup.",
      "raw_description_ko": "나무 책상 위 중앙 왼쪽에 갈색 가죽 지갑이 납작하게 놓여 있으며, 노트북과 커피잔 근처에 위치함."
    }
  ],
  "pipeline_meta": {
    "vlm_model": "Qwen2.5-VL-7B-Instruct-AWQ-INT4",
    "inference_time_sec": 87.4,
    "preprocess_time_sec": 1.8,
    "blur_correction_method": "laplacian_select",
    "burst_count": 5,
    "selected_frame_index": 2
  }
}
```

---

### 3-2. 주요 필드 설명

| 필드 | 타입 | 설명 |
|---|---|---|
| capture_id | string (UUID) | 촬영 고유 ID |
| timestamp | ISO 8601 | 촬영 시각 (KST +09:00) |
| image_path | string | 원본 이미지 저장 경로 |
| sharpness_score | float | 선택된 burst 이미지의 Laplacian variance 점수 |
| scene_summary | string | 전체 장면 한 줄 요약 (한국어) |
| location_context | string | 공간 추정 (예: "거실 책상 위") |
| **objects[ ]** | array | 인식된 물체 목록 (핵심 필드) |
| ↳ name | string | 물체명 (한국어) |
| ↳ name_en | string | 물체명 (영어, 검색 정확도용) |
| ↳ confidence | float 0~1 | VLM 인식 신뢰도 |
| ↳ position.frame_region | string | 9분할 프레임 위치 |
| ↳ position.surface | string | 놓인 표면 |
| ↳ visual_features | object | color / material / brand / shape |
| ↳ nearby_objects[ ] | string[] | 주변 물체 목록 (교차 검색 핵심) |
| ↳ raw_description_en | string | VLM 영어 원문 보존 |
| ↳ raw_description_ko | string | 한국어 번역본 |
| pipeline_meta | object | 사용 모델 / 추론 시간 / 전처리 방법 |

> **nearby_objects 배열이 검색에서 매우 중요.**
> "지갑"을 직접 못 찾더라도 "노트북 옆 갈색 물체"로 간접 검색 가능.
> 나중에 벡터 DB (ChromaDB, Weaviate 등)에 넣어 의미 검색을 붙일 때 이 필드가 recall을 크게 높여줌.

---

### 3-3. 정제 방식 선택지

| 방식 | 설명 | 장단점 |
|---|---|---|
| **A. VLM에 직접 JSON 출력 요청** | 캡션과 JSON 생성을 하나의 프롬프트로 처리 | 빠르고 단순 / 형식 오류 가능성 있음 |
| **B. 캡션 → 경량 LLM 파싱** | VLM 자연어 캡션 → Qwen2.5-1.5B 같은 소형 LLM이 JSON 변환 | 안정적 구조화 / 추가 추론 시간 발생 |
| **C. Regex + 규칙 기반 파싱** | 캡션 텍스트를 패턴 매칭으로 분해 | 빠름 / VLM 출력 포맷 의존성 높음 |

---

## 전체 파이프라인 흐름 요약

| 단계 | 처리 내용 | 예상 시간 |
|---|---|---|
| 촬영 | XIAO ESP32S3에서 5장 burst (640×480) | ~5초 |
| 전처리 | 흔들림 보정 → 1장 선택/합성 → 리사이즈 → 정규화 | ~1~3초 |
| WiFi 전송 | 이미지 → 서버 (선택: SD카드 저장 병행 가능) | ~2~5초 |
| VLM 추론 | Qwen2.5-VL-7B-INT4 or GGUF → 캡션 생성 | ~60~150초 |
| 한국어 변환 | 영어 캡션 → 한국어 번역 (또는 직접 한국어 출력) | ~5~20초 |
| JSON 정제 | 캡션 → 구조화 JSON 변환 | ~5~30초 |
| 저장 | JSON + 이미지 → DB 또는 파일 시스템 | ~1초 |
| **총합** | | **~2~3분 (4분 이내 ✓)** |

---

## 추후 확장 고려사항

- **검색 기능 연동:** 생성된 JSON을 ChromaDB / Weaviate 같은 벡터 DB에 적재 후, "내 지갑 어딨어?" 질의를 임베딩해서 유사도 검색
- **시간 기반 필터:** "아까 점심 때 어딨었어?" → timestamp 기반 필터링 후 검색
- **다국어 처리:** raw_description_en과 raw_description_ko를 모두 저장해두면 검색 쿼리 언어에 무관하게 대응 가능
- **fine-tuning 가능성:** 촬영 데이터가 쌓이면 자주 등장하는 본인의 물건(본인 지갑, 특정 가방 등)에 대해 LoRA fine-tuning으로 인식 정확도 향상 가능
