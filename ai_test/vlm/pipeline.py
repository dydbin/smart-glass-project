import json
import re
import os
import cv2
from PIL import Image
import torch
import uuid
import time
from datetime import datetime
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()


class VLMPipeline:

    # ──────────────────────────────────────────────
    # 1단계: 물체 목록만 추출
    # ──────────────────────────────────────────────
    SYSTEM_PROMPT_LIST = """You are a helpful assistant. Always respond in Korean. Output valid JSON only. No code blocks."""

    USER_PROMPT_LIST = """이미지에서 보이는 물체를 빠짐없이 모두 나열하세요.

규칙:
- 작거나 부분만 보여도 포함
- 브랜드 제품은 브랜드명 포함 (예: AirPods, 맥북, 아이패드, Apple Pencil)
- 태블릿/스마트패드/아이패드도 반드시 포함
- 같은 종류가 여러 개면 모두 포함 (예: AirPods 케이스가 2개면 "AirPods 케이스 1", "AirPods 케이스 2")
- 가구/벽/바닥 제외
- JSON 배열만 출력

["물체1", "물체2", ...]"""

    # ──────────────────────────────────────────────
    # 2단계: 물체별 상세 정보 추출
    # ──────────────────────────────────────────────
    SYSTEM_PROMPT_DETAIL = """You are a helpful assistant. Always respond in Korean. Output valid JSON only. No code blocks."""

    USER_PROMPT_DETAIL = """이미지에서 "{object_name}" 하나에 대해서만 아래 JSON 형식으로 출력하세요.
설명 없이 JSON만 출력하세요.

{{
  "name": "{object_name}",
  "confidence": 0.0,
  "position": {{
    "depth_hint": "near/mid/far",
    "surface": "놓인 표면 또는 기준 물체. 예: 맥북 위, 아이폰 왼쪽 테이블, 테이블 위 왼쪽"
  }},
  "visual_features": {{
    "color": "색상",
    "material": "재질",
    "brand": null,
    "shape": "형태"
  }},
  "nearby_objects": ["주변 물체1", "주변 물체2"],
  "raw_description_ko": "한 문장 묘사"
}}

규칙:
- JSON만 출력
- confidence는 0.0~1.0 숫자
- brand는 문자열 또는 null
- nearby_objects는 문자열 배열
- surface는 절대 비워두지 말 것
- 반드시 한국어로"""

    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.is_7b = "7B" in model_id or "7b" in model_id

    # ──────────────────────────────────────────────
    # VRAM 유틸
    # ──────────────────────────────────────────────
    def get_vram_usage(self):
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gb":     round(torch.cuda.memory_allocated() / 1024**3, 2),
            "reserved_gb":      round(torch.cuda.memory_reserved()  / 1024**3, 2),
            "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "total_gb":         round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        }

    def reset_vram_peak(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    # ──────────────────────────────────────────────
    # 선명도
    # ──────────────────────────────────────────────
    def calculate_sharpness(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    # ──────────────────────────────────────────────
    # 모델 로드
    # ──────────────────────────────────────────────
    def load_model(self):
        if self.model is not None:
            return

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        print(f"Loading model: {self.model_id}")
        vram_before = self.get_vram_usage()
        print(f"  VRAM before load: {vram_before.get('allocated_gb', 0):.2f}GB / {vram_before.get('total_gb', 0):.2f}GB")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        max_pixels = 640 * 480 if self.is_7b else 320 * 320
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            max_pixels=max_pixels
        )

        vram_after = self.get_vram_usage()
        print(f"  VRAM after load:  {vram_after.get('allocated_gb', 0):.2f}GB / {vram_after.get('total_gb', 0):.2f}GB")
        print("Model loaded.")

    # ──────────────────────────────────────────────
    # nearby fallback: 후보 수집
    # ──────────────────────────────────────────────
    def is_noisy_name(self, name):
        """언더스코어, 순수 영어 소문자 등 노이즈성 이름 판별"""
        if "_" in name:
            return True
        # 완전히 영어 소문자로만 구성된 경우 (macbook, airpods, iphone 등)
        if name == name.lower() and name.isascii() and name.replace(" ", "").isalpha():
            return True
        return False

    def collect_nearby_candidates(self, objects):
        """nearby_objects에 언급된 물체 중 main list에 없는 것들 수집
        노이즈성 이름(언더스코어, 순수영어소문자)은 제외"""
        main_names = {o["name"] for o in objects}
        candidates = []
        seen = set()
        for obj in objects:
            for nearby in obj.get("nearby_objects", []):
                if nearby not in main_names and nearby not in seen:
                    if not self.is_noisy_name(nearby):
                        candidates.append(nearby)
                        seen.add(nearby)
        return candidates

    # ──────────────────────────────────────────────
    # VLM 기반 중복 제거
    # ──────────────────────────────────────────────
    def deduplicate_with_vlm(self, image, object_list):
        """모델에게 목록을 주고 같은 물체 합쳐달라고 요청"""
        if not object_list:
            return []

        # 노이즈 이름 사전 제거 (언더스코어, 순수영어소문자)
        clean_list = [x for x in object_list if not self.is_noisy_name(x)]
        if not clean_list:
            return object_list
        if len(clean_list) < len(object_list):
            print(f"  [dedup] 노이즈 이름 제거: {set(object_list) - set(clean_list)}")

        prompt = f"""다음은 이미지에서 감지된 물체 목록입니다:
{json.dumps(clean_list, ensure_ascii=False)}

규칙:
- 진짜 같은 물체를 가리키는 항목만 하나로 합칠 것 (예: "맥북"과 "노트북"이 동일한 물체면 하나만)
- 표기만 다른 동일 물체도 합칠 것 (예: "AirPods 케이스"와 "AirPods 캡"이 같은 물체면 하나만)
- 가장 구체적인 이름 사용 (예: "노트북"보다 "맥북" 선택)
- 같은 종류라도 이미지에서 위치가 다른 별개 물체면 반드시 둘 다 유지
- 확실하지 않으면 제거하지 말고 유지할 것
- 출력 이름은 반드시 위 입력 목록에 있는 이름 그대로 사용할 것 (새 이름 만들지 말 것)
- JSON 배열만 출력

["물체1", "물체2", ...]"""

        print("  [dedup] VLM 중복 제거 중...")
        raw = self._infer(image, self.SYSTEM_PROMPT_LIST, prompt, max_new_tokens=256)
        print(f"  [dedup] raw: {raw.strip()}")
        result = self.extract_json(raw, expect_array=True)
        deduped = [x for x in result if isinstance(x, str) and x.strip()]

        # 입력 목록에 없는 이름 제거 (환각 방지)
        input_set = set(clean_list)
        deduped = [x for x in deduped if x in input_set]

        # 결과가 원본보다 절반 이상 줄었으면 잘린 것으로 판단하고 원본 반환
        if not deduped or len(deduped) < len(clean_list) // 2:
            print(f"  [dedup] 결과 비정상 (원본 {len(clean_list)}개 → {len(deduped)}개) → 원본 유지")
            return clean_list
        return deduped

    # ──────────────────────────────────────────────
    # 브랜드 자동 추론
    # ──────────────────────────────────────────────
    BRAND_MAP = {
        "airpods": "Apple", "아이폰": "Apple", "맥북": "Apple",
        "macbook": "Apple", "iphone": "Apple", "아이패드": "Apple",
        "ipad": "Apple", "magsafe": "Apple", "apple pencil": "Apple",
        "갤럭시": "Samsung", "galaxy": "Samsung", "버즈": "Samsung",
        "buds": "Samsung", "갤럭시탭": "Samsung",
    }

    def infer_brand_from_name(self, name):
        name_lower = name.lower()
        for keyword, brand in self.BRAND_MAP.items():
            if keyword in name_lower:
                return brand
        return None

    # ──────────────────────────────────────────────
    # 누락된 대형 물체 체크
    # ──────────────────────────────────────────────
    def check_missing_objects(self, image, current_objects):
        """현재 인식된 목록을 보여주고 놓친 물체가 있는지 VLM에게 확인"""
        current_names = [o["name"] for o in current_objects]
        prompt = f"""현재 이미지에서 인식된 물체 목록입니다:
{json.dumps(current_names, ensure_ascii=False)}

이미지를 다시 확인하고, 위 목록에서 빠진 물체가 있으면 추가해서 전체 목록을 JSON 배열로 출력하세요.
특히 태블릿, 아이패드, 스타일러스 펜, Apple Pencil 같은 물체가 있는지 확인하세요.
빠진 것이 없으면 그대로 출력하세요.

["물체1", "물체2", ...]"""

        print("  [누락체크] 대형 물체 누락 확인 중...")
        raw = self._infer(image, self.SYSTEM_PROMPT_LIST, prompt, max_new_tokens=256)
        print(f"  [누락체크] raw: {raw.strip()}")
        result = self.extract_json(raw, expect_array=True)
        updated = [x for x in result if isinstance(x, str) and x.strip() and not self.is_noisy_name(x)]
        if not updated:
            # 노이즈만 나왔거나 아무것도 없으면 → 새로 발견된 물체 없음
            print(f"  [누락체크] 새로 발견된 물체: []")
            return []
        # 새로 추가된 것만 반환 (current_names에 없는 것)
        added = [x for x in updated if x not in current_names]
        print(f"  [누락체크] 새로 발견된 물체: {added}")
        return added
    def estimate_confidence(self, obj):
        """채워진 필드 수 기반 정보 충실도를 proxy confidence로 사용"""
        score = 0.5
        surface = obj.get("position", {}).get("surface", "")
        if surface and surface != "unknown" and surface != "테이블 위":
            # "맥북 위", "아이폰 왼쪽" 처럼 구체적인 surface면 가산
            score += 0.2
        if obj.get("visual_features", {}).get("brand"):
            score += 0.1
        if len(obj.get("nearby_objects", [])) >= 2:
            score += 0.2
        return round(min(score, 1.0), 2)

    # ──────────────────────────────────────────────
    # 공통 추론 함수
    # ──────────────────────────────────────────────
    def _infer(self, image, system_prompt, user_prompt, max_new_tokens=256):
        from qwen_vl_utils import process_vision_info

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": user_prompt}
            ]}
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.3
            )

        decoded = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        return decoded

    # ──────────────────────────────────────────────
    # JSON 추출
    # ──────────────────────────────────────────────
    def extract_json(self, text, expect_array=False):
        from json_repair import repair_json

        text = text.replace("```json", "").replace("```", "").strip()

        if expect_array:
            try:
                start = text.find("[")
                end = text.rfind("]")
                if start != -1 and end != -1:
                    candidate = text[start:end+1]
                    repaired = repair_json(candidate)
                    return json.loads(repaired)
            except:
                pass
            return []
        else:
            try:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    candidate = text[start:end+1]
                    json.loads(candidate)
                    return candidate
            except:
                pass
            try:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    candidate = text[start:end+1]
                    repaired = repair_json(candidate)
                    json.loads(repaired)
                    return repaired
            except:
                pass
            return None

    # ──────────────────────────────────────────────
    # 후처리 정규화
    # ──────────────────────────────────────────────
    def normalize_confidence(self, value):
        if isinstance(value, (int, float)):
            return round(float(value), 2)
        if isinstance(value, str):
            cleaned = re.sub(r"[^\d.]", "", value)
            try:
                f = float(cleaned)
                return round(f / 100 if f > 1 else f, 2)
            except:
                return 0.0
        return 0.0

    def normalize_brand(self, value):
        if value is None or value == {} or value == []:
            return None
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and len(value) > 0:
            return str(value[0])
        if isinstance(value, dict) and len(value) > 0:
            return list(value.keys())[0]
        return None

    def normalize_nearby_objects(self, value):
        if not isinstance(value, list):
            return []
        result = []
        seen = set()
        for item in value:
            if isinstance(item, list):
                for sub in item:
                    if isinstance(sub, str) and sub.strip() and sub.strip() not in seen:
                        result.append(sub.strip())
                        seen.add(sub.strip())
            elif isinstance(item, str) and item.strip() and item.strip() not in seen:
                result.append(item.strip())
                seen.add(item.strip())
        return result

    def normalize_position(self, position):
        if not isinstance(position, dict):
            return {"depth_hint": "unknown", "surface": "unknown"}
        clean = {}
        valid_keys = {"depth_hint", "surface"}
        for k, v in position.items():
            is_clean_key = all(ord(c) < 128 for c in k) and "\\" not in k
            if is_clean_key and k in valid_keys:
                clean[k] = v if isinstance(v, str) else "unknown"
        for key in valid_keys:
            if key not in clean:
                clean[key] = "unknown"
        return clean

    def normalize_location_context(self, value):
        """location_context가 dict/list로 깨져서 들어와도 string으로 정규화"""
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            # {"type_of_space": ["카페"]} 같은 형태 처리
            v = list(value.values())[0] if value else "알 수 없음"
            if isinstance(v, list):
                return v[0] if v else "알 수 없음"
            return str(v)
        if isinstance(value, list):
            return value[0] if value else "알 수 없음"
        return "알 수 없음"

    def normalize_object(self, obj, object_id, expected_name=None):
        if not isinstance(obj, dict):
            return None
        obj["object_id"]      = object_id
        obj["position"]       = self.normalize_position(obj.get("position", {}))
        obj["nearby_objects"] = self.normalize_nearby_objects(obj.get("nearby_objects", []))
        # name_en 필드 제거
        obj.pop("name_en", None)
        # 모델이 name을 노이즈 형태로 바꿨으면 원래 이름으로 복원
        if expected_name and self.is_noisy_name(obj.get("name", "")):
            print(f"  [정규화] name 복원: '{obj['name']}' → '{expected_name}'")
            obj["name"] = expected_name
        vf = obj.get("visual_features", {})
        if isinstance(vf, dict):
            vf["brand"] = self.normalize_brand(vf.get("brand"))
            if vf["brand"] is None:
                vf["brand"] = self.infer_brand_from_name(obj.get("name", ""))
            obj["visual_features"] = vf
        else:
            obj["visual_features"] = {
                "color": "unknown", "material": "unknown",
                "brand": self.infer_brand_from_name(obj.get("name", "")),
                "shape": "unknown"
            }
        # brand 채운 후 confidence 계산 (순서 중요)
        obj["confidence"] = self.estimate_confidence(obj)
        return obj

    # ──────────────────────────────────────────────
    # LLM 질의용 경량 컨텍스트 생성
    # ──────────────────────────────────────────────
    def to_query_context(self, result):
        """LLM에 넘길 경량화된 컨텍스트. 메타데이터 제거, 핵심만 유지."""
        objects_summary = []
        for obj in result["objects"]:
            objects_summary.append({
                "name":    obj["name"],
                "surface": obj["position"]["surface"],
                "nearby":  obj["nearby_objects"],
                "color":   obj["visual_features"]["color"],
                "brand":   obj["visual_features"]["brand"],
            })
        return {
            "timestamp":       result["timestamp"],
            "location":        result["location_context"],
            "scene":           result["scene_summary"],
            "objects":         objects_summary,
        }

    # ──────────────────────────────────────────────
    # 1단계: 물체 목록 추출
    # ──────────────────────────────────────────────
    def get_object_list(self, image):
        print("  [1단계] 물체 목록 추출 중...")
        raw = self._infer(
            image,
            self.SYSTEM_PROMPT_LIST,
            self.USER_PROMPT_LIST,
            max_new_tokens=256
        )
        print(f"  [1단계] raw: {raw.strip()}")
        result = self.extract_json(raw, expect_array=True)
        # 노이즈 이름(언더스코어, 순수영어소문자) 제외
        return [x for x in result if isinstance(x, str) and x.strip() and not self.is_noisy_name(x)]

    # ──────────────────────────────────────────────
    # 2단계: 물체별 상세 추출
    # ──────────────────────────────────────────────
    def get_object_detail(self, image, object_name, object_id):
        prompt = self.USER_PROMPT_DETAIL.format(object_name=object_name)
        raw = self._infer(
            image,
            self.SYSTEM_PROMPT_DETAIL,
            prompt,
            max_new_tokens=256
        )
        cleaned = self.extract_json(raw, expect_array=False)
        if cleaned is None:
            print(f"  [2단계] '{object_name}' 파싱 실패 → raw: {raw.strip()[:80]}")
            return None
        try:
            obj = json.loads(cleaned)
            return self.normalize_object(obj, object_id, expected_name=object_name)
        except:
            return None

    # ──────────────────────────────────────────────
    # 장면 요약
    # ──────────────────────────────────────────────
    def get_scene_summary(self, image):
        prompt = """이미지의 전체 장면을 한 문장으로 요약하고 공간 유형을 JSON으로 출력하세요.

{
  "scene_summary": "10단어 이내 요약",
  "location_context": "공간 유형 (예: 거실, 카페, 사무실)"
}"""
        raw = self._infer(image, "You are a helpful assistant. Output valid JSON only.", prompt, max_new_tokens=256)
        cleaned = self.extract_json(raw, expect_array=False)
        if cleaned:
            try:
                return json.loads(cleaned)
            except:
                pass
        return {"scene_summary": "알 수 없음", "location_context": "알 수 없음"}

    # ──────────────────────────────────────────────
    # 단일 이미지 실행
    # ──────────────────────────────────────────────
    def run_single(self, image_path):
        self.load_model()
        torch.cuda.empty_cache()
        self.reset_vram_peak()

        sharpness = self.calculate_sharpness(image_path)

        resize_to = (640, 480) if self.is_7b else (320, 320)
        image = Image.open(image_path).convert("RGB")
        # resize 없이 원본 해상도 유지 → processor의 max_pixels가 자동 조정
        # 단, 메모리 초과 방지를 위해 4K 이상이면 절반으로만 축소
        w, h = image.size
        if w * h > 3840 * 2160:
            image = image.resize((w // 2, h // 2), Image.LANCZOS)
            print(f"  [이미지] 4K 초과로 절반 축소: {w}x{h} → {w//2}x{h//2}")
        else:
            print(f"  [이미지] 원본 해상도 유지: {w}x{h}")

        start_time = time.time()

        # 1단계: 물체 목록
        object_list = self.get_object_list(image)
        print(f"  [1단계] 인식된 물체: {object_list}")

        # 장면 요약
        scene_info = self.get_scene_summary(image)
        print(f"  [장면] {scene_info}")

        # 2단계: 물체별 상세
        objects = []
        for i, obj_name in enumerate(object_list, start=1):
            print(f"  [2단계] ({i}/{len(object_list)}) '{obj_name}' 상세 추출 중...")
            detail = self.get_object_detail(image, obj_name, i)
            if detail:
                objects.append(detail)
                print(f"  [2단계] '{obj_name}' 완료")
            else:
                print(f"  [2단계] '{obj_name}' 실패 → 스킵")

        # nearby 후보 수집 → 전체 후보 목록 구성 → VLM dedup
        nearby_candidates = self.collect_nearby_candidates(objects)
        all_candidates = [o["name"] for o in objects] + nearby_candidates
        deduped_list = self.deduplicate_with_vlm(image, all_candidates)
        print(f"  [dedup] 최종 물체 목록: {deduped_list}")

        # dedup 결과 기준으로 없는 물체만 추가 추출
        existing_names = {o["name"] for o in objects}
        for obj_name in deduped_list:
            if obj_name not in existing_names:
                print(f"  [보완] '{obj_name}' 상세 추출 중...")
                next_id = len(objects) + 1
                detail = self.get_object_detail(image, obj_name, next_id)
                if detail:
                    objects.append(detail)
                    print(f"  [보완] '{obj_name}' 추가 완료")

        # dedup에서 제거된 물체 필터링
        deduped_set = set(deduped_list)
        objects = [o for o in objects if o["name"] in deduped_set]

        # 누락된 대형 물체 체크 (아이패드 등)
        missed_large = self.check_missing_objects(image, objects)
        for obj_name in missed_large:
            print(f"  [누락보완] '{obj_name}' 상세 추출 중...")
            next_id = len(objects) + 1
            detail = self.get_object_detail(image, obj_name, next_id)
            if detail:
                objects.append(detail)
                print(f"  [누락보완] '{obj_name}' 추가 완료")

        # object_id 재정렬
        for i, obj in enumerate(objects, start=1):
            obj["object_id"] = i

        inference_time = round(time.time() - start_time, 3)
        vram_info = self.get_vram_usage()
        print(f"  VRAM peak: {vram_info.get('max_allocated_gb', 0):.2f}GB")

        result = {
            "capture_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "sharpness_score": round(sharpness, 2),
            "inference_time": inference_time,
            "scene_summary": scene_info.get("scene_summary", "알 수 없음"),
            "location_context": self.normalize_location_context(
                scene_info.get("location_context", "알 수 없음")
            ),
            "objects": objects,
            "pipeline_meta": {
                "vlm_model": self.model_id,
                "object_count": len(objects),
                "vram_allocated_gb":  vram_info.get("allocated_gb"),
                "vram_peak_gb":       vram_info.get("max_allocated_gb"),
                "vram_total_gb":      vram_info.get("total_gb"),
            }
        }

        # JSON 저장
        os.makedirs("output", exist_ok=True)
        output_filename = f"output/{result['capture_id']}.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[저장 완료] {output_filename}")

        # LLM 질의용 경량 컨텍스트 별도 저장
        query_context = self.to_query_context(result)
        query_filename = f"output/{result['capture_id']}_query.json"
        with open(query_filename, "w", encoding="utf-8") as f:
            json.dump(query_context, f, indent=2, ensure_ascii=False)
        print(f"[질의용 저장] {query_filename}")

        return result


if __name__ == "__main__":
    pipeline = VLMPipeline(model_id="Qwen/Qwen2.5-VL-7B-Instruct")
    result = pipeline.run_single("ai_test/vlm/esp640x480.jfif")
    print(json.dumps(result, indent=2, ensure_ascii=False))