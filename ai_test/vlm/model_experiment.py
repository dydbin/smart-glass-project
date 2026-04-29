import argparse
import torch
import time
import os
import json
from PIL import Image
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

def get_args():
    parser = argparse.ArgumentParser(description="VLM Model Experimentation Script")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", 
                        help="Model ID to test (e.g., Qwen/Qwen2.5-VL-7B-Instruct, OpenGVLab/InternVL2-8B)")
    parser.add_argument("--image", type=str, required=True, help="Path to the test image")
    parser.add_argument("--quant", type=str, choices=["awq", "int4", "fp16", "bf16"], default="awq",
                        help="Quantization/Precision type")
    return parser.parse_args()

def benchmark_model(model_id, image_path, quant):
    print(f"\n{'='*50}")
    print(f"Benchmarking Model: {model_id}")
    print(f"Quantization: {quant}")
    print(f"{'='*50}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")

    from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering, BitsAndBytesConfig
    
    # Set dtype and quantization config
    if quant in ["4bit", "int4", "awq"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        torch_dtype = torch.float16
    elif quant == "bf16":
        bnb_config = None
        torch_dtype = torch.bfloat16
    else:
        bnb_config = None
        torch_dtype = torch.float16

    start_time = time.time()
    
    try:
        if "Qwen2.5-VL" in model_id:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForVisualQuestionAnswering.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True
            )
        
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds.")

        # VRAM Usage after load
        if device == "cuda":
            vram_used = torch.cuda.memory_allocated() / (1024**3)
            print(f"VRAM Used (after load): {vram_used:.2f} GB")

        # Prepare input
        image = Image.open(image_path).convert("RGB")
        
        # Simple Prompt (Reflecting the plan)
        prompt = "이미지의 물체들을 상세하게 설명하고 JSON 형식으로 핵심 정보를 정리해주세요."
        
        # This is a simplified inference loop for benchmarking
        # Actual implementation details (like template) vary by model
        query = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n<|image_pad|>{prompt}\n<|assistant|>\n"
        
        # Placeholder for actual processing which is model-dependent
        print("Starting inference...")
        inf_start = time.time()
        
        # ... logic for processor(images=image, text=query) ...
        # ... logic for model.generate() ...
        
        # Mock output for now to demonstrate the benchmark flow
        output_text = "인식 결과: 책상, 노트북, 지갑 등이 보입니다."
        
        inf_time = time.time() - inf_start
        print(f"Inference completed in {inf_time:.2f} seconds.")
        
        if device == "cuda":
            vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"Peak VRAM Usage: {vram_peak:.2f} GB")

        print("\n--- Model Output ---")
        print(output_text)

    except Exception as e:
        print(f"Error during benchmarking: {e}")

if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
    else:
        benchmark_model(args.model, args.image, args.quant)
