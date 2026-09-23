import transformers
import torch
from datetime import datetime

model_id = f"/Users/dmitri/Documents/LLM/models--meta-llama--Llama-2-13b-chat-hf/snapshots/a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"

now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"[{now_str}] start loading model {model_id}...")
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"[{now_str}] end loading model {model_id}...\n")

def call_ai(question: str):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] start inference model {model_id}...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] end inference model {model_id}...\n")

if __name__ == "__main__":
    call_ai("Explain the theory of relativity in simple terms using max 200 tokens.")
    call_ai("What is the capital of France?")
    call_ai("Write a short poem about the sea.")