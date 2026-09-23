from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import torch
import gc

model_id = f"/Users/dmitri/Documents/LLM/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"

now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"[{now_str}] start loading model {model_id}...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    local_files_only=True,
    dtype=dtype,
    low_cpu_mem_usage=True,
)
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"[{now_str}] end loading model {model_id}...\n")

now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"[{now_str}] start inference model {model_id}...")
prompt = "Give me a short introduction to large language model with 200 tokens ."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"[{now_str}] end inference model {model_id}...\n")