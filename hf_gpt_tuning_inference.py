# ============================================================
# LoRA FINE-TUNED GPT-2 INFERENCE
# Purpose: Run inference using base GPT-2 + trained LoRA adapter
# The LoRA adapter was trained on sales conversation data
# ============================================================

# AutoTokenizer: Converts text <-> token IDs
# AutoModelForCausalLM: Loads causal language model (next-token prediction)
# StoppingCriteria: Custom logic to stop text generation at specific tokens
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
# PeftModel: Loads LoRA adapter and merges it with base model
from peft import PeftModel

# ============================================================
# 1. LOAD BASE MODEL + LoRA ADAPTER
# ============================================================
# Path to locally downloaded GPT-2 model (124M parameters)
model_id = '/Users/dmitri/Documents/LLM/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e'
# Load the original pre-trained GPT-2 (frozen weights)
# device_map="auto" = auto-place on best device (MPS on Mac, CUDA on GPU, CPU as fallback)
base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
# Load tokenizer - converts text to token IDs and back
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load LoRA adapter on top of base model
# This adds the small trained LoRA weights (~few MB) to the frozen base model
# "./lora_adapter" contains: adapter_config.json + adapter_model.safetensors
# Alternative: load from checkpoint: PeftModel.from_pretrained(base_model, "./lora_output/checkpoint-500")
model = PeftModel.from_pretrained(base_model, "./lora_adapter")
# Set to evaluation mode - disables dropout layers for consistent output
# Training mode uses dropout for regularization, inference doesn't need it
model.eval()

# ============================================================
# 2. STOPPING CRITERIA
# ============================================================
# Custom class to stop generation when model outputs "Customer:" token sequence
# Without this, model keeps generating entire multi-turn conversations endlessly
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids  # List of token ID sequences to stop at

    def __call__(self, input_ids, scores, **kwargs):
        # Check if last generated tokens match any stop sequence
        for stop_id in self.stop_token_ids:
            if input_ids[0][-len(stop_id):].tolist() == stop_id:
                return True  # Stop generation
        return False  # Continue generation

# Define stop strings - stop when model starts generating next customer turn
stop_strings = ["\n\nCustomer:", "\nCustomer:"]
# Convert stop strings to token IDs (what the model actually sees internally)
stop_token_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]
# Wrap in StoppingCriteriaList (required format for generate())
stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

# ============================================================
# 3. INFERENCE FUNCTION
# ============================================================
def get_sales_response(customer_message, conversation_history=""):
    """
    Get salesman response for customer message

    Args:
        customer_message: Latest customer message
        conversation_history: Previous conversation turns (optional)
    """
    # Build prompt matching training data format
    # CRITICAL: Prompt format MUST exactly match how the model was trained
    # Training data format: "You are a in the role of a Salesman..."
    # Mismatch = bad/random output because model doesn't recognize the pattern
    if conversation_history:
        prompt = f"""You are a in the role of a Salesman. Here is a conversation:
                {conversation_history} Customer: {customer_message}

                Answer as a Salesman to the previous Statement to convince the person to buy the product or service.
                Salesman:"""
    else:
        prompt = f"""You are a in the role of a Salesman. Here is a conversation:
                Customer: {customer_message}

                Answer as a Salesman to the previous Statement to convince the person to buy the product or service.
                Salesman:"""

    # Tokenize prompt and move to same device as model (MPS/GPU/CPU)
    # return_tensors="pt" = return PyTorch tensors (required for model input)
    # .to(model.device) = move tensors to same device as model to avoid device mismatch error
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response using the LoRA-enhanced model
    outputs = model.generate(
        **inputs,               # Unpack input_ids and attention_mask

        max_new_tokens=200,     # Maximum tokens to generate in response
                                # Higher = longer responses but slower
                                # Tuning: 50 (short) | 100 (medium) | 200 (long) | 500 (very long)

        temperature=0.8,        # Controls randomness of output
                                # Lower = more deterministic/focused, Higher = more creative/random
                                # Tuning: 0.1 (very focused) | 0.5 (balanced) | 0.8 (creative) | 1.0 (random) | 1.5+ (chaotic)

        top_p=0.95,             # Nucleus sampling - only consider tokens within top P cumulative probability
                                # Filters out unlikely/nonsensical tokens. Works together with temperature.
                                # Tuning: 0.5 (conservative) | 0.9 (standard) | 0.95 (diverse) | 1.0 (no filtering)

        do_sample=True,         # Enable sampling (random selection based on probabilities)
                                # False = greedy decoding (always pick highest prob token - deterministic, same output every time)
                                # True = sample from distribution (varied, more natural responses)

        repetition_penalty=1.1, # Penalize tokens that already appeared, reducing repetitive text
                                # 1.0 = no penalty | 1.1 (light) | 1.3 (moderate) | 1.5 (heavy)
                                # Too high = incoherent output as model avoids common words

        stopping_criteria=stopping_criteria,  # Stop when "Customer:" is generated
    )

    # Decode token IDs back to human-readable text
    # skip_special_tokens=True = remove <|endoftext|> and other special tokens from output
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the new generated text (remove the input prompt)
    salesman_response = full_response[len(prompt):].strip()

    # Safety cleanup: remove any trailing "Customer:" turn if stopping criteria missed it
    if "\nCustomer:" in salesman_response:
        salesman_response = salesman_response.split("\nCustomer:")[0].strip()

    return salesman_response

# ============================================================
# 4. TEST EXAMPLES
# ============================================================
# Single-turn: one question, one answer
print("=== Single-turn example ===")
question = "Hi, Im interested in purchasing a new smartphone. Can you help me choose the best one?"
print(f"Human: {question}")
response = get_sales_response(question)
print(f"Salesman: {response}\n")

# Multi-turn: continuing a conversation with previous context
print("=== Multi-turn example ===")
history = "Human: I need a laptop for work. Salesman: Great! What kind of work will you be doing?"
question = "Mostly data analysis and some light video editing."
print(f"{history}\nHuman: {question}")
response = get_sales_response(question, conversation_history=history)
print(f"Salesman: {response}\n")

# ============================================================
# 5. INTERACTIVE CHAT LOOP
# ============================================================
print("=== Interactive mode ===")
print("Type your customer message (or 'quit' to exit):\n")
conversation = ""  # Accumulates conversation history for multi-turn context
while True:
    customer_msg = input("Customer: ")
    if customer_msg.lower() == 'quit':
        break

    response = get_sales_response(customer_msg, conversation_history=conversation)
    print(f"Salesman: {response}\n")

    # Append current turn to conversation history for next round
    # This gives the model context of all previous turns in the conversation
    # Note: GPT-2 max context = 1024 tokens, long conversations will be truncated
    if conversation:
        conversation += f" Customer: {customer_msg} Salesman: {response}"
    else:
        conversation = f"Customer: {customer_msg} Salesman: {response}"
