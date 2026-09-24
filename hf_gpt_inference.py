from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

# Load base model only (no LoRA)
model_id = '/Users/dmitri/Documents/LLM/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e'
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Stopping criteria - stop at "Salesman:" completion
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_token_ids:
            if input_ids[0][-len(stop_id):].tolist() == stop_id:
                return True
        return False

stop_strings = ["\n\nCustomer:", "\nCustomer:"]
stop_token_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]
stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

# Format prompt like training data
def get_sales_response(customer_message, conversation_history=""):
    """
    Get salesman response for customer message

    Args:
        customer_message: Latest customer message
        conversation_history: Previous conversation turns (optional)
    """
    # Build prompt matching training format
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

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.1,
        stopping_criteria=stopping_criteria,
    )

    # Extract response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    salesman_response = full_response[len(prompt):].strip()

    # Clean up
    if "\nCustomer:" in salesman_response:
        salesman_response = salesman_response.split("\nCustomer:")[0].strip()

    return salesman_response

# Test examples
print("=== Single-turn example ===")
question = "Hi, Im interested in purchasing a new smartphone. Can you help me choose the best one?"
response = get_sales_response(question)
print(f"Salesman: {response}\n")

print("=== Multi-turn example ===")
history = "Customer: I need a laptop for work. Salesman: Great! What kind of work will you be doing?"
question = "Mostly data analysis and some light video editing."
response = get_sales_response(question, conversation_history=history)
print(f"Salesman: {response}\n")

print("=== Interactive mode ===")
print("Type your customer message (or 'quit' to exit):\n")
conversation = ""
while True:
    customer_msg = input("Customer: ")
    if customer_msg.lower() == 'quit':
        break

    response = get_sales_response(customer_msg, conversation_history=conversation)
    print(f"Salesman: {response}\n")

    # Update history
    if conversation:
        conversation += f" Customer: {customer_msg} Salesman: {response}"
    else:
        conversation = f"Customer: {customer_msg} Salesman: {response}"


# model_id = '/Users/dmitri/Documents/LLM/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e'
# generator = pipeline('text-generation', model=model_id)
# set_seed(42)
# question = "Hi, Im interested in investing in the stock market. Can you help me?"
# result = generator(question, max_length=30, num_return_sequences=5)
# message_text = result[len(result)-1]['generated_text']
# print(message_text)