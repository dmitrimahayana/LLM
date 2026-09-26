# transformers: Hugging Face library for loading/training LLMs
# datasets: Library for loading datasets from Hugging Face Hub
# peft: Parameter-Efficient Fine-Tuning library (LoRA, etc.)
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
import json
from datetime import datetime
import os


# ============================================================
# 1. LOAD DATASET
# ============================================================
# Load sales conversation dataset from Hugging Face Hub
# split="train" = use the training split (not test/validation)
dataset_name = "goendalf666/sales-conversations-instruction-ext"
train_dataset = load_dataset(dataset_name, split="train[:90%]")
eval_dataset = load_dataset(dataset_name, split="train[90%:]")
# Print dataset size to verify how many samples we have (20,940 total)
print(f"Training Dataset size: {len(train_dataset)}")
print(f"Evaluation Dataset size: {len(eval_dataset)}")

# ============================================================
# 2. LOAD BASE MODEL & TOKENIZER
# ============================================================
# Path to locally downloaded GPT-2 model (124M parameters)
model_id = '/Users/dmitri/Documents/LLM/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e'
# Load the pre-trained GPT-2 model
# device_map="auto" = automatically place model on available device (CPU/GPU/MPS)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto"
)
# Load tokenizer - converts text to token IDs that the model understands
tokenizer = AutoTokenizer.from_pretrained(model_id)

# GPT-2 doesn't have a pad token by default
# Set pad_token to eos_token (end-of-sequence) so padding works during batched training
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 3. PREPROCESS & TOKENIZE DATASET
# ============================================================
# This function converts raw text into tokenized format the model can train on
def preprocess(examples):
    # Dataset has single column named "0" containing full conversation text
    texts = examples["0"]

    # Convert text strings into token IDs
    tokenized = tokenizer(
        texts,
        truncation=True,      # Cut off text longer than max_length
        max_length=512,        # Max tokens per sample (512 tokens ≈ 400 words)
                               # Shorter = faster training, but may cut off long conversations
                               # Tuning: 128 (fast POC) | 256 (balanced) | 512 (full conversations) | 1024 (GPT-2 max)
        padding="max_length",  # Pad shorter texts to max_length with pad tokens
                               # Ensures all samples have same length for batching
        return_tensors=None    # Return as Python lists (not PyTorch tensors)
                               # Required for dataset.map() compatibility
    )
    # For causal language modeling, labels = input_ids (model predicts next token)
    # The loss is computed by shifting: predict token[i+1] from token[i]
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Apply preprocessing to entire dataset
# batched=True = process multiple samples at once (faster than one-by-one)
# remove_columns = drop original text column, keep only tokenized columns (input_ids, attention_mask, labels)
train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(preprocess, batched=True, remove_columns=eval_dataset.column_names)

# ============================================================
# 4. CONFIGURE LoRA (Low-Rank Adaptation)
# ============================================================
# LoRA freezes the original model weights and injects small trainable matrices
# Instead of training all 124M params, we only train ~0.5M params (99.6% reduction)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Task type: Causal Language Modeling (next token prediction)

    r=16,              # Rank of the low-rank matrices (decomposition dimension)
                       # Higher r = more capacity but more parameters & slower
                       # Tuning: 4 (minimal) | 8 (standard) | 16 (good quality) | 32 (high capacity) | 64 (max)

    lora_alpha=64,     # Scaling factor for LoRA weights. Controls how much LoRA affects output.
                       # Rule of thumb: set to 2x-4x of r
                       # Higher alpha = stronger LoRA influence
                       # Tuning: alpha/r ratio matters. Common: alpha=2*r

    lora_dropout=0.05, # Dropout rate applied to LoRA layers for regularization
                       # Prevents overfitting by randomly zeroing some weights during training
                       # Tuning: 0.0 (no dropout) | 0.05 (light) | 0.1 (moderate) | 0.2 (heavy, small datasets)

    target_modules=["c_attn", "c_proj"],  # Which model layers to apply LoRA to
                       # c_attn = combined Q/K/V attention projection (GPT-2 specific name)
                       # c_proj = output projection after attention
                       # More modules = more trainable params but better learning
                       # Options for GPT-2: ["c_attn"] | ["c_attn","c_proj"] | ["c_attn","c_proj","c_fc","mlp.c_proj"]
)

# Apply LoRA config to the model - wraps original layers with LoRA adapters
model = get_peft_model(model, lora_config)
# Print how many parameters are trainable vs total
# Example output: "trainable params: 0.59M || all params: 124.44M || trainable%: 0.47%"
model.print_trainable_parameters()

# ============================================================
# 5. TRAINING ARGUMENTS
# ============================================================
training_args = TrainingArguments(
    output_dir="./lora_output",   # Directory to save checkpoints and training logs

    per_device_train_batch_size=4, # Number of samples processed per forward/backward pass
                                   # Higher = faster but uses more memory
                                   # Tuning: 1 (low memory) | 4 (balanced) | 8-16 (fast, needs more RAM/VRAM)

    num_train_epochs=3,            # Number of full passes through the entire dataset
                                   # More epochs = better learning but risk of overfitting
                                   # Tuning: 1 (quick test) | 3 (standard) | 5-10 (small datasets)

    logging_steps=50,              # Log training loss every N steps
                                   # Lower = more frequent logging, useful for monitoring

    learning_rate=2e-4,            # How fast the model updates weights (step size)
                                   # Too high = unstable training, too low = slow learning
                                   # Tuning: 1e-5 (conservative) | 5e-5 (careful) | 2e-4 (LoRA default) | 5e-4 (aggressive)

    warmup_steps=100,              # Gradually increase learning rate from 0 to target over N steps
                                   # Prevents training instability at the start
                                   # Tuning: 0 (no warmup) | 50-100 (short) | 500 (long training)

    gradient_accumulation_steps=2, # Accumulate gradients over N steps before updating weights
                                   # Effective batch size = per_device_train_batch_size × gradient_accumulation_steps
                                   # Here: 4 × 2 = 8 effective batch size
                                   # Useful when you can't fit large batches in memory
                                   # Tuning: 1 (no accumulation) | 2-4 (common) | 8-16 (simulates large batch)

    # --- Evaluation settings ---
    eval_strategy="steps",         # Eval during training
    eval_steps=100,                # Eval every 100 steps
    metric_for_best_model="loss",  # Track validation loss
    load_best_model_at_end=True,   # If True, loads the best checkpoint (by eval loss) after training
                                   # Requires eval_dataset and evaluation_strategy to be set

    # --- Checkpoint settings ---
    save_strategy="steps",         # When to save checkpoints: "steps" | "epoch" | "no"
    save_steps=500,                # Save checkpoint every N steps (only when save_strategy="steps")
                                   # Lower = more frequent saves (safe from crashes) but uses disk space
    save_total_limit=3,            # Keep only the N most recent checkpoints (deletes older ones)
                                   # Saves disk space. Set higher if you want to compare checkpoints.
)

# ============================================================
# 6. SAVE EXPERIMENT CONFIGURATION
# ============================================================
# Track all hyperparameters and settings for reproducibility and comparison
# Gather all experiment parameters
lora_adapter_config_path = "./lora_adapter"
os.makedirs(lora_adapter_config_path, exist_ok=True)
experiment_config = {
    # Experiment metadata
    "experiment_name": f"gpt2_lora_sales_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "description": "GPT-2 LoRA fine-tuning on sales conversations",

    # Model configuration
    "base_model": "gpt2-124M",
    "model_path": model_id,
    "total_params": "124M",
    "trainable_params": "~0.59M",

    # Dataset configuration
    "dataset": dataset_name,
    "dataset_size": len(train_dataset) + len(eval_dataset),
    "train_samples": len(train_dataset),
    "eval_samples": len(eval_dataset),
    "train_eval_split": "90/10",

    # LoRA hyperparameters
    "lora": {
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "target_modules": list(lora_config.target_modules),  # Convert set to list
        "task_type": str(lora_config.task_type),
    },

    # Tokenization settings
    "tokenization": {
        "max_length": 512,
        "padding": "max_length",
        "truncation": True,
    },

    # Training hyperparameters
    "training": {
        "num_train_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
        "learning_rate": training_args.learning_rate,
        "warmup_steps": training_args.warmup_steps,
        "logging_steps": training_args.logging_steps,
        "eval_steps": training_args.eval_steps,
        "save_steps": training_args.save_steps,
        "save_total_limit": training_args.save_total_limit,
        "load_best_model_at_end": training_args.load_best_model_at_end,
        "metric_for_best_model": training_args.metric_for_best_model,
    },

    # Computed values
    "total_training_steps": (len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)) * training_args.num_train_epochs,

    # Output paths
    "output_dir": training_args.output_dir,
    "adapter_save_path": lora_adapter_config_path,
}

# Save with date-based filename
experiment_folder = f"{lora_adapter_config_path}/experiments"
experiment_filename = f"{experiment_folder}/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
os.makedirs(experiment_folder, exist_ok=True)
with open(experiment_filename, "w") as f:
    json.dump(experiment_config, f, indent=2)

print(f"\n{'='*60}")
print(f"Experiment config saved to: {experiment_filename}")
print(f"{'='*60}\n")

# ============================================================
# 7. TRAIN
# ============================================================
# Trainer handles the training loop: forward pass, loss computation, backpropagation, optimization
trainer = Trainer(
    model=model,                   # The LoRA-wrapped model
    args=training_args,            # Training configuration from above
    train_dataset=train_dataset,   # The tokenized dataset
    eval_dataset=eval_dataset,     # Add eval data
)

# Start training from scratch
print(f"Starting training: {experiment_config['experiment_name']}")
print(f"Total steps: {experiment_config['total_training_steps']}")
print(f"Effective batch size: {experiment_config['training']['effective_batch_size']}\n")
trainer.train()
# To resume from checkpoint after interruption (e.g. power cut), use:
# trainer.train(resume_from_checkpoint=True)  # Auto-finds latest checkpoint in output_dir
# Or specify checkpoint:
# trainer.train(resume_from_checkpoint="./lora_output/checkpoint-500")

# ============================================================
# 8. SAVE TRAINING RESULTS & LoRA ADAPTER
# ============================================================
# Saves only the LoRA weights (small, ~few MB), not the full model
# To use later: load base model + this adapter with PeftModel.from_pretrained()
model.save_pretrained(lora_adapter_config_path)

# Add final training metrics to experiment config
training_history = trainer.state.log_history
final_train_loss = [x['loss'] for x in training_history if 'loss' in x][-1] if training_history else None
final_eval_loss = [x['eval_loss'] for x in training_history if 'eval_loss' in x][-1] if training_history else None

experiment_config["results"] = {
    "final_train_loss": final_train_loss,
    "final_eval_loss": final_eval_loss,
    "total_steps_trained": trainer.state.global_step,
    "training_completed": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}

# Update experiment file with results
with open(experiment_filename, "w") as f:
    json.dump(experiment_config, f, indent=2)

# Also save copy with adapter for easy reference
adapter_config_path = f"{lora_adapter_config_path}/experiment_config.json"
with open(adapter_config_path, "w") as f:
    json.dump(experiment_config, f, indent=2)

print(f"\n{'='*60}")
print(f"Training complete!")
print(f"Final train loss: {final_train_loss:.4f}" if final_train_loss else "N/A")
print(f"Final eval loss: {final_eval_loss:.4f}" if final_eval_loss else "N/A")
print(f"Adapter saved to: {lora_adapter_config_path}")
print(f"Experiment config: {experiment_filename}")
print(f"{'='*60}\n")
