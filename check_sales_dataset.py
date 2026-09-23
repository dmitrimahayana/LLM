from datasets import load_dataset

# Load dataset
dataset = load_dataset("goendalf666/sales-conversations-instruction-ext", split="train")

print(f"Total samples: {len(dataset)}")
print(f"\nColumns: {dataset.column_names}")
print(f"\nFirst example:")
print(dataset[0])
print("\n" + "="*80)
print("\nSecond example:")
print(dataset[1])
