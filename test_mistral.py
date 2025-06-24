import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the model name
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Load the tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for efficiency
    device_map="auto"  # Automatically place model on GPU if available
)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prepare a simple prompt
prompt = "Hello! Can you tell me something interesting about movies?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate a response
print("Generating response...")
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    top_k=50,
    do_sample=True
)

# Decode and print the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Model response: {response}")

print("Test completed successfully!")