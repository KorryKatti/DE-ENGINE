import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Verify that PyTorch can detect the GPU (if available)
print(f"CUDA Available: {torch.cuda.is_available()}")

# Load a pre-trained model (GPT-2) to make sure transformers is working
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

print("Setup complete!")
