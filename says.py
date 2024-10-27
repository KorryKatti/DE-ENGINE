from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

# Load the trained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./results")
tokenizer = AutoTokenizer.from_pretrained("./results")

# Ensure model is in evaluation mode
model.eval()

while True:
    question = input("Enter your question (type 'quit' to exit): ")
    if question.lower() == 'quit':
        break

    # Add prefix "question:" to align with training format
    input_text = f"question: {question}"
    
    # Encode the input question
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=50)
    
    # Generate an answer
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=50,               # Maximum length of the generated answer
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,       # Number of answers to generate
            no_repeat_ngram_size=2,       # Prevents repetition
            early_stopping=True,          # Stops early when model is confident
            num_beams=5                   # Use beam search for better quality
        )

    # Decode and print the output
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Answer:", answer)
