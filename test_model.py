from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set pad token to the eos token
tokenizer.pad_token = tokenizer.eos_token

# Move the model to evaluation mode
model.eval()

# Function to generate answer for a given question
def generate_answer(question):
    # Tokenize the input question and create attention mask
    input_ids = tokenizer.encode(question, return_tensors='pt')
    
    # Generate output from the model
    output = model.generate(input_ids, 
                             max_length=1150, 
                             num_return_sequences=1, 
                             no_repeat_ngram_size=2,
                             pad_token_id=tokenizer.pad_token_id)

    # Decode the generated output
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return answer

# Main loop for user input
while True:
    # Get user input
    user_question = input("Enter a math question (or type 'exit' to quit): ")

    # Break the loop if the user types 'exit'
    if user_question.lower() == 'exit':
        print("Exiting the program.")
        break

    # Generate and print the answer
    answer = generate_answer(user_question)
    print(f"Answer: {answer}\n")
