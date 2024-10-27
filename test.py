# math_solver_simple.py
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import os
from datetime import datetime

# Global variables for session
history = []
session_file = "math_solver_session.json"

def load_and_setup_model(model_name="deepseek-ai/deepseek-math-7b-base"):
    """Load the model and tokenizer"""
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Add special tokens for math
    special_tokens = {
        "additional_special_tokens": [
            "[STEP]",
            "[EQUATION]",
            "[SOLUTION]",
            "[EXPLANATION]"
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Set the padding token
    tokenizer.pad_token = tokenizer.eos_token  # This line sets the eos_token as the padding token

    return model, tokenizer


def prepare_training_data(data):
    """Prepare the training dataset"""
    formatted_data = []
    for item in data:
        text = (
            f"Problem: {item['question']}\n"
            f"[STEP] Let's solve this step by step:\n"
            f"[EQUATION] {item['equation']}\n"
            f"[SOLUTION] {item['answer']}\n"
            f"[EXPLANATION] {item['explanation']}\n"
        )
        formatted_data.append({"text": text})
    
    return Dataset.from_pandas(pd.DataFrame(formatted_data))

def train_model(model, tokenizer, training_data, output_dir="math_solver_model"):
    """Train the model on the dataset"""
    dataset = prepare_training_data(training_data)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        tokens["labels"] = tokens["input_ids"].copy()  # Set labels as a copy of input_ids
        return tokens
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=1e-5,
        fp16=torch.cuda.is_available()
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print("Training completed!")


def solve_problem(model, tokenizer, problem):
    """Generate solution for a math problem"""
    input_text = (
        f"Problem: {problem}\n"
        f"[STEP] Let's solve this step by step:\n"
    )
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
        model = model.to("cuda")
    
    outputs = model.generate(
        inputs["input_ids"],
        max_length=512,
        temperature=0.3,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

def load_history():
    """Load previous session history"""
    global history
    try:
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                history = json.load(f)
    except Exception as e:
        print(f"Couldn't load previous session: {e}")
        history = []

def save_history():
    """Save current session history"""
    try:
        with open(session_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Couldn't save session: {e}")

def main():
    # Load training data
    print("Loading training data...")
    with open('derivative_training.json', 'r') as f:
        training_data = json.load(f)['training_data']
    
    # Setup model and train
    model, tokenizer = load_and_setup_model()
    train_model(model, tokenizer, training_data)
    
    # Load previous history
    load_history()
    
    print("\n=== Math Derivative Solver ===")
    print("Commands: 'quit', 'history', 'clear', 'help'")
    
    while True:
        try:
            print("\n" + "="*50)
            command = input("\nEnter problem or command: ").strip()
            
            if not command:
                continue
                
            if command.lower() == 'quit':
                print("\nSaving session...")
                save_history()
                break
                
            elif command.lower() == 'help':
                print("\nCommands:")
                print("  quit    - Exit program")
                print("  history - Show previous solutions")
                print("  clear   - Clear screen")
                print("  help    - Show this message")
                continue
                
            elif command.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
                
            elif command.lower() == 'history':
                if not history:
                    print("\nNo previous solutions found.")
                    continue
                print("\nPrevious solutions:")
                for i, item in enumerate(history[-5:], 1):
                    print(f"\n{i}. Problem: {item['problem']}")
                    print(f"   Time: {item['timestamp']}")
                    print(f"   Solution: {item['solution'][:100]}...")
                continue
            
            # Solve problem
            print("\nSolving...")
            solution = solve_problem(model, tokenizer, command)
            print("\nSolution:")
            print(solution)
            
            # Save to history
            history.append({
                'problem': command,
                'solution': solution,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Enter 'quit' to exit properly.")
            continue
            
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            continue

if __name__ == "__main__":
    main()