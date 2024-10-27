from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load the dataset from the CSV file
dataset = load_dataset('csv', data_files='math_dataset.csv')

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add a padding token (since GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to the end-of-sequence token

model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define a function to tokenize the dataset
def preprocess(example):
    # Tokenize the question and answer
    inputs = tokenizer(example['question'], truncation=True, padding='max_length', max_length=128)
    outputs = tokenizer(example['answer'], truncation=True, padding='max_length', max_length=128)
    
    # Return input_ids for the input and labels for the answer
    return {'input_ids': inputs['input_ids'], 'labels': outputs['input_ids']}

# Apply the preprocessing function to the dataset
tokenized_dataset = dataset.map(preprocess)

# Define data collator to handle padding for batches
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')

print("Fine-tuning complete! Model saved in './fine_tuned_model'.")
