import json
import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments

from transformers import AutoTokenizer

model_name = "google/mt5-base"  # Replace with your model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load your dataset
data = [
    {"question": "Differentiate x^2", "answer": "2x"},
    {"question": "Differentiate x^3", "answer": "3x^2"},
    {"question": "Differentiate sin(x)", "answer": "cos(x)"},
    {"question": "Differentiate cos(x)", "answer": "-sin(x)"},
    {"question": "Differentiate e^x", "answer": "e^x"},
    {"question": "Differentiate ln(x)", "answer": "1/x"},
    {"question": "Differentiate 5x^4", "answer": "20x^3"},
    {"question": "Differentiate x^2 * sin(x)", "answer": "2x * sin(x) + x^2 * cos(x)"},
    {"question": "Differentiate x / (x + 1)", "answer": "1 / (x + 1)^2"},
    {"question": "Differentiate sqrt(x)", "answer": "1 / (2 * sqrt(x))"},
    {"question": "Differentiate tan(x)", "answer": "sec^2(x)"},
    {"question": "Differentiate x * e^x", "answer": "(1 + x) * e^x"},
    {"question": "Differentiate ln(x^2 + 1)", "answer": "2x / (x^2 + 1)"},
    {"question": "Differentiate sin(x) * cos(x)", "answer": "cos(2x)"},
    {"question": "Differentiate 1/x", "answer": "-1 / x^2"},
    {"question": "Differentiate arctan(x)", "answer": "1 / (1 + x^2)"},
    {"question": "Differentiate x^5 + 3x^2", "answer": "5x^4 + 6x"},
    {"question": "Differentiate e^(3x)", "answer": "3 * e^(3x)"},
    {"question": "Differentiate x^2 * ln(x)", "answer": "2x * ln(x) + x"},
    {"question": "Differentiate sin(x^2)", "answer": "2x * cos(x^2)"},
]

# Create Dataset
train_dataset = Dataset.from_dict({
    "question": [item["question"] for item in data],
    "answer": [item["answer"] for item in data],
})

# Preprocessing function
def preprocess_function(examples):
    # Tokenize questions and answers
    model_inputs = tokenizer(examples['question'], max_length=128, padding='max_length', truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['answer'], max_length=128, padding='max_length', truncation=True)

    model_inputs['labels'] = labels['input_ids']
    
    # Ensure that labels are correctly handled (replace padding tokens)
    model_inputs['labels'] = np.where(np.array(model_inputs['labels']) == tokenizer.pad_token_id, -100, model_inputs['labels'])

    return model_inputs

# Tokenize the dataset
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./trained_model")
tokenizer.save_pretrained("./trained_model")
