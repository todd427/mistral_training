# train_lora.py
#
# Train a LoRA model on a dataset.
#
# USAGE:
#     python train_lora.py
#
#
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

def train():
    # === CONFIGURATION ===
    model_name = "mistralai/Mistral-7B-v0.1"  # Change as needed
    data_path = "../data/train.jsonl"

    # === LOAD DATA ===
    dataset = load_dataset("json", data_files=data_path, split="train")
    print("Sample record:", dataset[0])

    # === LOAD TOKENIZER ===
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Make sure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === TOKENIZE DATA (with remove_columns!) ===
    def tokenize_fn(batch):
        text = batch.get("prompt", "") + " " + batch.get("completion", "")
        out = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        return {
            "input_ids": out["input_ids"],
            "attention_mask": out["attention_mask"],
            "labels": out["input_ids"]
        }

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=False,
        remove_columns=dataset.column_names  # <-- THIS LINE IS KEY
    )
    print("Sample tokenized:", tokenized_dataset[0])

    # === LOAD MODEL (NO META TENSORS) ===
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=False,       # <-- This avoids meta tensor errors!
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None                # Loads on CPU; Trainer will move to GPU
    )

    # === INTEGRATE LORA (PEFT) ===
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Example; adjust for your model if needed
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # === TRAINING ARGUMENTS ===
    training_args = TrainingArguments(
        per_device_train_batch_size=2,     # Adjust for your GPU/CPU
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        output_dir="./lora_out",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
    )

    # === TRAINER ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    print("Training finished!")

if __name__ == "__main__":
    train()
