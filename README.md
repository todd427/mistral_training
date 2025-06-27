# Mistral LoRA Project

This project demonstrates LoRA fine-tuning using the Mistral-7B model.

## Structure

- `data/`: Contains training and evaluation `.jsonl` files.
- `scripts/train_lora.py`: Training script.
- `scripts/eval_lora.py`: Basic evaluation script.
- `configs/lora_config.json`: LoRA configuration.

## Usage

1. Place training data into `data/train.jsonl`.
2. Run training:
   ```bash
   python scripts/train_lora.py
# mistral_training
