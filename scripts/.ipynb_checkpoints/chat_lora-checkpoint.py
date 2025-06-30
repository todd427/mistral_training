# chat_lora.py
#
# Chat with a LoRA model.
#
# USAGE:
#     python chat_lora.py
#
#
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_DIR = "./lora_out/checkpoint-96"


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
model = PeftModel.from_pretrained(model, LORA_DIR)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

print("Welcome to ToddBot! (Type Ctrl+C or Ctrl+D to exit)\n")

while True:
    try:
        user_prompt = input("You: ")
        prompt = user_prompt
        # If you trained with a special prompt template, you can wrap here!
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"ToddBot: {response}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        break
