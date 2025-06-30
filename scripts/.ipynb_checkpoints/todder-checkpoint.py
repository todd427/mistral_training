# todder.py
# Chat with LoRA-powered Toddie, with persona cues and logging.
import time
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---- MODEL SETUP ----
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_DIR = "./lora_out/checkpoint-96"   # Update if needed

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
model = PeftModel.from_pretrained(model, LORA_DIR)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# ---- HUMANLIKE CUES ----
HUMANLIKE_RESPONSES = [
    "(slow sigh...)",
    "Please pardon me, I'm just waking up.",
    "(thinking...)",
    "(pauses for dramatic effect)",
    "(clears throat)",
]

def log_conversation(user, bot, meta="", filename="todder_chat_log.jsonl"):
    log_entry = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "user": user,
        "todder": bot,
        "meta": meta
    }
    with open(filename, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def toddbot_reply(user_input):
    meta = ""
    reply_prefix = ""
    if random.random() < 0.25:
        meta = random.choice(HUMANLIKE_RESPONSES)
        reply_prefix = f"{meta}\n"
    # --- LLM Generation ---
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Remove the input prompt if echoed in the response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Only return the part after the user input if it's repeated
    if full_response.lower().startswith(user_input.lower()):
        llm_reply = full_response[len(user_input):].strip()
    else:
        llm_reply = full_response.strip()
    reply = reply_prefix + llm_reply
    return reply, meta

print("Good morning, toddie! (Type 'exit' or Ctrl+C to quit)\n")

while True:
    user_prompt = input("You: ")
    if user_prompt.lower() in ["exit", "quit"]:
        break
    prompt = (
        "You are Toddie, a two-year-old, cheerful kid.\n"
        f"You: {user_prompt}\n"
        "Toddie:"
    )
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
    if "Toddie:" in response:
        llm_reply = response.split("Toddie:")[-1].strip()
    else:
        llm_reply = response.strip()
    print(f"Toddie: {llm_reply}")