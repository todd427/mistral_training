# toddie_chat.py
#
# Chat with a LoRA model, humanlike cues, and conversation logging.
#
# USAGE:
#     python toddie_chat.py
#

import time
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- Settings ---
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_DIR = "./lora_out"         # adjust if needed
LOG_FILE = "toddBot_chat_log.jsonl"

HUMANLIKE_RESPONSES = [
    "(slow sigh...)",
    "Please pardon me, I'm just waking up.",
    "(thinking...)",
    "(pauses for dramatic effect)",
    "(clears throat)",
]

# --- Functions ---

def log_conversation(user, bot, meta=""):
    log_entry = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "user": user,
        "toddBot": bot,
        "meta": meta
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"[Error logging conversation]: {e}")

def toddbot_reply_llm(user_input, tokenizer, model):
    # Tokenize and generate reply from LLM
    prompt = user_input
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
    # Decode full response and cut to reply only (removes prompt if needed)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Post-process: remove prompt from start, if present
    if response.startswith(user_input):
        response = response[len(user_input):].lstrip()
    return response.strip()

# --- Load LLM ---
print("[⏳] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
model = PeftModel.from_pretrained(model, LORA_DIR)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
print("[✅] Model loaded!")

print("toddy is awake! Type 'exit' or Ctrl+C/Ctrl+D to quit.\n")

while True:
    try:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        start = time.time()
        llm_reply = toddbot_reply_llm(user_input, tokenizer, model)
        # Insert humanlike meta-cue 25% of the time
        meta = ""
        if random.random() < 0.25:
            meta = random.choice(HUMANLIKE_RESPONSES)
            reply = f"{meta}\n{llm_reply}"
        else:
            reply = llm_reply
        elapsed = time.time() - start

        print(f"toddy: {reply}\n")
        log_conversation(user_input, reply, meta)

    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        break