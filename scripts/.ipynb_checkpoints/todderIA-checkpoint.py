# todderIA.py
#
# ToddBot response generator.
#
# USAGE:
#     python todderIA.py
#
#
import time
import json
import random

HUMANLIKE_RESPONSES = [
    "(slow sigh...)",
    "Please pardon me, I'm just waking up.",
    "(thinking...)",
    "(pauses for dramatic effect)",
    "(clears throat)",
    # Add your favorites!
]

def log_conversation(user, bot, meta="", filename="toddBot_chat_log.jsonl"):
    log_entry = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "user": user,
        "toddBot": bot,
        "meta": meta
    }
    with open(filename, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def toddbot_reply(user_input):
    # Here you’d call your LLM or a mock reply for demo purposes:
    reply = "Here's my answer to: " + user_input
    # Randomly choose a humanlike response, or set rules (e.g. first reply, slow reply, etc.)
    meta = ""
    if random.random() < 0.25:  # 25% chance to add a humanlike cue
        meta = random.choice(HUMANLIKE_RESPONSES)
        reply = f"{meta}\n{reply}"
    return reply, meta

print("toddBot is awake! Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    start = time.time()
    reply, meta = toddbot_reply(user_input)
    elapsed = time.time() - start

    print(f"toddBot: {reply}")
    log_conversation(user_input, reply, meta)
