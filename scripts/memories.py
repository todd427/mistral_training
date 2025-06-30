# memories.py
#
# Generate memory cards for ToddBot.
#
# USAGE:
#     python memories.py
#
#
import json

def main():
    print("== ToddBot Memory Card Generator ==")
    print("Type a personal fact, memory, or 'exit' to finish.\n")
    print("Example: My birthday is April 27, 1956.")
    print("Example: The protagonist of 'Cody's War' is Cody.")
    print("Example: I wrote my first novel at age 11.\n")

    memories = []
    while True:
        mem = input("Enter a memory: ")
        if mem.lower() in ("exit", "quit"):
            break
        # Choose prompt style: statement or Q&A
        style = input("Prompt style? [1] Statement (default), [2] Q&A: ").strip()
        if style == "2":
            question = input("Enter the question for this memory: ")
            memories.append({
                "prompt": question.strip(),
                "completion": mem.strip()
            })
        else:
            # Default: “Remember this:” prompt
            memories.append({
                "prompt": "Remember this:",
                "completion": mem.strip()
            })
        print("Added!\n")
    
    # Save to file
    with open("memories.jsonl", "a") as f:
        for m in memories:
            f.write(json.dumps(m) + "\n")

    print(f"Saved {len(memories)} memories to memories.jsonl")

if __name__ == "__main__":
    main()
