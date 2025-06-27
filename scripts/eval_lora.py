# eval_lora.py
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def evaluate():
    base_model = "mistralai/Mistral-7B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, "./results")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = "Who is Todd McCaffrey?"
    output = generator(prompt, max_length=100, do_sample=True)
    print(output[0]["generated_text"])

if __name__ == "__main__":
    evaluate()
