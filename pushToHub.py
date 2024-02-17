from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "/workspace/jyh/Zero-Haruhi/train_2024-02-16-17-51"

model = AutoModelForCausalLM.from_pretrained(
    checkpoint, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

model.push_to_hub("Haruhi-Zero-GLM3-6B-0_4")
tokenizer.push_to_hub("Haruhi-Zero-GLM3-6B-0_4")
