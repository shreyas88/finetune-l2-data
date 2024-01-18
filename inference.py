from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "myllm-finetune_fp16_batch24"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
input_text = "Health benefits of regular exercise"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids)
predicted_text = tokenizer.decode(output[0], skip_special_tokens=False)
print(predicted_text)
