import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize parser
parser = argparse.ArgumentParser(description="Generate text from a language model")
# Adding argument
parser.add_argument("--input_text", type=str, required=True, help="Input text to generate text from")
# Parse arguments
args = parser.parse_args()

model_path = "myllm-finetune_fp16_batch16"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Use the input text from the command line argument
input_text = args.input_text
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids)
predicted_text = tokenizer.decode(output[0], skip_special_tokens=False)
print(predicted_text)

