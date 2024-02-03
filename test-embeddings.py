import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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

model.eval()
with torch.no_grad():
    last_hidden_state = model(**input_ids, output_hidden_states=True).hidden_states[-1]
sum_embeddings = torch.sum(last_hidden_state, dim=1)

print(sum_embeddings)


