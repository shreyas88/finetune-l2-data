from llama_index import Document,VectorStoreIndex,ServiceContext
import pandas as pd
import argparse
from embeddings-finetune import WOBEmbeddings

# Initialize parser
parser = argparse.ArgumentParser(description="Generate text from a language model")
# Adding argument
parser.add_argument("--query", type=str, required=True, help="Input text to generate text from")
# Parse arguments
args = parser.parse_args()

df = pd.read_csv('data/train_chat.csv')
df['finalText'] = df['instruction']+" " + df['text']
docs = [Document(text=t[1]) for t in df['finalText'].items()]

# embedding generation
service_context = ServiceContext.from_defaults(
    embed_model=WOBEmbeddings()
)
index = VectorStoreIndex.from_documents(docs, service_context=service_context, show_progress=True)

query_engine = index.as_query_engine()
response = query_engine.query(args.query)
print(response)
