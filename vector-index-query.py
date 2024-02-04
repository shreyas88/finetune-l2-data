from llama_index import Document,VectorStoreIndex,ServiceContext
import pandas as pd
import argparse
from embeddings_finetune import WOBEmbeddings
import llama_index

llama_index.set_global_handler("simple")

# Initialize parser
parser = argparse.ArgumentParser(description="Generate text from a language model")
# Adding argument
parser.add_argument("--query", type=str, required=True, help="Input text to generate text from")
# Parse arguments
args = parser.parse_args()


df = pd.read_csv('data/train_chat_sample.csv')
df['finalText'] = df['instruction']+" " + df['text']
docs = [Document(text=t[1]) for t in df['finalText'].items()]

# embedding generation
service_context = ServiceContext.from_defaults(
    embed_model=WOBEmbeddings(embed_batch_size=10), llm=None
)
index = VectorStoreIndex.from_documents(docs, service_context=service_context, show_progress=True)

query_engine = index.as_query_engine()
response = query_engine.query(args.query)
print(response)
