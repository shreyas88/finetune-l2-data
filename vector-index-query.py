from llama_index import Document,VectorStoreIndex,ServiceContext
import pandas as pd
import argparse
from embeddings_finetune import WOBEmbeddings
import llama_index
from llama_index import Document
from llama_index.ingestion import IngestionPipeline, IngestionCache
from llama_index.ingestion.cache import RedisCache
from llama_index.vector_stores import WeaviateVectorStore
import weaviate


llama_index.set_global_handler("simple")

# Initialize parser
parser = argparse.ArgumentParser(description="Generate text from a language model")
# Adding argument
parser.add_argument("--query", type=str, required=True, help="Input text to generate text from")
parser.add_argument("--batch_size", type=int, required=False, help="Batch size for embedding generation")
parser.add_argument('--load', action='store_true')
parser.set_defaults(load=False)

# Parse arguments
args = parser.parse_args()

if args.batch_size is None:
    args.batch_size = 10

df = pd.read_csv('data/train_chat_sample.csv')
df['finalText'] = df['instruction']+" " + df['text']
docs = [Document(text=t[1]) for t in df['finalText'].items()]


client = weaviate.Client(
  url="https://test-wob-project-8zsw4jwy.weaviate.network",
)

vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name="CachingTest"
)

'''
ingest_cache = IngestionCache(
    cache=RedisCache.from_host_and_port(host="127.0.0.1", port=6379),
    collection="my_test_cache",
)
'''

pipeline = IngestionPipeline(
    transformations=[
        WOBEmbeddings(embed_batch_size=args.batch_size),
    ],
    vector_store=vector_store,
)

if args.load:
    pipeline.load("./pipeline_storage")

# Ingest directly into a vector db
nodes = pipeline.run(documents=docs, show_progress=True)
pipeline.persist("./pipeline_storage")

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    service_context=ServiceContext.from_defaults(
    embed_model=WOBEmbeddings(embed_batch_size=args.batch_size), llm=None),
)

query_engine = index.as_query_engine()
response = query_engine.query(args.query)
print(response)
