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
from llama_index.storage.docstore import SimpleDocumentStore



llama_index.set_global_handler("simple")

# Initialize parser
parser = argparse.ArgumentParser(description="Generate text from a language model")
# Adding argument
parser.add_argument("--query", type=str, required=True, help="Input text to generate text from")

# Parse arguments
args = parser.parse_args()

client = weaviate.Client(
  url="https://test-wob-project-8zsw4jwy.weaviate.network",
)

vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name="CachingTest"
)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    service_context=ServiceContext.from_defaults(
    embed_model=WOBEmbeddings(), llm=None),
)

query_engine = index.as_query_engine()
response = query_engine.query(args.query)
print(response)
