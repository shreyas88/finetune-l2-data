from typing import Any, List

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.embeddings.base import BaseEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class WOBEmbeddings(BaseEmbedding):
    def __init__(self, **kwargs: Any) -> None:
	self.model_path = "myllm-finetune_fp16_batch16"
	self.tokenizer = AutoTokenizer.from_pretrained(model_path)
	self.model = AutoModelForCausalLM.from_pretrained(model_path)
	super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
	return "wobembeddings"

    async def _aget_query_embedding(self, query: str) -> List[float]:
	return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
	return self._get_text_embedding(text)

    def embed_text(self, text: str) -> List[float]:
	input_ids = tokenizer.encode(text, return_tensors="pt")
	last_hidden_state = self.model(**{"input_ids":input_ids, "output_hidden_states":True}).hidden_states[-1]
	return torch.sum(last_hidden_state, dim=1)

    def _get_query_embedding(self, query: str) -> List[float]:
	return self.embed_text(query)

    def _get_text_embedding(self, text: str) -> List[float]:
	return self.embed_text(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
	embeddings = [ self.embed_text(text) for text in texts]
	return embeddings

