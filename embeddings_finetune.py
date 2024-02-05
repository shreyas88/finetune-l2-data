from typing import Any, List

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.embeddings.base import BaseEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import torch


class WOBEmbeddings(BaseEmbedding):
    _model_path: str  = PrivateAttr()
    _tokenizer: PreTrainedTokenizer = PrivateAttr()
    _model: PreTrainedModel = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
      self._model_path = "myllm-finetune_fp16_batch16"
      self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
      self._model = AutoModelForCausalLM.from_pretrained(self._model_path)
      super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
      return "wobembeddings"

    async def _aget_query_embedding(self, query: str) -> List[float]:
      return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
      return self._get_text_embedding(text)

    def embed_text(self, text: str) -> List[float]:
      input_ids = self._tokenizer.encode(text, return_tensors="pt")
      last_hidden_state = self._model(**{"input_ids":input_ids, "output_hidden_states":True}).hidden_states[-1]
      return torch.sum(last_hidden_state, dim=1).squeeze(0).tolist()
    
    def batch_embed_text(self, texts: List[str]) -> List[List[float]]:
      input_ids = torch.stack([ self._tokenizer.encode(text, return_tensors="pt") for text in texts ])
      last_hidden_state = self._model(**{"input_ids":input_ids, "output_hidden_states":True}).hidden_states[-1]
      list_tensors = torch.sum(last_hidden_state, dim=1).tolist()
      return [ tensor.squeeze(0).tolist() for tensor in list_tensors ]

    def _get_query_embedding(self, query: str) -> List[float]:
      return self.embed_text(query)

    def _get_text_embedding(self, text: str) -> List[float]:
      return self.embed_text(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
      return self.batch_embed_text(texts)
