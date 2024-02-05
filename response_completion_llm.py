from typing import Optional, List, Mapping, Any

from llama_index.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.base import llm_completion_callback
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from llama_index.bridge.pydantic import PrivateAttr

class ResponseCompletionLLM(CustomLLM):
    _model_path: str  = PrivateAttr()
    _tokenizer: PreTrainedTokenizer = PrivateAttr()
    _model: PreTrainedModel = PrivateAttr()

    context_window: int = 4096
    num_output: int = 512
    model_name: str = "mistral-instruct-7b"

    def __init__(self, *args, **kwargs):
        self._model_path = "myllm-finetune_fp16_batch16"
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        super().__init__(*args, **kwargs)

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        output = self._model.generate(input_ids, max_new_tokens=500)
        predicted_text = self._tokenizer.decode(output[0], skip_special_tokens=False)
        return CompletionResponse(text=predicted_text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        output = self._model.generate(input_ids, max_new_tokens=500)
        predicted_text = self._tokenizer.decode(output[0], skip_special_tokens=False)
        yield CompletionResponse(text=predicted_text)

