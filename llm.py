import random
import torch
from typing import Tuple, Any, Dict
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline, set_seed


class LLM:
    __instance = None

    def __new__(cls, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]):
        if LLM.__instance is None :
            LLM.__instance = super(LLM, cls).__new__(cls, *args, **kwargs)
            cls._model_name_or_path = "TehVenom/Pygmalion-7b-Merged-Safetensors"

            seed = random.randint(0, 2**32-1)
            set_seed(seed)

            cls._tokenizer = LlamaTokenizer.from_pretrained(
                cls._model_name_or_path,
                clean_up_tokenization_spaces=True
            )
            cls._model = LlamaForCausalLM.from_pretrained(
                cls._model_name_or_path,device_map='auto',
                torch_dtype=torch.float16,
                use_safetensors=True,
                trust_remote_code=False
            )
        return LLM.__instance

    def model_name(self) -> str:
        return self._model_name_or_path.replace("/", "_")

    def token_count(self, text: str) -> int:
        tokens = self._tokenizer.encode(text)
        return len(tokens)

    def create_completion(self, prompt: str, max_new_tokens: int) -> str:
        length = len(prompt)
        pipe = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        result = pipe(prompt)[0]['generated_text']
        return result[length:]

