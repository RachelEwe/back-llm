import os
import random
import torch
from typing import Tuple, Any, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
from auto_gptq import exllama_set_max_input_length


class LLM:
    __instance = None

    def __new__(cls, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]):
        if LLM.__instance is None :
            LLM.__instance = super(LLM, cls).__new__(cls, *args, **kwargs)
            cls._model_name_or_path = "/home/dev/models/" + os.getenv("MODEL_NAME")
            seed = random.randint(0, 2**32-1)
            set_seed(seed)

            cls._tokenizer = AutoTokenizer.from_pretrained(
                cls._model_name_or_path,
                clean_up_tokenization_spaces=True,
                legacy=True
            )
            cls._model = AutoModelForCausalLM.from_pretrained(
                cls._model_name_or_path,device_map='auto',
                torch_dtype=torch.float16,
                trust_remote_code=False
            )
            if cls._model.config.quantization_config.use_exllama is True:
                cls._model = exllama_set_max_input_length(cls._model, max_input_length=8192)
            print(cls._model.config)
        return LLM.__instance

    def model_name(self) -> str:
        return self._model_name_or_path.replace("/", "_")

    def token_count(self, text: str) -> int:
        tokens = self._tokenizer.encode(text)
        return len(tokens)

    def create_completion(self, **kwargs) -> str:
        prompt = kwargs["prompt"]
        kwargs.pop("prompt")
        length = len(prompt)
        pipe = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            do_sample=True,
            **kwargs
        )
        result = pipe(prompt)[0]['generated_text']
        return result[length:]


