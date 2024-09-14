from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForPreTraining,BitsAndBytesConfig
import torch
from typing import Optional, Callable
from torch import nn
from transformers import AutoModelForCausalLM
import types
import os
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from .kv_cache_model import KVCacheModel
from .speculative_decode import *
from .generate import Generate

class GPTFast:

    def __model_compile(model:nn.Module, spec_decode:bool = False) -> nn.Module:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        model.prefill = torch.compile(model.prefill, dynamic=True, fullgraph=True)
        model = model.to(device)
        if spec_decode:
            assert hasattr(model, "draft_model"), "You have passed spec_decode = True in your torch_compile but your model doesn't have a draft model."
            draft_model = model.draft_model
            draft_model = torch.compile(draft_model, mode="reduce-overhead", fullgraph=True)
            draft_model.prefill = torch.compile(draft_model.prefill, dynamic=True, fullgraph=True)
            draft_model = draft_model.to(device)

        return model

    def __gpt_fast(model_name:str,trust_remote_code=False, **spec_dec_kwargs):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config = quantization_config,trust_remote_code=trust_remote_code)
        except:
            model = AutoModelForPreTraining.from_pretrained(model_name, quantization_config = quantization_config,trust_remote_code=trust_remote_code)
        
        model = KVCacheModel(model)
        spec_decode = False
        if spec_dec_kwargs:
            draft_model_name = spec_dec_kwargs.pop("draft_model_name")
            try:
                draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name,trust_remote_code=trust_remote_code)
            except:
                draft_model = AutoModelForPreTraining.from_pretrained(draft_model_name,trust_remote_code=trust_remote_code)
            draft_model = KVCacheModel(draft_model)
            model = add_speculative_decoding(model, draft_model, **spec_dec_kwargs)
            spec_decode = True
        model = GPTFast.__model_compile(model, spec_decode=spec_decode)
        return model
    
    def __argmax(self, probabilities):
        # Use argmax to get the token with the maximum probability
        max_prob_index = torch.argmax(probabilities, dim=-1)
        return max_prob_index.unsqueeze(0)

    def __generate_output_fast(model_name, draft_model_name,sequence, max_length,trust_remote_code=False):
        print("Experimental!!! Generating output using gpt fast with speculative decoding upto 6x faster than normal gpt generation. if fails go back to normal gpt generation.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = GPTFast.__gpt_fast(model_name, trust_remote_code=trust_remote_code, draft_model_name=draft_model_name, sample_function=GPTFast.__argmax)
        model.to(device)
        tokenizer = Generate.load_tokenizer(model_name)
        input_tokens = tokenizer.encode(sequence, return_tensors="pt").to(device)
        output = model.generate(cur_tokens=input_tokens, max_tokens=max_length, speculate_k=6)
        return output,tokenizer
    
    def __generate_output_from_model(model, tokenizer, sequence, max_length):
        print("Experimental!!! Generating output using gpt fast with speculative decoding upto 6x faster than normal gpt generation. if fails go back to normal gpt generation.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_tokens = tokenizer.encode(sequence, return_tensors="pt").to(device)
        output = model.generate(cur_tokens=input_tokens, max_tokens=max_length, speculate_k=6)
        return output,tokenizer
    
    def gpt_fast(model_name:str ,trust_remote_code=False, **spec_dec_kwargs):
        return GPTFast.__gpt_fast(model_name, trust_remote_code=trust_remote_code, **spec_dec_kwargs)
    
    def generate_output_fast(model_name, draft_model_name, sequence, max_length, trust_remote_code=False,gguf_file = None):
        return GPTFast.__generate_output_fast(model_name, draft_model_name, sequence, max_length, trust_remote_code=trust_remote_code)
    
    def generate_text_fast(model_name, draft_model_name, sequence, max_length, trust_remote_code=False):
        output,tokenizer = GPTFast.__generate_output_fast(model_name, draft_model_name, sequence, max_length, trust_remote_code=trust_remote_code)
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        return tokenizer.decode(output[0], skip_special_tokens=True)
    
    def load_model(model_path,draft_model_name,trust_remote_code=False,gguf_file=None):
        return GPTFast.__gpt_fast(model_path, trust_remote_code=trust_remote_code, draft_model_name=draft_model_name, sample_function=GPTFast.__argmax)
    
    def generate_output_from_model(model, tokenizer, sequence, max_length):
        return GPTFast.__generate_output_from_model(model, tokenizer, sequence, max_length)
    