from transformers import AutoModelForPreTraining, AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,TextIteratorStreamer,GenerationConfig,AutoProcessor
import torch
from threading import Thread


class Generate:
    default_chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<start_of_turn>system\n' + message['content'] | trim + '<end_of_turn>\n' }}{% elif message['role'] == 'user' %}{{ '<start_of_turn>user\n' + message['content'] | trim + '<end_of_turn>\n' }}{% elif message['role'] == 'assistant' %}{{ '<start_of_turn>assistant\n' + message['content'] | trim + '<end_of_turn>\n' }}{% else %}{{ raise_exception('Unsupported role: ' + message['role']) }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>assistant\n' }}{% endif %}"
    
    def __load_model(model_path, trust_remote_code=False,gguf_file=None,auto_quantize=None):
        model = None
        quantization_config = None
        try:
            if auto_quantize.lower() == "4bit" or auto_quantize.lower() == "high":
                print("Using 4bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            elif  auto_quantize.lower() == "8bit" or auto_quantize.lower() == "low":
                print("Using 8bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit =True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            else:
                print("Error: Either give 4bit/high or 8bit/low as the value of auto_quantize. moving ahead without any quantization" )
                quantization_config = None
        except:
            quantization_config = None
        if quantization_config is not None:
            print("Quantization config: ",quantization_config)
            try:
                model = AutoModelForPreTraining.from_pretrained(model_path,trust_remote_code=trust_remote_code,gguf_file=gguf_file,quantization_config = quantization_config)
            except:
                model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=trust_remote_code,gguf_file=gguf_file,quantization_config = quantization_config)
        else:
            try:
                model = AutoModelForPreTraining.from_pretrained(model_path,trust_remote_code=trust_remote_code,gguf_file=gguf_file)
            except:
                model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=trust_remote_code,gguf_file=gguf_file)
        

        return model


    def __load_tokenizer(tokenizer_path,gguf_file=None):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,gguf_file=gguf_file)
        return tokenizer
    
    def __load_processor(tokenizer_path,use_fast=False,gguf_file=None):
        processor = AutoProcessor.from_pretrained(tokenizer_path, use_fast=use_fast, gguf_file=gguf_file)
        return processor
    def __generate_output(model_path, sequence, max_length,trust_remote_code=False,gguf_file=None,auto_quantize=None):
        model = Generate.__load_model(model_path,trust_remote_code=trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize)
        tokenizer = Generate.__load_tokenizer(model_path,gguf_file=gguf_file)
        ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
        final_outputs = model.generate(
            ids,
            do_sample=True,
            max_length=max_length,
            pad_token_id=model.config.eos_token_id,
            top_k=50,
            top_p=0.95,
        )
        return final_outputs,tokenizer
    
    def __generate_output_stream(model_path, sequence, max_length,temperature = 0.1,trust_remote_code=False,gguf_file=None,auto_quantize = None):
        try:
            model = Generate.__load_model(model_path,trust_remote_code=trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize)
            tokenizer = Generate.__load_tokenizer(model_path,gguf_file=gguf_file)
            if tokenizer.chat_template is None:
                print("Warning: Chat template not found in tokenizer. Applying default chat template")
                tokenizer.chat_template = Generate.default_chat_template
            inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
            inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_config = GenerationConfig(
                temperature=temperature,
                do_sample=True,
            )
            generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_length, generation_config=generation_config)
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            return thread, streamer
        except Exception as e:
            raise RuntimeError(f"Failed to setup streaming generation: {str(e)}")
    
    def __generate_chat_output(model_path, sequence, max_length,temperature = 0.1,trust_remote_code=False,gguf_file=None,auto_quantize = None):
        try:
            model = Generate.__load_model(model_path,trust_remote_code=trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize)
            tokenizer = Generate.__load_tokenizer(model_path,gguf_file=gguf_file)
            if tokenizer.chat_template is None:
                print("Warning: Chat template not found in tokenizer. Applying default chat template")
                tokenizer.chat_template = Generate.default_chat_template
            formatted_chat = tokenizer.apply_chat_template(sequence, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
            inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
            final_outputs = model.generate(
                **inputs, 
                max_new_tokens=max_length, 
                temperature=temperature)
            return inputs,final_outputs,tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to generate chat output: {str(e)}")
    
    def __generate_chat_output_stream(model_path, sequence, max_length,temperature = 0.1,trust_remote_code=False,gguf_file=None,auto_quantize = None):
        try:
            model = Generate.__load_model(model_path,trust_remote_code=trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize)
            tokenizer = Generate.__load_tokenizer(model_path,gguf_file=gguf_file)
            if tokenizer.chat_template is None:
                print("Warning: Chat template not found in tokenizer. Applying default chat template")
                tokenizer.chat_template = Generate.default_chat_template
            formatted_chat = tokenizer.apply_chat_template(sequence, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
            inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_config = GenerationConfig(
                temperature=temperature,
                do_sample=True,
            )
            generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_length, generation_config=generation_config)
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            return thread, streamer
        except Exception as e:
            raise RuntimeError(f"Failed to setup streaming generation: {str(e)}")
    
    def __generate_text(model_path, sequence, max_length,trust_remote_code=False,gguf_file=None,auto_quantize=None):
        final_outputs,tokenizer = Generate.__generate_output(model_path, sequence, max_length,trust_remote_code=trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize)
        
        print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))
        return (tokenizer.decode(final_outputs[0], skip_special_tokens=True))

    def __generate_output_from_model(model, tokenizer, sequence, max_length,trust_remote_code=False):
        ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
        final_outputs = model.generate(
            ids,
            do_sample=True,
            max_length=max_length,
            pad_token_id=model.config.eos_token_id,
            top_k=50,
            top_p=0.95,
        )
        return final_outputs,tokenizer
    
    def __generate_output_from_model_stream(model, tokenizer, sequence, max_length,temperature=0.1,trust_remote_code=False):
        try:
            if tokenizer.chat_template is None:
                print("Warning: Chat template not found in tokenizer. Applying default chat template")
                tokenizer.chat_template = Generate.default_chat_template
            inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
            inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_config = GenerationConfig(
                temperature=temperature,
                do_sample=True,
            )
            generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_length, generation_config=generation_config)
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            return thread, streamer
        except Exception as e:
            raise RuntimeError(f"Failed to setup streaming generation: {str(e)}")
    
    def __generate_chat_output_from_model(model, tokenizer, sequence, max_length,temperature=0.1,trust_remote_code=False):
        # ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
        if tokenizer.chat_template is None:
            print("Warning: Chat template not found in tokenizer. Applying default chat template")
            tokenizer.chat_template = Generate.default_chat_template
        formatted_chat = tokenizer.apply_chat_template(sequence, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
        inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
        final_outputs = model.generate(
            **inputs, 
            max_new_tokens=max_length, 
            temperature=temperature)
        return inputs,final_outputs,tokenizer
    
    def __generate_chat_output_from_model_stream(model, tokenizer, sequence, max_length,temperature=0.1,trust_remote_code=False):
        try:
            # ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
            if tokenizer.chat_template is None:
                print("Warning: Chat template not found in tokenizer. Applying default chat template")
                tokenizer.chat_template = Generate.default_chat_template
            formatted_chat = tokenizer.apply_chat_template(sequence, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
            inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_config = GenerationConfig(
                temperature=temperature,
                do_sample=True,
            )
            generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_length, generation_config=generation_config)
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            return thread, streamer
        except Exception as e:
            raise RuntimeError(f"Failed to setup streaming generation: {str(e)}")
        
    def __generate_output_with_processor(model, processor, messages, max_length, temperature=0.1):
        try:
            if processor.chat_template is None:
                print("Warning: Chat template not found in processor. Applying default chat template")
                processor.chat_template = Generate.default_chat_template
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            with torch.inference_mode():
                final_outputs = model.generate(**inputs, max_new_tokens=max_length, do_sample=False, temperature=temperature)
            return inputs, final_outputs, processor
        except Exception as e:
            raise RuntimeError(f"Failed to generate output with processor: {str(e)}")

    def __generate_output_with_processor_stream(model, processor, messages, max_length, temperature=0.1):
        try:
            if processor.chat_template is None:
                print("Warning: Chat template not found in processor. Applying default chat template")
                processor.chat_template = Generate.default_chat_template
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
            generation_config = GenerationConfig(
                temperature=temperature,
                do_sample=True,
            )
            generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_length, generation_config=generation_config)
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            return thread, streamer
        except Exception as e:
            raise RuntimeError(f"Failed to generate output with processor stream: {str(e)}")
    def generate_output(model_path, sequence, max_length,trust_remote_code=False,gguf_file=None,auto_quantize=None):
        return Generate.__generate_output(model_path, sequence, max_length,trust_remote_code=trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize)
    
    def generate_output_stream(model_path, sequence, max_length,trust_remote_code=False,gguf_file=None,auto_quantize=None):
        return Generate.__generate_output_stream(model_path, sequence, max_length,trust_remote_code=trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize)
    
    def generate_text(model_path, sequence, max_length,trust_remote_code=False,gguf_file=None):
        return Generate.__generate_text(model_path, sequence, max_length,trust_remote_code=trust_remote_code,gguf_file=gguf_file)
    
    def load_tokenizer(tokenizer_path,gguf_file=None):
        return Generate.__load_tokenizer(tokenizer_path,gguf_file=gguf_file)
    
    def load_processor(tokenizer_path,use_fast=False,gguf_file=None):
        return Generate.__load_processor(tokenizer_path,use_fast=use_fast,gguf_file=gguf_file)
    
    def load_model(model_path,trust_remote_code=False,gguf_file = None,auto_quantize=None):
        return Generate.__load_model(model_path,trust_remote_code=trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize)
    
    def generate_output_from_model(model, tokenizer, sequence, max_length,trust_remote_code=False):
        return Generate.__generate_output_from_model(model, tokenizer, sequence, max_length,trust_remote_code=trust_remote_code)
    
    def generate_output_from_model_stream(model, tokenizer, sequence, max_length,trust_remote_code=False):
        return Generate.__generate_output_from_model_stream(model, tokenizer, sequence, max_length,trust_remote_code=trust_remote_code)
    
    def generate_chat_output(model_path, sequence, max_length,temperature = 0.1,trust_remote_code=False,gguf_file = None,auto_quantize=None):
        return Generate.__generate_chat_output(model_path, sequence, max_length,temperature = temperature,trust_remote_code=trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize)
    
    def generate_chat_output_stream(model_path, sequence, max_length,temperature = 0.1,trust_remote_code=False,gguf_file = None,auto_quantize=None):
        return Generate.__generate_chat_output_stream(model_path, sequence, max_length,temperature = temperature,trust_remote_code=trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize)
    
    def generate_chat_output_from_model(model, tokenizer, sequence, max_length,temperature=0.1,trust_remote_code=False):
        return Generate.__generate_chat_output_from_model(model, tokenizer, sequence, max_length,temperature=temperature,trust_remote_code=trust_remote_code)
    
    def generate_chat_output_from_model_stream(model, tokenizer, sequence, max_length,temperature=0.1,trust_remote_code=False):
        return Generate.__generate_chat_output_from_model_stream(model, tokenizer, sequence, max_length,temperature=temperature,trust_remote_code=trust_remote_code)
    
    def generate_output_with_processor(model, processor, messages, max_length, temperature=0.1):
        return Generate.__generate_output_with_processor(model, processor, messages, max_length, temperature)

    def generate_output_with_processor_stream(model, processor, messages, max_length, temperature=0.1):
        return Generate.__generate_output_with_processor_stream(model, processor, messages, max_length, temperature)