from transformers import AutoModelForPreTraining, AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
import torch


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
    
    def __generate_chat_output(model_path, sequence, max_length,temprature = 0.1,trust_remote_code=False,gguf_file=None,auto_quantize = None):
        model = Generate.__load_model(model_path,trust_remote_code=trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize)
        tokenizer = Generate.__load_tokenizer(model_path,gguf_file=gguf_file)
        if tokenizer.chat_template is None:
            print("Warning: Chat template not found in tokenizer. Appling default chat template")
            tokenizer.chat_template = Generate.default_chat_template
        formatted_chat = tokenizer.apply_chat_template(sequence, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
        inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
        final_outputs = model.generate(
            **inputs, 
            max_new_tokens=max_length, 
            temperature=temprature)
        return inputs,final_outputs,tokenizer
    
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
    
    def __generate_chat_output_from_model(model, tokenizer, sequence, max_length,temprature=0.1,trust_remote_code=False):
        # ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
        if tokenizer.chat_template is None:
            print("Warning: Chat template not found in tokenizer. Appling default chat template")
            tokenizer.chat_template = Generate.default_chat_template
        formatted_chat = tokenizer.apply_chat_template(sequence, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
        inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
        final_outputs = model.generate(
            **inputs, 
            max_new_tokens=max_length, 
            temperature=temprature)
        return inputs,final_outputs,tokenizer

    def generate_output(model_path, sequence, max_length,trust_remote_code=False,gguf_file=None,auto_quantize=None):
        return Generate.__generate_output(model_path, sequence, max_length,trust_remote_code=trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize)
    
    def generate_text(model_path, sequence, max_length,trust_remote_code=False,gguf_file=None):
        return Generate.__generate_text(model_path, sequence, max_length,trust_remote_code=trust_remote_code,gguf_file=gguf_file)
    
    def load_tokenizer(tokenizer_path,gguf_file=None):
        return Generate.__load_tokenizer(tokenizer_path,gguf_file=gguf_file)
    
    def load_model(model_path,trust_remote_code=False,gguf_file = None,auto_quantize=None):
        return Generate.__load_model(model_path,trust_remote_code=trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize)
    
    def generate_output_from_model(model, tokenizer, sequence, max_length,trust_remote_code=False):
        return Generate.__generate_output_from_model(model, tokenizer, sequence, max_length,trust_remote_code=trust_remote_code)
    
    def generate_chat_output(model_path, sequence, max_length,temprature = 0.1,trust_remote_code=False,gguf_file = None,auto_quantize=None):
        return Generate.__generate_chat_output(model_path, sequence, max_length,temprature = temprature,trust_remote_code=trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize)
    
    def generate_chat_output_from_model(model, tokenizer, sequence, max_length,temprature=0.1,trust_remote_code=False):
        return Generate.__generate_chat_output_from_model(model, tokenizer, sequence, max_length,temprature=temprature,trust_remote_code=trust_remote_code)
    
