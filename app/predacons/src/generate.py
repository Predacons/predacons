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
        """
        Loads a tokenizer from the specified path.
        
        Args:
            tokenizer_path: Path to the pretrained tokenizer.
            gguf_file: Optional file for GGUF format support.
        
        Returns:
            The loaded tokenizer instance.
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,gguf_file=gguf_file)
        return tokenizer
    
    def __load_processor(tokenizer_path,use_fast=False,gguf_file=None):
        """
        Loads a processor from the specified path.
        
        Args:
            tokenizer_path: Path to the pretrained processor or model directory.
            use_fast: Whether to use the fast implementation if available.
            gguf_file: Optional GGUF file for processor configuration.
        
        Returns:
            An instance of AutoProcessor loaded from the given path.
        """
        processor = AutoProcessor.from_pretrained(tokenizer_path, use_fast=use_fast, gguf_file=gguf_file)
        return processor
    def __generate_output(model_path, sequence, max_length,trust_remote_code=False,gguf_file=None,auto_quantize=None):
        """
        Generates output token IDs from a pretrained model given an input sequence.
        
        Loads the specified model and tokenizer, encodes the input sequence, and generates output tokens using sampling with top-k and top-p settings.
        
        Args:
            model_path: Path to the pretrained model.
            sequence: Input text sequence to generate output from.
            max_length: Maximum length of the generated output.
        
        Returns:
            A tuple containing the generated output token IDs and the tokenizer used.
        """
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
        """
        Streams chat-style text generation from a model using a tokenizer and input sequence.
        
        Formats the input sequence as a chat prompt using a chat template, tokenizes it, and initiates streaming generation in a separate thread. Returns the thread and a streamer for consuming generated text in real time.
        
        Raises:
            RuntimeError: If streaming generation setup fails.
        """
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
        """
        Generates model outputs from chat messages using a processor for input preparation.
        
        If the processor lacks a chat template, a default template is applied. The messages are formatted and tokenized using the processor, moved to the model's device, and passed to the model for generation without sampling. Returns the prepared inputs, generated outputs, and the processor.
        """
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
        """
        Performs streaming text generation using a processor and a language model.
        
        Applies a chat template to the input messages, tokenizes them, and initiates streaming generation in a separate thread. Returns the thread and a streamer for consuming generated text in real time.
        
        Raises:
            RuntimeError: If streaming generation setup fails.
        """
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
        """
        Generates output token IDs from a language model given an input sequence.
        
        Loads the specified model and tokenizer, encodes the input sequence, and generates output tokens using sampling with a maximum output length.
        
        Returns:
            A tuple containing the generated token IDs and the tokenizer instance.
        """
        return Generate.__generate_output(model_path, sequence, max_length,trust_remote_code=trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize)
    
    def generate_output_stream(model_path, sequence, max_length,trust_remote_code=False,gguf_file=None,auto_quantize=None):
        return Generate.__generate_output_stream(model_path, sequence, max_length,trust_remote_code=trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize)
    
    def generate_text(model_path, sequence, max_length,trust_remote_code=False,gguf_file=None):
        return Generate.__generate_text(model_path, sequence, max_length,trust_remote_code=trust_remote_code,gguf_file=gguf_file)
    
    def load_tokenizer(tokenizer_path,gguf_file=None):
        """
        Loads and returns a tokenizer from the specified path.
        
        Args:
            tokenizer_path: Path to the tokenizer directory or file.
            gguf_file: Optional file for GGUF format support.
        
        Returns:
            The loaded tokenizer instance.
        """
        return Generate.__load_tokenizer(tokenizer_path,gguf_file=gguf_file)
    
    def load_processor(tokenizer_path,use_fast=False,gguf_file=None):
        """
        Loads and returns a processor from the specified path.
        
        Args:
            tokenizer_path: Path to the processor or tokenizer directory.
            use_fast: Whether to use the fast implementation, if available.
            gguf_file: Optional path to a GGUF file for processor configuration.
        
        Returns:
            An instance of the loaded processor.
        """
        return Generate.__load_processor(tokenizer_path,use_fast=use_fast,gguf_file=gguf_file)
    
    def load_model(model_path,trust_remote_code=False,gguf_file = None,auto_quantize=None):
        """
        Loads a pretrained model from the specified path with optional quantization and trust settings.
        
        Args:
            model_path: Path to the pretrained model directory or file.
            trust_remote_code: Whether to allow execution of custom code from the model repository.
            gguf_file: Optional file for GGUF format models.
            auto_quantize: Optional quantization mode ("4bit", "8bit", "high", "low").
        
        Returns:
            The loaded model instance.
        """
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
        """
        Streams chat-style text generation output from a pre-loaded model and tokenizer.
        
        Returns:
            A tuple containing the thread handling generation and a streamer for iterating over generated text.
        """
        return Generate.__generate_chat_output_from_model_stream(model, tokenizer, sequence, max_length,temperature=temperature,trust_remote_code=trust_remote_code)
    
    def generate_output_with_processor(model, processor, messages, max_length, temperature=0.1):
        """
        Generates text output from a model using a processor and a list of chat messages.
        
        Args:
            messages: A list of chat messages to be processed and used as input.
            max_length: The maximum number of tokens to generate.
            temperature: Sampling temperature for generation, controlling randomness.
        
        Returns:
            A tuple containing the processed input tensors, generated output tensors, and the processor instance.
        """
        return Generate.__generate_output_with_processor(model, processor, messages, max_length, temperature)

    def generate_output_with_processor_stream(model, processor, messages, max_length, temperature=0.1):
        """
        Streams generated text output from a model using a processor and chat-style messages.
        
        Returns:
            A tuple containing the thread handling generation and a streamer for iterating over generated text.
        """
        return Generate.__generate_output_with_processor_stream(model, processor, messages, max_length, temperature)