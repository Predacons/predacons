from transformers import AutoModelForPreTraining, AutoTokenizer,AutoModelForCausalLM

class Generate:
    def __load_model(model_path, trust_remote_code=False):
        model = None
        try:
            model = AutoModelForPreTraining.from_pretrained(model_path,trust_remote_code=trust_remote_code)
        except:
            model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=trust_remote_code)
        return model


    def __load_tokenizer(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return tokenizer
    
    def __generate_output(model_path, sequence, max_length,trust_remote_code=False):
        model = Generate.__load_model(model_path,trust_remote_code=trust_remote_code)
        tokenizer = Generate.__load_tokenizer(model_path)
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
    
    def __generate_text(model_path, sequence, max_length,trust_remote_code=False):
        final_outputs,tokenizer = Generate.__generate_output(model_path, sequence, max_length,trust_remote_code=trust_remote_code)
        
        print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))
        return (tokenizer.decode(final_outputs[0], skip_special_tokens=True))

    def generate_output(model_path, sequence, max_length,trust_remote_code=False):
        return Generate.__generate_output(model_path, sequence, max_length,trust_remote_code=trust_remote_code)
    
    def generate_text(model_path, sequence, max_length,trust_remote_code=False):
        return Generate.__generate_text(model_path, sequence, max_length,trust_remote_code=trust_remote_code)