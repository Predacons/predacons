from transformers import AutoModelForPreTraining, AutoTokenizer

class Generate:
    def __load_model(model_path):
        model = AutoModelForPreTraining.from_pretrained(model_path)
        return model


    def __load_tokenizer(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return tokenizer

    def __generate_text(model_path, sequence, max_length):
        
        model = Generate.__load_model(model_path)
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
        print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))

    def generate_text(model_path, sequence, max_length):
        Generate.__generate_text(model_path, sequence, max_length)