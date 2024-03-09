from .load_data import LoadData
from .train_predacons import TrainPredacons
from .generate import Generate
from .data_preparation import DataPreparation
from .speculative_fast_generation import GPTFast
import torch

def rollout():
    print("Predacons rollout !!!")
    print("Predacons Version: v0.0.110")
    print("\nread_documents_from_directory -- Load data from directory")
    print("    directory -- Directory path")
    print("\nread_multiple_files -- Load data from multiple files")
    print("    file_paths -- list of File paths")
    print("\nclean_text -- Clean text")
    print("    text -- Text")
    print("\ntrain -- Train Predacons")
    print("    train_file_path -- Train file path")
    print("    model_name -- Model name")
    print("    output_dir -- Output directory")
    print("    overwrite_output_dir -- Overwrite output directory")
    print("    per_device_train_batch_size -- Per device train batch size")
    print("    num_train_epochs -- Number of train epochs")
    print("    save_steps -- Save steps")
    print("    trust_remote_code -- Trust remote code")
    print("\ntrainer -- returns trainer")
    print("    train_file_path -- Train file path")
    print("    model_name -- Model name")
    print("    output_dir -- Output directory")
    print("    overwrite_output_dir -- Overwrite output directory")
    print("    per_device_train_batch_size -- Per device train batch size")
    print("    num_train_epochs -- Number of train epochs")
    print("    save_steps -- Save steps")
    print("    trust_remote_code -- Trust remote code")
    print("\ngenerate_text -- Generate text (Deprecating soon, use text_generate instead)")
    print("    model_path -- Model path")
    print("    sequence -- Sequence")
    print("    max_length -- Max length")
    print("    trust_remote_code -- Trust remote code (default False)")
    print("    use_fast_generation -- Use fast generation using speculative decoding (default False)")
    print("    draft_model_name -- Draft model name / path (default None)")
    print("\ngenerate_output -- returns output and tokenizer (Deprecating soon, use generate instead)")
    print("    model_path -- Model path")
    print("    sequence -- Sequence")
    print("    max_length -- Max length")
    print("    trust_remote_code -- Trust remote code (default False)")
    print("    use_fast_generation -- Use fast generation using speculative decoding (default False)")
    print("    draft_model_name -- Draft model name / path (default None)")
    print("\ngenerate -- Generate text or output")
    print("    model_path -- Model path")
    print("    sequence -- Sequence")
    print("    max_length -- Max length")
    print("    trust_remote_code -- Trust remote code (default False)")
    print("    use_fast_generation -- Use fast generation using speculative decoding (default False)")
    print("    draft_model_name -- Draft model name / path (default None)")
    print("    model -- give a preloaded Model (default None)")
    print("    tokenizer -- give a preloaded Tokenizer (default None)")
    print("\ntext_generate -- Generate text and print")
    print("    model_path -- Model path")
    print("    sequence -- Sequence")
    print("    max_length -- Max length")
    print("    trust_remote_code -- Trust remote code (default False)")
    print("    use_fast_generation -- Use fast generation using speculative decoding (default False)")
    print("    draft_model_name -- Draft model name / path (default None)")
    print("    model -- give a preloaded Model (default None)")
    print("    tokenizer -- give a preloaded Tokenizer (default None)")
    print("\nload_model -- Load model")
    print("    model_path -- Model path")
    print("    trust_remote_code -- Trust remote code (default False)")
    print("    use_fast_generation -- Use fast generation using speculative decoding (default False)")
    print("    draft_model_name -- Draft model name / path (default None)")
    print("\nload_tokenizer -- Load tokenizer")
    print("    tokenizer_path -- Tokenizer path")
    print("\ngenerate_text_data_source_openai -- Generate text data source using openai")
    print("    client -- openai client")
    print("    gpt_model -- GPT model used for generation")
    print("    prompt -- Prompt to generate data source")
    print("    number_of_examples -- Number of examples")
    print("    temperature -- Temperature (default 0.5)")
    print("\ngenerate_text_data_source_ll -- Generate text data source using local or hugging face llm")
    print("    model_path -- Model path or hugging face model name")
    print("    sequence -- Sequence (prompt)")
    print("    max_length -- Max length of the generated text")
    print("    number_of_examples -- Number of examples")
    print("    trust_remote_code -- Trust remote code")
    print("\nPredacons rollout !!!")

# Load data
def read_documents_from_directory(directory,encoding="utf-8"):
    return LoadData.read_documents_from_directory(directory,encoding)

def read_multiple_files(file_paths):
    return LoadData.read_multiple_files(file_paths)

def clean_text(text):
    return LoadData.clean_text(text)

# Train Predacons
def train(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps,
          trust_remote_code = False,
          resume_from_checkpoint=True):
    TrainPredacons.train(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps,
          trust_remote_code = trust_remote_code,
          resume_from_checkpoint=resume_from_checkpoint)

# get trainer
def trainer(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps,
          trust_remote_code = False):
    return TrainPredacons.trainer(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps,
          trust_remote_code = trust_remote_code)
    
# Generate text
def generate_text(model_path, sequence, max_length,trust_remote_code = False,use_fast_generation=False, draft_model_name=None):
    print("For repetitive generation first load model and then use text_generate. It will be faster.")
    print("will be deprecated soon, use text_generate")
    if use_fast_generation:
        print("generate_text using fast generation")
        if draft_model_name == None:
            print("Draft model is required for fast generation. Using base model as draft model, but it may increase memory utilization. try to use draft model name for better performance.")
            draft_model_name = model_path
        return GPTFast.generate_text_fast(model_path, draft_model_name, sequence, max_length,trust_remote_code = trust_remote_code)
    else:
        print("generate_text using default generation")
        return Generate.generate_text(model_path, sequence, max_length,trust_remote_code = trust_remote_code) 

# Generate output
def generate_output(model_path, sequence, max_length,trust_remote_code = False,use_fast_generation=False, draft_model_name=None):
    print("For repetitive generation first load model and then use generate. It will be faster.")
    print("will be deprecated soon, use generate instead")
    if use_fast_generation:
        print("generate_output using fast generation")
        if draft_model_name == None:
            print("Draft model is required for fast generation. Using base model as draft model, but it may increase memory utilization. try to use draft model name for better performance.")
            draft_model_name = model_path
        return GPTFast.generate_output_fast(model_path, draft_model_name, sequence, max_length,trust_remote_code = trust_remote_code)
    else:
        print("generate_output using default generation")
        return Generate.generate_output(model_path, sequence, max_length,trust_remote_code = trust_remote_code) 

# generate new

def generate(*args, **kwargs):
    if 'model_path' in kwargs and 'sequence' in kwargs:
        model_path = kwargs['model_path']
        sequence = kwargs['sequence']
        max_length = kwargs.get('max_length', 50)
        trust_remote_code = kwargs.get('trust_remote_code', False)
        use_fast_generation = kwargs.get('use_fast_generation', False)
        draft_model_name = kwargs.get('draft_model_name', None)
        if use_fast_generation:
            print("generate_output using fast generation")
            if draft_model_name == None:
                print("Draft model is required for fast generation. Using base model as draft model, but it may increase memory utilization. try to use draft model name for better performance.")
                draft_model_name = model_path
            return GPTFast.generate_output_fast(model_path, draft_model_name, sequence, max_length,trust_remote_code = trust_remote_code)
        else:
            print("generate_output using default generation")
            return Generate.generate_output(model_path, sequence, max_length,trust_remote_code = trust_remote_code) 
    
    elif 'model' in kwargs and 'tokenizer' in kwargs and 'sequence' in kwargs:
        model = kwargs['model']
        tokenizer = kwargs['tokenizer']
        sequence = kwargs['sequence']
        max_length = kwargs.get('max_length', 50)
        trust_remote_code = kwargs.get('trust_remote_code', False)
        if type(model) == torch._dynamo.eval_frame.OptimizedModule:
            print("generate_output using fast generation")
            return GPTFast.generate_output_from_model(model, tokenizer, sequence, max_length)
        else:
            return Generate.generate_output_from_model(model, tokenizer, sequence, max_length,trust_remote_code=trust_remote_code)
    else:
        raise ValueError("Invalid arguments")
    
def text_generate(*args, **kwargs):
    output,tokenizer = generate(*args, **kwargs)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Data preparation
def generate_text_data_source_openai(client,gpt_model,prompt,number_of_examples,temperature =0.5):
    return DataPreparation.generate_text_data_source_openai(client,gpt_model,prompt,number_of_examples,temperature)

def generate_text_data_source_llm(model_path, sequence, max_length,number_of_examples,trust_remote_code=False):
    return DataPreparation.generate_text_data_source_llm(model_path, sequence, max_length,number_of_examples,trust_remote_code=trust_remote_code)

# Get model and tokenizer
def load_model(model_path,trust_remote_code=False,use_fast_generation=False, draft_model_name=None):
    if use_fast_generation:
        print("load_model using fast generation")
        if draft_model_name == None:
            print("Draft model is required for fast generation. Using base model as draft model, but it may increase memory utilization. try to use draft model name for better performance.")
            draft_model_name = model_path
        return GPTFast.load_model(model_path, draft_model_name,trust_remote_code=trust_remote_code)
    else:
        print("load_model using default generation")
        return Generate.load_model(model_path,trust_remote_code=trust_remote_code)

def load_tokenizer(tokenizer_path):
    return Generate.load_tokenizer(tokenizer_path)
 

