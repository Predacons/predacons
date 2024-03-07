from .load_data import LoadData
from .train_predacons import TrainPredacons
from .generate import Generate
from .data_preparation import DataPreparation
from .speculative_fast_generation import GPTFast

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
    print("\ngenerate_text -- Generate text")
    print("    model_path -- Model path")
    print("    sequence -- Sequence")
    print("    max_length -- Max length")
    print("    trust_remote_code -- Trust remote code (default False)")
    print("    use_fast_generation -- Use fast generation using speculative decoding (default False)")
    print("    draft_model_name -- Draft model name / path (default None)")
    print("\ngenerate_output -- returns output and tokenizer")
    print("    model_path -- Model path")
    print("    sequence -- Sequence")
    print("    max_length -- Max length")
    print("    trust_remote_code -- Trust remote code (default False)")
    print("    use_fast_generation -- Use fast generation using speculative decoding (default False)")
    print("    draft_model_name -- Draft model name / path (default None)")
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
    if use_fast_generation:
        if draft_model_name:
            if draft_model_name == None:
                print("Draft model is required for fast generation. Using base model as draft model, but it may increase memory utilization. try to use draft model name for better performance.")
                draft_model_name = model_path
            return GPTFast.generate_text_fast(model_path, draft_model_name, sequence, max_length,trust_remote_code = trust_remote_code)
        else:
            return Generate.generate_text(model_path, sequence, max_length,trust_remote_code = trust_remote_code) 

# Generate output
def generate_output(model_path, sequence, max_length,trust_remote_code = False,use_fast_generation=False, draft_model_name=None):
    if use_fast_generation:
        if draft_model_name:
            if draft_model_name == None:
                print("Draft model is required for fast generation. Using base model as draft model, but it may increase memory utilization. try to use draft model name for better performance.")
                draft_model_name = model_path
            return GPTFast.generate_output_fast(model_path, draft_model_name, sequence, max_length,trust_remote_code = trust_remote_code)
        else:
            return Generate.generate_output(model_path, sequence, max_length,trust_remote_code = trust_remote_code) 

# Data preparation
def generate_text_data_source_openai(client,gpt_model,prompt,number_of_examples,temperature =0.5):
    return DataPreparation.generate_text_data_source_openai(client,gpt_model,prompt,number_of_examples,temperature)

def generate_text_data_source_llm(model_path, sequence, max_length,number_of_examples,trust_remote_code=False):
    return DataPreparation.generate_text_data_source_llm(model_path, sequence, max_length,number_of_examples,trust_remote_code=trust_remote_code)

 

