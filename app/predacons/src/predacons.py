from .load_data import LoadData
from .train_predacons import TrainPredacons
from .generate import Generate

def rollout():
    print("Predacons rollout !!!")
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
    print("    trust_remote_code -- Trust remote code")
    print("\ngenerate_output -- returns output and tokenizer")
    print("    model_path -- Model path")
    print("    sequence -- Sequence")
    print("    max_length -- Max length")
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
          trust_remote_code = False):
    TrainPredacons.train(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps,
          trust_remote_code = trust_remote_code)

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
def generate_text(model_path, sequence, max_length,trust_remote_code = False):
    return Generate.generate_text(model_path, sequence, max_length,trust_remote_code = trust_remote_code)

# Generate output
def generate_output(model_path, sequence, max_length,trust_remote_code = False):
    return Generate.generate_output(model_path, sequence, max_length,trust_remote_code = trust_remote_code)

