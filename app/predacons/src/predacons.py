import load_data
import train_predacons
import generate

def rollout():
    print("Predacons rollout !!!")
    print("read_documents_from_directory -- Load data from directory")
    print("clean_text -- Clean text")
    print("train -- Train Predacons")
    print("generate_text -- Generate text")

# Load data
def read_documents_from_directory(directory,encoding="utf-8"):
    return load_data.LoadData.read_documents_from_directory(directory,encoding)
def clean_text(text):
    return load_data.LoadData.clean_text(text)

# Train Predacons
def train(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):
    train_predacons.TrainPredacons.train(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps)
    
# Generate text
def generate_text(model_path, sequence, max_length):
    generate.Generate.generate_text(model_path, sequence, max_length)

