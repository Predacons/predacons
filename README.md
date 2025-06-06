# Predacons
Predacons is a Python library based on transformers used for transfer learning. It offers a suite of tools for data processing, model training, and text generation, making it easier to apply advanced machine learning techniques to your projects.

![PyPI](https://img.shields.io/pypi/v/predacons)   ![Downloads](https://img.shields.io/pypi/dm/predacons)   ![License](https://img.shields.io/pypi/l/predacons)   ![Python Version](https://img.shields.io/pypi/pyversions/predacons)

## Features
Predacons provides a comprehensive set of features for working with transformer models, including:

- **Data Loading**: Easily load data from directories or files.
- **Text Cleaning**: Clean your text data with built-in functions.
- **Model Training**: Train transformer models with custom data.
- **Text Generation**: Generate text using trained models.
- **Text Streaming**: Stream text generation using trained models.
- **Chat Generation**: Generate chat responses using trained models.
- **Chat Streaming**: Stream chat generation using trained models.
- **Embeddings**: Generate embeddings for sentences using pre-trained transformer models. and is fully compatible with langchain methods


## Installation
To install Predacons, use the following pip command:
```bash
pip install predacons
```

## Usage
Here's a quick start guide to using Predacons in your Python projects:

```python
from predacons import predacons

# Initialize the library
predacons.rollout()

# Load documents from a directory
predacons.read_documents_from_directory('your/directory/path')

# Clean text data
cleaned_text = predacons.clean_text("Your dirty text here")

# Train a model with your data
predacons.train(train_file_path="path/to/train/file",
                model_name="your_model_name",
                output_dir="path/to/output/dir",
                overwrite_output_dir=True,
                per_device_train_batch_size=4,
                num_train_epochs=3,
                save_steps=100)

# Generate text using a trained model
generated_text = predacons.generate_text(model_path="path/to/your/model",
                                         sequence="Seed text for generation",
                                         max_length=50)
# 

# Stream text generation using a trained model
for text in predacons.text_stream(model_path="path/to/your/model",
                                  sequence="Seed text for generation",
                                  max_length=50):
    print(text)

# Get text streamer
thread,streamer = predacons.text_generate(model=model, tokenizer=tokenizer, sequence=seq, max_length=100, temperature=0.1, stream=True)

# You can also use a processor instead of a tokenizer for model-based generation:
thread,streamer = predacons.text_generate(model=model, processor=processor, sequence=seq, max_length=100, temperature=0.1, stream=True)

thread.start()
try:
    out = ""
    for new_text in streamer:
        out = out + new_text
        print(new_text, end=" ")
finally:
    thread.join()

# Generate chat using a trained model
chat = [
    {"role": "user", "content": "Hey, what is a car?"}
]
chat_output = predacons.chat_generate(model=model,
        sequence=chat,
        max_length=50,
        tokenizer=tokenizers,
        trust_remote_code=True)
# You can also use a processor instead of a tokenizer for chat generation:
chat_output = predacons.chat_generate(model=model,
        sequence=chat,
        max_length=50,
        processor=processor,
        trust_remote_code=True)

# Stream chat generation using a trained model
for chat in predacons.chat_stream(model=model,
                                  sequence=chat,
                                  max_length=50,
                                  tokenizer=tokenizers,
                                  trust_remote_code=True):
    print(chat)
# You can also use a processor instead of a tokenizer for chat streaming:
for chat in predacons.chat_stream(model=model,
                                  sequence=chat,
                                  max_length=50,
                                  processor=processor,
                                  trust_remote_code=True):
    print(chat)

# get chat streamer
thread,streamer = predacons.chat_generate(model=model, tokenizer = tokenizer, sequence = chat, max_length=500, temperature=0.1,stream=True)

thread.start()
try:
    out = ""
    for new_text in streamer:
        out = out + new_text
        print(new_text, end="")
finally:
    thread.join()
# Generate embeddings for sentences
from predacons.src.embeddings import PredaconsEmbedding

# this embedding_model object can be used directly in every method langchain   
embedding_model = PredaconsEmbedding(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
sentence_embeddings = embedding_model.get_embedding(["Your sentence here", "Another sentence here"])
```
## Contributing
Contributions to the Predacons library are welcome! If you have suggestions for improvements or new features, please open an issue first to discuss your ideas. For code contributions, please submit a pull request.

## License

This project is licensed under multiple licenses:

- For **free users**, the project is licensed under the terms of the GNU Affero General Public License (AGPL). See  [`LICENSE-AGPL`](LICENSE-AGPL) for more details.

- For **paid users**, there are two options:
    - A perpetual commercial license. See [`LICENSE-COMMERCIAL-PERPETUAL`](LICENSE-COMMERCIAL-PERPETUAL) for more details.
    - A yearly commercial license. See [`LICENSE-COMMERCIAL-YEARLY`](LICENSE-COMMERCIAL-YEARLY) for more details.

Please ensure you understand and comply with the license that applies to you.

