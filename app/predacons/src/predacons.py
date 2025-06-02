from .load_data import LoadData
from .train_predacons import TrainPredacons
from .generate import Generate
from .data_preparation import DataPreparation
from .speculative_fast_generation import GPTFast
import torch
import pandas as pd

def rollout():
    """
    Prints version information and a comprehensive usage guide for all available Predacons functions.
    
    Displays descriptions and parameter details for data loading, training, text generation, streaming, chat, model/tokenizer loading, and data preparation functions provided by the Predacons module.
    """
    print("Predacons rollout !!!")
    print("Predacons Version: v0.0.130")
    print("\nread_documents_from_directory -- Load data from directory")
    print("    directory -- Directory path")
    print("\nread_multiple_files -- Load data from multiple files")
    print("    file_paths -- list of File paths")
    print("\nclean_text -- Clean text")
    print("    text -- Text")
    print("\nread_csv -- Read csv file")
    print("    file_path -- File path")
    print("\ntrain_legacy -- Train Predacons")
    print("    train_file_path -- Train file path")
    print("    model_name -- Model name")
    print("    output_dir -- Output directory")
    print("    overwrite_output_dir -- Overwrite output directory")
    print("    per_device_train_batch_size -- Per device train batch size")
    print("    num_train_epochs -- Number of train epochs")
    print("    save_steps -- Save steps")
    print("    trust_remote_code -- Trust remote code")
    print("\ntrainer_legacy -- returns trainer")
    print("    train_file_path -- Train file path")
    print("    model_name -- Model name")
    print("    output_dir -- Output directory")
    print("    overwrite_output_dir -- Overwrite output directory")
    print("    per_device_train_batch_size -- Per device train batch size")
    print("    num_train_epochs -- Number of train epochs")
    print("    save_steps -- Save steps")
    print("    trust_remote_code -- Trust remote code")
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
    print("    apply_chat_template -- use chat template (defauly False)")
    print("\ngenerate -- Generate text or output")
    print("    model_path -- Model path")
    print("    sequence -- Sequence")
    print("    max_length -- Max length")
    print("    trust_remote_code -- Trust remote code (default False)")
    print("    use_fast_generation -- Use fast generation using speculative decoding (default False)")
    print("    draft_model_name -- Draft model name / path (default None)")
    print("    model -- give a preloaded Model (default None)")
    print("    tokenizer -- give a preloaded Tokenizer (default None)")
    print("    processor -- give a preloaded Processor (default None), alternative to tokenizer for model-based generation")
    print("\ntext_generate -- Generate text and print")
    print("    model_path -- Model path")
    print("    sequence -- Sequence")
    print("    max_length -- Max length")
    print("    trust_remote_code -- Trust remote code (default False)")
    print("    use_fast_generation -- Use fast generation using speculative decoding (default False)")
    print("    draft_model_name -- Draft model name / path (default None)")
    print("    model -- give a preloaded Model (default None)")
    print("    tokenizer -- give a preloaded Tokenizer (default None)")
    print("    processor -- give a preloaded Processor (default None), alternative to tokenizer for model-based generation")
    print("\ntext_stream -- stream text and print")
    print("    model_path -- Model path")
    print("    sequence -- Sequence")
    print("    max_length -- Max length")
    print("    trust_remote_code -- Trust remote code (default False)")
    print("\nchat_generate -- Generate chat and print")
    print("    model_path -- Model path")
    print("    sequence -- Sequence")
    print("    max_length -- Max length")
    print("    trust_remote_code -- Trust remote code (default False)")
    print("    use_fast_generation -- Use fast generation using speculative decoding (default False)")
    print("    draft_model_name -- Draft model name / path (default None)")
    print("    model -- give a preloaded Model (default None)")
    print("    tokenizer -- give a preloaded Tokenizer (default None)")
    print("    apply_chat_template -- use chat template (defauly False)")
    print("    dont_print_output -- Dont print output (default False)")
    print("    gguf_file -- GGUF file path (default None)")
    print("    auto_quantize -- Automatically apply quantization (default None)")
    print("\nchat_stream -- Stream chat and print")
    print("    model_path -- Model path")
    print("    sequence -- Sequence")
    print("    max_length -- Max length")
    print("    trust_remote_code -- Trust remote code (default False)")
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
def read_documents_from_directory(directory, encoding="utf-8"):
    """
    Read documents from a directory.

    Args:
        directory (str): The path to the directory containing the documents.
        encoding (str, optional): The encoding of the documents. Defaults to "utf-8".

    Returns:
        list: A list of documents read from the directory.
    """
    return LoadData.read_documents_from_directory(directory, encoding)

def read_multiple_files(file_paths):
    """
    Read and load data from multiple files.

    Args:
        file_paths (list): A list of file paths to read data from.

    Returns:
        object: The loaded data.

    """
    return LoadData.read_multiple_files(file_paths)

def clean_text(text):
    """
    Cleans the given text by removing any unwanted characters or formatting.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.

    """
    return LoadData.clean_text(text)

def read_csv(file_path, encoding="utf-8"):
    """
    Read a CSV file and return the data as a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the CSV file.
    - encoding (str, optional): The encoding of the CSV file. Default is 'utf-8'.

    Returns:
    - pandas.DataFrame: The data from the CSV file.

    Example:
    >>> data = read_csv('/path/to/file.csv')
    """
    return LoadData.read_csv(file_path, encoding)

# Train Predacons
def train_legacy(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps,
          trust_remote_code = False,
          resume_from_checkpoint=True):
    """
    Trains the Predacons model using legacy training method.

    Args:
        train_file_path (str): The path to the training file.
        model_name (str): The name of the model.
        output_dir (str): The directory to save the trained model.
        overwrite_output_dir (bool): Whether to overwrite the output directory if it already exists.
        per_device_train_batch_size (int): The batch size for training.
        num_train_epochs (int): The number of training epochs.
        save_steps (int): The number of steps between saving checkpoints.
        trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
        resume_from_checkpoint (bool, optional): Whether to resume training from a checkpoint. Defaults to True.
    """
    TrainPredacons.train(train_file_path = train_file_path,
        model_name = model_name,
        output_dir = output_dir,
        overwrite_output_dir = overwrite_output_dir,
        per_device_train_batch_size = per_device_train_batch_size,
        num_train_epochs = num_train_epochs,
        save_steps = save_steps,
        trust_remote_code = trust_remote_code,
        resume_from_checkpoint=resume_from_checkpoint)

# get trainer
def trainer_legacy(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps,
          trust_remote_code = False):
    """
    Trains the Predacons model using legacy training method.

    Args:
        train_file_path (str): The path to the training file.
        model_name (str): The name of the model.
        output_dir (str): The directory to save the trained model.
        overwrite_output_dir (bool): Whether to overwrite the output directory if it already exists.
        per_device_train_batch_size (int): The batch size for training.
        num_train_epochs (int): The number of training epochs.
        save_steps (int): The number of steps between saving checkpoints.
        trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
        resume_from_checkpoint (bool, optional): Whether to resume training from a checkpoint. Defaults to True.
    """
    return TrainPredacons.trainer(train_file_path=train_file_path,
        model_name = model_name,
        output_dir = output_dir,
        overwrite_output_dir = overwrite_output_dir,
        per_device_train_batch_size = per_device_train_batch_size,
        num_train_epochs = num_train_epochs,
        save_steps = save_steps,
        trust_remote_code = trust_remote_code)

def train(*args, **kwargs):
    return TrainPredacons.train(*args, **kwargs)

def trainer(*args, **kwargs):
    """
    Prepares a trainer instance for the Predacons model with the provided arguments and keyword arguments.
    
    This function serves as a wrapper that calls the `trainer` method of the `TrainPredacons` class, forwarding all
    received arguments and keyword arguments. It is designed to configure and return a trainer instance without
    immediately starting the training process, allowing for further customization or inspection of the trainer
    configuration before training.
    
    ## Parameters

    - `*args`: Arbitrary positional arguments. Currently, this method does not utilize positional arguments but is designed to be flexible for future extensions.

    - `**kwargs`: Arbitrary keyword arguments used for configuring the training process. The supported keywords include:

    - `use_legacy_trainer` (bool): If `True`, uses a legacy training approach. Default is `False`.
    - `model_name` (str): The name or path of the pre-trained model to be used.
    - `train_file_path` (str): Path to the training dataset file.
    - `tokenizer` (Tokenizer): An instance of a tokenizer.
    - `output_dir` (str): Directory where the model and tokenizer will be saved after training.
    - `overwrite_output_dir` (bool): If `True`, overwrite the output directory.
    - `per_device_train_batch_size` (int): Batch size per device during training.
    - `num_train_epochs` (int): Total number of training epochs.
    - `quantization_config` (dict): Configuration for model quantization.
    - `auto_quantize` (str): Automatically apply quantization. Accepts "4bit"/"high" for high compression or "8bit"/"low" for lower compression.
    - `trust_remote_code` (bool): If `True`, allows the execution of remote code during model loading.
    - `peft_config` (dict): Configuration for Parameter Efficient Fine-Tuning (PEFT) techniques like LoRA.
    - `auto_lora_config` (bool): If `True`, automatically configures LoRA for the model.
    - `training_args` (TrainingArguments): Configuration arguments for the Hugging Face `Trainer`.
    - `train_file_type` (str): Type of the training file. Supported types are "text", "csv", "json".
    - `train_dataset` (Dataset): A pre-loaded dataset. If provided, `train_file_path` is ignored.
    - `preprcess_function` (callable): A function to preprocess the dataset.
    - `resume_from_checkpoint` (str): Path to a directory containing a checkpoint from which training is to resume.
    - `save_steps` (int): Number of steps after which the model is saved.

    ## Returns

    - Returns an instance of `Trainer` or `SFTTrainer`, configured according to the provided arguments. This object is ready to be used for training the model.

    ## Example Usage

    ```python
    trainer = TrainPredacons.trainer(
        model_name='bert-base-uncased',
        train_file_path='./data/train.txt',
        tokenizer=my_tokenizer,
        output_dir='./model_output',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        auto_quantize='4bit'
    )
    """
    return TrainPredacons.trainer(*args, **kwargs)

# Generate text
def generate_text(model_path, sequence, max_length,trust_remote_code = False,use_fast_generation=False, draft_model_name=None,gguf_file=None):
    """
    Generate text using the specified model.

    Args:
        model_path (str): The path to the model.
        sequence (str): The input sequence to generate text from.
        max_length (int): The maximum length of the generated text.
        trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
        use_fast_generation (bool, optional): Whether to use fast generation. Defaults to False.
        draft_model_name (str, optional): The name of the draft model. Defaults to None.

    Returns:
        str: The generated text.
    """
    print("For repetitive generation first load model and then use text_generate. It will be faster.")
    print("will be deprecated soon, use text_generate")
    if use_fast_generation:
        print("generate_text using fast generation")
        if draft_model_name == None:
            print("Draft model is required for fast generation. Using base model as draft model, but it may increase memory utilization. try to use draft model name for better performance.")
            draft_model_name = model_path
        return GPTFast.generate_text_fast(model_path, draft_model_name, sequence, max_length,trust_remote_code = trust_remote_code,gguf_file=gguf_file)
    else:
        print("generate_text using default generation")
        return Generate.generate_text(model_path, sequence, max_length,trust_remote_code = trust_remote_code,gguf_file=gguf_file) 

# Generate output
def generate_output(model_path, sequence, max_length,trust_remote_code = False,use_fast_generation=False, draft_model_name=None,temperature=0.1,apply_chat_template = False,gguf_file=None,auto_quantize=None):
    """
    Generates output using the specified model.

    Args:
        model_path (str): The path to the model.
        sequence (str): The input sequence for generating the output.
        max_length (int): The maximum length of the generated output.
        trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
        use_fast_generation (bool, optional): Whether to use fast generation. Defaults to False.
        draft_model_name (str, optional): The name of the draft model. Defaults to None.
        temperature (float, optional): The temperature parameter for controlling the randomness of the generated output. Defaults to 0.1.
        apply_chat_template (bool, optional): Whether to apply the chat template. Defaults to False.
        gguf_file (str, optional): The path to the GGUF file. Defaults to None.
        auto_quantize (str, optional): Automatically apply quantization. Accepts "4bit"/"high" for high compression or "8bit"/"low" for lower compression. Defaults to None.

    Returns:
        str: The generated output.
    """
    print("For repetitive generation first load model and then use generate. It will be faster.")
    print("will be deprecated soon, use generate instead")
    if use_fast_generation:
        if apply_chat_template == True:
                print("apply_chat_template not supported with fast generation yet")
        print("generate_output using fast generation")
        if draft_model_name == None:
            print("Draft model is required for fast generation. Using base model as draft model, but it may increase memory utilization. try to use draft model name for better performance.")
            draft_model_name = model_path
        return GPTFast.generate_output_fast(model_path, draft_model_name, sequence, max_length,trust_remote_code = trust_remote_code,gguf_file=gguf_file)
    else:
        if apply_chat_template == True:
            print("chat generate using default generation")
            return Generate.generate_chat_output(model_path, sequence, max_length,temperature = temperature,trust_remote_code = trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize) 
        print("generate_output using default generation")
        return Generate.generate_output(model_path, sequence, max_length,trust_remote_code = trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize) 

# generate new

def generate(*args, **kwargs):
    """
    Generates text output using a model path or preloaded model, supporting chat templates, streaming, and fast generation.
    
    Depending on the provided arguments, this function can generate text from a model path or a preloaded model object. It supports chat-style output, streaming results, speculative decoding (fast generation), quantization, and GGUF file integration. When using a preloaded model, either a tokenizer or processor must be provided. Raises a ValueError if required arguments are missing or invalid.
    
    Args:
        *args: Positional arguments (not used directly).
        **kwargs: Keyword arguments controlling generation behavior.
    
    Returns:
        The generated text output, or a (thread, streamer) tuple if streaming is enabled.
    
    Raises:
        ValueError: If required arguments are missing or invalid.
    """
    if 'model_path' in kwargs and ('sequence' in kwargs or 'chat' in kwargs):
        model_path = kwargs['model_path']
        sequence = kwargs['sequence']
        max_length = kwargs.get('max_length', 50)
        trust_remote_code = kwargs.get('trust_remote_code', False)
        use_fast_generation = kwargs.get('use_fast_generation', False)
        draft_model_name = kwargs.get('draft_model_name', None)
        apply_chat_template = kwargs.get('apply_chat_template',False)
        temperature= kwargs.get('temperature',0.1)
        gguf_file = kwargs.get('gguf_file',None)
        auto_quantize = kwargs.get('auto_quantize',None)
        stream = kwargs.get('stream',False)
        if use_fast_generation:
            if apply_chat_template == True:
                print("apply_chat_template not supported with fast generation yet")
            if auto_quantize != None:
                print("auto quantize is not supported with fast generation yet")
            print("generate_output using fast generation")
            if draft_model_name == None:
                print("Draft model is required for fast generation. Using base model as draft model, but it may increase memory utilization. try to use draft model name for better performance.")
                draft_model_name = model_path
            return GPTFast.generate_output_fast(model_path, draft_model_name, sequence, max_length,trust_remote_code = trust_remote_code,gguf_file = gguf_file)
        else:
            if apply_chat_template == True:
                if stream:
                    return Generate.generate_chat_output_stream(model_path, sequence, max_length,temperature = temperature,trust_remote_code = trust_remote_code, gguf_file = gguf_file,auto_quantize=auto_quantize)
                return Generate.generate_chat_output(model_path, sequence, max_length,temperature = temperature,trust_remote_code = trust_remote_code, gguf_file = gguf_file,auto_quantize=auto_quantize) 
            print("generate_output using default generation")
            if stream:
                return Generate.generate_output_stream(model_path, sequence, max_length,trust_remote_code = trust_remote_code,gguf_file = gguf_file,auto_quantize=auto_quantize)
            return Generate.generate_output(model_path, sequence, max_length,trust_remote_code = trust_remote_code,gguf_file = gguf_file,auto_quantize=auto_quantize) 
    
    elif 'model' in kwargs and 'sequence' in kwargs:
        model = kwargs['model']
        sequence = kwargs['sequence']
        max_length = kwargs.get('max_length', 50)
        trust_remote_code = kwargs.get('trust_remote_code', False)
        apply_chat_template = kwargs.get('apply_chat_template',False)
        temperature= kwargs.get('temperature',0.1)
        stream = kwargs.get('stream',False)
        processor = kwargs.get('processor', None)
        tokenizer = kwargs.get('tokenizer', None)
        if processor is not None:
            if apply_chat_template:
                if stream:
                    return Generate.generate_output_with_processor_stream(model, processor, sequence, max_length, temperature=temperature)
                return Generate.generate_output_with_processor(model, processor, sequence, max_length, temperature=temperature)
            else:
                # If not chat, fallback to tokenizer if available
                if tokenizer is not None:
                    if stream:
                        return Generate.generate_output_from_model_stream(model, tokenizer, sequence, max_length,trust_remote_code=trust_remote_code)
                    return Generate.generate_output_from_model(model, tokenizer, sequence, max_length,trust_remote_code=trust_remote_code)
                else:
                    raise ValueError("Processor-based generation requires apply_chat_template=True or a tokenizer.")
        else:
            if tokenizer is not None:
                if apply_chat_template:
                    if stream:
                        return Generate.generate_chat_output_from_model_stream(model, tokenizer, sequence, max_length,temperature = temperature,trust_remote_code=trust_remote_code)
                    return Generate.generate_chat_output_from_model(model, tokenizer, sequence, max_length,temperature = temperature,trust_remote_code=trust_remote_code)
                try:
                    if type(model) == torch._dynamo.eval_frame.OptimizedModule:
                        print("generate_output using fast generation")
                        return GPTFast.generate_output_from_model(model, tokenizer, sequence, max_length)
                    else:
                        if stream:
                            return Generate.generate_output_from_model_stream(model, tokenizer, sequence, max_length,trust_remote_code=trust_remote_code)
                        return Generate.generate_output_from_model(model, tokenizer, sequence, max_length,trust_remote_code=trust_remote_code)
                except Exception as e:
                    print("Exception occurred while loading torch._dynamo.eval_frame.OptimizedModule")
                    print("generate_output using default generation")
                    if stream:
                        return Generate.generate_output_from_model_stream(model, tokenizer, sequence, max_length,trust_remote_code=trust_remote_code)
                    return Generate.generate_output_from_model(model, tokenizer, sequence, max_length,trust_remote_code=trust_remote_code)
            else:
                raise ValueError("Must provide either a processor or tokenizer with model.")
    else:
        raise ValueError("Invalid arguments")
    
def text_generate(*args, **kwargs):
    """
    Generates text using the provided model, tokenizer, or processor, and prints the decoded output if possible.
    
    Supports both standard and streaming generation modes. If streaming is enabled, returns the streaming thread and streamer objects; otherwise, prints and returns the decoded text when possible, or the raw output.
    """
    stream = kwargs.get('stream',False)
    if stream:
        thread, streamer = generate(*args, **kwargs)
        return thread, streamer
    result = generate(*args, **kwargs)
    # result can be (output, tokenizer) or (inputs, output, tokenizer) or (inputs, output, processor)
    if isinstance(result, tuple):
        if len(result) == 2:
            output, tok = result
            if hasattr(tok, 'decode'):
                print(tok.decode(output[0], skip_special_tokens=True))
                return tok.decode(output[0], skip_special_tokens=True)
            else:
                return output
        elif len(result) == 3:
            inputs, output, tok = result
            if hasattr(tok, 'decode'):
                print(tok.decode(output[0][inputs['input_ids'].size(1):], skip_special_tokens=True))
                return tok.decode(output[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
            else:
                return output
    return result
def _handle_stream(thread, streamer):
    """
    Starts a streaming thread and prints streamed text output incrementally.
    
    Args:
        thread: The thread responsible for generating streamed output.
        streamer: An iterable yielding chunks of text as they are generated.
    
    Returns:
        The full concatenated output text after streaming is complete.
    """
    thread.start()
    try:
        out = ""
        for new_text in streamer:
            out = out + new_text
            print(new_text, end="")
        return out
    finally:
        thread.join()
def text_stream(*args, **kwargs):
    """
    Streams generated text output using the provided arguments.
    
    Enables streaming mode for text generation and returns either the streaming objects or prints the streamed output incrementally, depending on the `return_streamer` flag.
    
    Args:
        *args: Positional arguments forwarded to the generation function.
        **kwargs: Keyword arguments for generation configuration.
    
    Returns:
        If `return_streamer` is True, returns the streaming thread and streamer objects.
        Otherwise, prints the streamed output and returns the final text.
    """
    kwargs['stream'] = True
    
    thread, streamer = generate(*args, **kwargs)
    return_streamer = kwargs.get('return_streamer',False)
    if return_streamer:
        return thread, streamer
    return _handle_stream(thread, streamer)
    

def chat_generate(*args, **kwargs):
    """
    Generates chat-style output using the specified model and input sequence.
    
    If streaming is enabled, returns the streaming thread and streamer objects. Otherwise, returns the generated chat output as a string, decoding it if possible. The function automatically applies a chat template to the input.
    
    Args:
        *args: Positional arguments forwarded to the underlying generation function.
        **kwargs: Keyword arguments for generation configuration.
    
    Returns:
        The generated chat output as a string, or a tuple of (thread, streamer) if streaming is enabled.
    """
    kwargs['apply_chat_template'] = True
    stream = kwargs.get('stream',False)
    if stream:
        thread, streamer = generate(*args, **kwargs)
        return thread, streamer
    dont_print_output = kwargs.get('dont_print_output', False)
    result = generate(*args, **kwargs)
    if isinstance(result, tuple) and len(result) == 3:
        inputs, output, tok = result
        if hasattr(tok, 'decode'):
            if not dont_print_output:
                print(tok.decode(output[0][inputs['input_ids'].size(1):], skip_special_tokens=True))
            return tok.decode(output[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
        else:
            return output
    return result
def chat_stream(*args, **kwargs):
    """
    Streams chat-style text generation using the provided arguments.
    
    Enables both streaming and chat template application, returning either the streaming objects or printing the streamed output incrementally.
    
    Returns:
        If `return_streamer` is True, returns the streaming thread and streamer objects.
        Otherwise, prints the streamed output and returns the final result.
    """
    kwargs['stream'] = True
    kwargs['apply_chat_template'] = True
    thread, streamer = generate(*args, **kwargs)
    return_streamer = kwargs.get('return_streamer',False)
    if return_streamer:
        return thread, streamer
    
    return _handle_stream(thread, streamer)
# Data preparation
def generate_text_data_source_openai(client,gpt_model,prompt,number_of_examples,temperature =0.5):
    """
    Generates a text data source using OpenAI's GPT model.

    Parameters:
    - client: The OpenAI client object.
    - gpt_model: The GPT model to use for generating text.
    - prompt: The prompt to start the text generation.
    - number_of_examples: The number of text examples to generate.
    - temperature: The temperature parameter for controlling the randomness of the generated text. Default is 0.5.

    Returns:
    - The generated text data source.
    """
    return DataPreparation.generate_text_data_source_openai(client,gpt_model,prompt,number_of_examples,temperature)

def generate_text_data_source_llm(model_path, sequence, max_length,number_of_examples,trust_remote_code=False):
    """
    Generate a text data source for language model training.

    Args:
        model_path (str): The path to the language model.
        sequence (str): The input sequence to generate data from.
        max_length (int): The maximum length of the generated data.
        number_of_examples (int): The number of examples to generate.
        trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.

    Returns:
        str: The generated text data source.
    """
    return DataPreparation.generate_text_data_source_llm(model_path, sequence, max_length,number_of_examples,trust_remote_code=trust_remote_code)

# Get model and tokenizer
def load_model(model_path,trust_remote_code=False,use_fast_generation=False, draft_model_name=None,gguf_file=None,auto_quantize=None):
    """
    Load a model from the specified model_path.

    Args:
        model_path (str): The path to the model.
        trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
        use_fast_generation (bool, optional): Whether to use fast generation. Defaults to False.
        draft_model_name (str, optional): The name of the draft model. Defaults to None.
        gguf_file (str, optional): The path to the GGUF file. Defaults to None.
        auto_quantize (str, optional): Automatically apply quantization. Accepts "4bit"/"high" for high compression or "8bit"/"low" for lower compression. Defaults to None.

    Returns:
        Model: The loaded model.

    Raises:
        FileNotFoundError: If the model_path does not exist.
    """
    if use_fast_generation:
        print("load_model using fast generation")
        if draft_model_name == None:
            print("Draft model is required for fast generation. Using base model as draft model, but it may increase memory utilization. try to use draft model name for better performance.")
            draft_model_name = model_path
        return GPTFast.load_model(model_path, draft_model_name,trust_remote_code=trust_remote_code,gguf_file=gguf_file)
    else:
        print("load_model using default generation")
        return Generate.load_model(model_path,trust_remote_code=trust_remote_code,gguf_file=gguf_file,auto_quantize=auto_quantize)

def load_tokenizer(tokenizer_path,gguf_file=None):
    """
    Loads a tokenizer from the given path.
    
    Args:
        tokenizer_path: Path to the tokenizer file.
        gguf_file: Optional GGUF file for tokenizer configuration.
    
    Returns:
        The loaded tokenizer object.
    """
    return Generate.load_tokenizer(tokenizer_path,gguf_file=gguf_file)
 
def load_processor(processor_path,use_fast=False,gguf_file=None):
    """
    Loads a processor from a given path with optional fast processing and GGUF file support.
    
    Args:
        processor_path: Path to the processor file.
        use_fast: If True, enables fast processing mode.
        gguf_file: Optional path to a GGUF file for processor configuration.
    
    Returns:
        The loaded processor object.
    """
    return Generate.load_processor(processor_path,use_fast=use_fast,gguf_file=gguf_file)


