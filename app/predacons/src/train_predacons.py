from transformers import TextDataset, DataCollatorForLanguageModeling,BitsAndBytesConfig
from transformers import AutoTokenizer,AutoModelForPreTraining,AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from peft import LoraConfig
from datasets import load_dataset
import torch
from trl import SFTTrainer

class TrainPredacons:
    def get_feature_list(train_dataset):
        feature_list  = []
        for feature in  list(train_dataset['train'].features):
            if feature not in ['input_ids', 'attention_mask', 'Unnamed: 0']:
                    feature_list.append(feature)
        print(feature_list)
        return feature_list
    
    def create_formatting_func(train_dataset):
        feature_list  = TrainPredacons.get_feature_list(train_dataset)
        def auto_formatting_func(example):
            func_str =''
            for feature in feature_list:
                feature_str = f"{feature}: {example[feature][0]}\n"
                func_str = func_str + feature_str
            text = f"{func_str}"
            return [text]

        return auto_formatting_func

    def formatting_func(example):
        text = f"{example['text'][0]}"
        return [text]

    def __load_dataset(file_path, tokenizer, block_size = 128):
        dataset = TextDataset(
            tokenizer = tokenizer,
            file_path = file_path,
            block_size = block_size,
        )
        return dataset
    def lld(file_path, tokenizer, block_size = 128):
        dataset = TextDataset(
            tokenizer = tokenizer,
            file_path = file_path,
            block_size = block_size,
        )
        return dataset
    def __load_data_collator(tokenizer, mlm = False):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=mlm,
        )
        return data_collator


    def __trainer(*args,**kwargs):

        # use_legacy_trainer
        print("Initiating predacons training...")
        use_legacy_trainer = kwargs.get('use_legacy_trainer', False)
        if(use_legacy_trainer):
            print("Using lagacy trainer will be removed in further release.")
            tokenizer = AutoTokenizer.from_pretrained(kwargs.get('model_name', None))
            train_dataset = TrainPredacons.__load_dataset(kwargs.get('train_file_path', None), kwargs.get('tokenizer', None))
            data_collator = TrainPredacons.__load_data_collator(kwargs.get('tokenizer', None))

            tokenizer.save_pretrained(kwargs.get('output_dir', None))
            model = None
            try:
                model = AutoModelForPreTraining.from_pretrained(kwargs.get('model_name', None),trust_remote_code=kwargs.get('trust_remote_code', None))
            except:
                model = AutoModelForCausalLM.from_pretrained(kwargs.get('model_name', None),trust_remote_code=kwargs.get('trust_remote_code', None))

            model.save_pretrained(kwargs.get('output_dir', None))

            training_args = TrainingArguments(
                    output_dir=kwargs.get('output_dir', None),
                    overwrite_output_dir=kwargs.get('overwrite_output_dir', None),
                    per_device_train_batch_size=kwargs.get('per_device_train_batch_size', None),
                    num_train_epochs=kwargs.get('num_train_epochs', None),
                )

            trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=train_dataset,
            )
            return trainer

        # load model and tokenizer
        print("Loading model and tokenizer...")
        model_name = ""
        model = None
        tokenizer = None
        if 'model_name' in kwargs:
            model_name = kwargs['model_name']
            quantization_config = None
            if 'quantization_config' in kwargs:
                quantization_config = kwargs['quantization_config']
                print("Using user provided quantization_config...")
            auto_quantize = kwargs.get('auto_quantize', "")

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
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            trust_remote_code = kwargs.get('trust_remote_code', False)

            model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config = quantization_config,trust_remote_code =trust_remote_code)
            print(f"Model ready: {model_name} m")

        elif ('model' in kwargs and 'tokenizer' in kwargs):
            print("Using provided model and tokenizer...")
            try:
                model = kwargs['model']
                tokenizer = kwargs['tokenizer']
            except Exception as e:
                raise ValueError(f"error while loading model and tokenizer {e}")
        else:
            raise Exception("provide model name or model and tokenizer")

        # get output dir
        output_dir =""
        if 'output_dir' in kwargs:
            output_dir =  kwargs['output_dir']
            tokenizer.save_pretrained(output_dir)
            model.save_pretrained(output_dir)
        else:
            print("Warning: Output directory not provided saving everything to root dir, better stop execution and provide value in output_dir... " )

        # load peft_config
        peft_config = None
        auto_lora_config = kwargs.get('auto_lora_config',False)
        if auto_lora_config == True:
            peft_config = LoraConfig(
                r = 8,
                target_modules = ["q_proj", "o_proj", "k_proj", "v_proj",
                                "gate_proj", "up_proj", "down_proj"],
                task_type = "CAUSAL_LM",
            )
        else:
            peft_config = kwargs.get('peft_config',None)
        print(f"peft_config: {peft_config}")

        # load training_args
        print("Loading training_args...")
        training_args = kwargs.get('training_args',None)
        if training_args == None:
            per_device_train_batch_size = kwargs.get('per_device_train_batch_size',None)
            num_train_epochs =  kwargs.get('num_train_epochs',None)
            training_args = TrainingArguments(
                    output_dir = output_dir,
                    overwrite_output_dir = kwargs.get('overwrite_output_dir',None),
                    per_device_train_batch_size = kwargs.get('per_device_train_batch_size',1),
                    num_train_epochs=num_train_epochs,
                    fp16=True if auto_lora_config == True else True ,
                    optim="paged_adamw_8bit" if auto_lora_config == True else "paged_adamw_8bit",
                    save_steps = kwargs.get('save_steps',500)
                )

        # dataset configuration
        print(f"Creating data set...")
        train_file_type = kwargs.get('train_file_type',None)
        train_file_path = kwargs.get('train_file_path',None)
        train_dataset = kwargs.get('train_dataset',None)
        preprcess_function = kwargs.get('preprcess_function',None)
        if train_file_path != None and train_dataset == None:
            if train_file_type not in ("text","csv","json"):
                print("Error: File type not supported use text,csv or json trying to process as text")
                train_file_type = "text"
            train_dataset = load_dataset(train_file_type, data_files={"train": train_file_path})
            if train_file_type == "text":
                train_dataset = train_dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)
            else:
                if preprcess_function == None:
                    print("No preprcess_function provided using default preprcess_function")
                    first_feature = TrainPredacons.get_feature_list(train_dataset)[0]
                    print(f"Using first_feature: {first_feature}, pass preprcess_function to use a custom preprocess_function")
                    train_dataset = train_dataset.map(lambda samples: tokenizer(samples[first_feature]), batched=True)
                else:
                    train_dataset = train_dataset.map(preprcess_function, batched=True)
        else:
            print("Please provide train_file_path to load dataset other sources will be added in next release")
        print(f"train_dataset: {train_dataset}")
        try:
            print("Using custom formatting function...")
            formatting_func = TrainPredacons.create_formatting_func(train_dataset)
        except:
            print("Using default formatting function...")
            formatting_func = TrainPredacons.formatting_func
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset["train"],
            args=training_args,
            peft_config=peft_config,
            formatting_func = formatting_func

        )
        return trainer

    def __train(*args,**kwargs):

        trainer = TrainPredacons.__trainer(*args,**kwargs)
        try:
            checkpoint_folder_exists = any(folder.startswith('checkpoint') for folder in os.listdir(kwargs.get('output_dir', "")))
            trainer.train(resume_from_checkpoint = (checkpoint_folder_exists and kwargs.get('resume_from_checkpoint', "")))
        except:
            print('Error: Failed to resume from checkpoint. training from scratch.')
            trainer.train()
        trainer.save_model()

    def trainer(*args,**kwargs):
        return TrainPredacons.__trainer(*args,**kwargs)

    def train(*args,**kwargs):
        return TrainPredacons.__train(*args,**kwargs)