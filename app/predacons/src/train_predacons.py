from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import AutoTokenizer,AutoModelForPreTraining
from transformers import Trainer, TrainingArguments

class TrainPredacons:

    def __load_dataset(file_path, tokenizer, block_size = 128):
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
    
    def __train(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):
  
  
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_dataset = TrainPredacons.__load_dataset(train_file_path, tokenizer)
        data_collator = TrainPredacons.__load_data_collator(tokenizer)

        tokenizer.save_pretrained(output_dir)
            
        model = AutoModelForPreTraining.from_pretrained(model_name)

        model.save_pretrained(output_dir)

        training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=overwrite_output_dir,
                per_device_train_batch_size=per_device_train_batch_size,
                num_train_epochs=num_train_epochs,
            )

        trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
        )
            
        trainer.train()
        trainer.save_model()

    def train(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):
        TrainPredacons.__train(train_file_path,model_name,output_dir,overwrite_output_dir,per_device_train_batch_size,num_train_epochs,save_steps)