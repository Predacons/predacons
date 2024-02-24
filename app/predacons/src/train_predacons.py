from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import AutoTokenizer,AutoModelForPreTraining,AutoModelForCausalLM
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
    
    def __trainer(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps,trust_remote_code = False):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_dataset = TrainPredacons.__load_dataset(train_file_path, tokenizer)
        data_collator = TrainPredacons.__load_data_collator(tokenizer)

        tokenizer.save_pretrained(output_dir)
        model = None
        try:
            model = AutoModelForPreTraining.from_pretrained(model_name,trust_remote_code=trust_remote_code)
        except:
            model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=trust_remote_code)

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
        return trainer

    def __train(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps,trust_remote_code = False,resume_from_checkpoint=True):

        trainer = TrainPredacons.__trainer(train_file_path,model_name,
            output_dir,
            overwrite_output_dir,
            per_device_train_batch_size,
            num_train_epochs,
            save_steps,trust_remote_code = trust_remote_code)
        try:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        except:
            print('Failed to resume from checkpoint. training from scratch.')
            trainer.train()
        trainer.save_model()

    def trainer(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps,trust_remote_code = False):
        return TrainPredacons.__trainer(train_file_path,model_name,
            output_dir,
            overwrite_output_dir,
            per_device_train_batch_size,
            num_train_epochs,
            save_steps,trust_remote_code = trust_remote_code)
    
    def train(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps,trust_remote_code = False,resume_from_checkpoint = True):
        TrainPredacons.__train(train_file_path,model_name,output_dir,overwrite_output_dir,per_device_train_batch_size,num_train_epochs,save_steps,trust_remote_code = trust_remote_code,resume_from_checkpoint=resume_from_checkpoint)