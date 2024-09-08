import openai
import pandas as pd
from .generate import Generate
import random

class DataPreparation:
    def __generate_data_source_openai(client,gpt_model,prompt, existing_data, temperature=.5):
        
        messages=[
            {
                "role": "system",
                "content": f"You are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train, and from that, you will generate data samples, each with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\n$prompt_goes_here\n-----------\n\nresponse\n-----------\n$response_goes_here\n-----------\n```\n\nOnly one prompt/response pair should be generated per turn.\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.\n\nHere is the type of model we want to train:\n`{prompt}`"
            }
        ]
        if len(existing_data) > 0:
            if len(existing_data) > 10:
                existing_data = random.sample(existing_data, 10)
            for example in existing_data:
                messages.append({
                    "role": "assistant",
                    "content": example
                })

        completion = client.chat.completions.create(
            model=gpt_model,
            messages=messages,
        )
        return(completion.choices[0].message.content)
    
    def __generate_data_source_llm(model_path, sequence, max_length,existing_data,trust_remote_code = False):
        messages=f"system : you are assistant and you are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train, and from that, you will generate data samples, each with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\n$prompt_goes_here\n-----------\n\nresponse\n-----------\n$response_goes_here\n-----------\n```\n\nOnly one prompt/response pair should be generated per turn.\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.\n\nHere is the type of model we want to train:\n`{sequence}`"
            
        if len(existing_data) > 0:
            if len(existing_data) > 10:
                existing_data = random.sample(existing_data, 10)
            for example in existing_data:
                messages = messages + f" \n assistant : {example}"
        return Generate.generate_text(model_path, messages, max_length ,trust_remote_code = trust_remote_code)
        
    def __generate_text_data_source_openai(client,gpt_model,prompt,number_of_examples,temperature =0.5):
        prev_examples = []
        print(f'Generating data source for prompt: {prompt}')
        for i in range(int(number_of_examples)):
            example = DataPreparation.__generate_data_source_openai(client,gpt_model,prompt, prev_examples, temperature)
            prev_examples.append(example)

        training_data = "\n \n \n".join(prev_examples)
        return training_data
    
    def __generate_text_data_source_llm(model_path, sequence, max_length,number_of_examples,trust_remote_code=False):
        prev_examples = []
        print(f'Generating data source for prompt: {sequence}')
        for i in range(int(number_of_examples)):
            example=DataPreparation.__generate_data_source_llm(model_path, sequence, max_length,prev_examples,trust_remote_code = trust_remote_code)
            prev_examples.append(example)
        training_data = "\n \n \n".join(prev_examples)
        return training_data
    
    def generate_text_data_source_openai(client,gpt_model,prompt,number_of_examples,temperature =0.5):
        return DataPreparation.__generate_text_data_source_openai(client,gpt_model,prompt,number_of_examples,temperature)
    
    def generate_text_data_source_llm(model_path, sequence, max_length,number_of_examples,trust_remote_code=False):
        return DataPreparation.__generate_text_data_source_llm(model_path, sequence, max_length,number_of_examples,trust_remote_code=trust_remote_code)
