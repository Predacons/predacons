import openai
import pandas as pd


def generate_data(prompt, existing_data, temperature=.5):
    client = openai.AzureOpenAI(
        api_version=gpt_api_version,
        azure_endpoint=gpt_azure_endpoint,    
    )
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

def generate_data_source(prompt, number_of_examples = 10,temperature=.5):
    # prompt = "A model that takes in a puzzle-like reasoning-heavy question in English, and responds with a well-reasoned, step-by-step thought out response in hindi."
    existing_data = []
    for i in range(number_of_examples):
        print(f'Generating example {i}')
        example = generate_data_source(prompt, existing_data, temperature)
        existing_data.append(example)

    return existing_data

def generate_system_message(prompt):
    completion = client.chat.completions.create(
        model=gpt_model, 
        messages=[
                {
                  "role": "system",
                  "content": "You will be given a high-level description of the model we are training, and from that, you will generate a simple system prompt for that model to use. Remember, you are not generating the system message for data generation -- you are generating the system message to use for inference. A good format to follow is `Given $INPUT_DATA, you will $WHAT_THE_MODEL_SHOULD_DO.`.\n\nMake it as concise as possible. Include nothing but the system prompt in your response.\n\nFor example, never write: `\"$SYSTEM_PROMPT_HERE\"`.\n\nIt should be like: `$SYSTEM_PROMPT_HERE`."
                },
                {
                    "role": "user",
                    "content": prompt.strip(),
                }
              ]    )
    return(completion.choices[0].message.content)


prompts = []
responses = []

for example in prev_examples:
  try:
    split_example = example.split('-----------')
    prompts.append(split_example[1].strip())
    responses.append(split_example[3].strip())
  except:
    pass

df = pd.DataFrame({
    'prompt': prompts,
    'response': responses
})

df = df.drop_duplicates()

print('There are ' + str(len(df)) + ' successfully-generated examples. Here are the first few:')

df.head()