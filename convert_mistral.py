import json
import pandas

def text_formatting(data):

    # If the input column is not empty
    if data['input']:

        text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{data["instruction"]} \n\n### Input:\n{data["input"]}\n\n### Response:\n{data["output"]}"""

    else:

        text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{data["instruction"]}\n\n### Response:\n{data["output"]}""" 

    return text

def chat_formatting(data):
  text = f"<s>[INST] {data['instruction']} [/INST] {data['output']} </s>"
  return text

f=open("words_of_brandon.json")
data=f.read()
data_json=json.loads(data)
train = pd.DataFrame(data_json)
train['text'] = train.apply(text_formatting, axis =1)
train_chat = train[train['input'] == ''].reset_index(drop = True).copy()
train_chat['text'] = train_chat.apply(chat_formatting, axis =1)
train_chat.to_csv('train_chat.csv', index =False)
