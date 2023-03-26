from datasets import load_dataset
import json 
from flan.v2.templates import PATTERNS
import random
from tqdm import tqdm

DIR = "/home/ubuntu/LKLab-storage-texas/sejune"

def save_data(data,name,idx):
    # for idx in range(10):
    print('start')
    result = []
    idx_cnt =0 
    for _data in data:
        if "options" in _data.keys():
            e_formatted_d = {
            "config" : "none",
            "task" : name,
            "prompt" : str(idx),
            "labels_list" : [i.numpy().decode("utf-8") for i in _data['options']],
            "source" : _data['inputs_pretokenized'].numpy().decode("utf-8"),
            "target" : _data['targets_pretokenized'].numpy().decode("utf-8"),
            }
        else:
            e_formatted_d = {
            "config" : "none",
            "task" : name,
            "prompt" : str(idx),
            "source" : _data['inputs_pretokenized'].numpy().decode("utf-8"),
            "target" : _data['targets_pretokenized'].numpy().decode("utf-8"),
            }

        result.append(e_formatted_d)
        idx_cnt+=1
        if idx_cnt > 10000:
            break
        
    with open(f"{DIR}/dump/{idx}_{name}.json","w") as f:
        json.dump(result,f,indent=4)

def filter_empty(data):
    non_empty = []
    for _data in data:
        if _data['target'] == "":
            continue
        else:
            _data['target'] = _data['target'].replace("\n\n\n","")
            non_empty.append(_data)
    return non_empty

def random_filter(data,size):
    filtered_data = filter_empty(data)
    if len(filtered_data)>size:
        sampled_filtered_data = random.sample(filtered_data,size)
    else:
        sampled_filtered_data = filtered_data
    
    return {str(idx):i for idx,i in enumerate(sampled_filtered_data)}

def parse_data(data,size):
    dat = random_filter(data,size)
    return dat
