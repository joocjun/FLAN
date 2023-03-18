import json 
import os
from datasets import load_dataset
from flan.v2.templates import PATTERNS

DATA_DIR = "/home/ubuntu/LKLab-storage-texas/sejune"
# DATA_DIR = "."


def save_aeslc():
    data = load_dataset("aeslc",split="train")
    for idx in range(10):
        result = []
        temp = PATTERNS["aeslc"][idx]
        source = temp[0]
        target = temp[1]
        for _data in data:
            if "body" in source:
                e_formatted_d = {
                "config" : "none",
                "task" : "aeslc",
                "prompt" : str(idx),
                "source" : source.format(body=_data['email_body']),
                "target" : target.format(subject= _data['subject_line']),
                }
            else:
                e_formatted_d = {
                "config" : "none",
                "task" : "aeslc",
                "prompt" : str(idx),
                "source" : source.format(subject= _data['subject_line']),
                "target" : target.format(body=_data['email_body']),
                }
            result.append(e_formatted_d)
        with open(f"{DATA_DIR}/dump/{idx}_aeslc.json","w") as f:
            json.dump(result,f,indent=4)


def save_windogrande():
    data = load_dataset("winogrande","winogrande_xl",split="train")
    for idx in range(10):
        result = []
        temp = PATTERNS["winogrande"][idx]
        
        source = temp[0]
        target = temp[1]
        for _data in data:
            key = _data['answer']
            options = f"[Options]\n-{_data['option1']}\n-{_data['option2']}"
            e_formatted_d = {
            "config" : "none",
            "task" : "winogrande",
            "prompt" : str(idx),
            "labels_list" : [_data['option1'],_data['option2']],
            "source" : source.format(context=_data['sentence'],options_=options),
            "target" : target.format(answer= _data[f'option{key}']),
            }
            result.append(e_formatted_d)
        with open(f"{DATA_DIR}/dump/{idx}_winogrande.json","w") as f:
            json.dump(result,f,indent=4)

def save_definite():
    data_name = "definite_pronoun_resolution"
    data = load_dataset("definite_pronoun_resolution",split="train")

    for idx in range(10):
        result = []
        temp = PATTERNS[data_name][idx]
        source = temp[0]
        target = temp[1]
        for _data in data:
            options = f"[Options]\n-{_data['candidates'][0]}\n-{_data['candidates'][1]}"
            e_formatted_d = {
            "config" : "none",
            "task" : data_name,
            "prompt" : str(idx),
            "labels_list" : _data['candidates'],
            "source" : source.format(sentence=_data['sentence'],pronoun=_data['pronoun'],options_=options),
            "target" : target.format(answer= _data['candidates'][_data['label']]),
            }
            result.append(e_formatted_d)
        with open(f"{DATA_DIR}/dump/{idx}_{data_name}.json","w") as f:
            json.dump(result,f,indent=4)

save_definite()
save_aeslc()
save_windogrande()