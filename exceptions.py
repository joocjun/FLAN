import json 
import os
# from datasets import load_dataset
from flan.v2.templates import PATTERNS
import pandas as pd
import tensorflow as tf
import random
DATA_DIR = "/home/ubuntu/LKLab-storage-texas/sejune"
# DATA_DIR = "."
import random

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

# save_definite()
# save_aeslc()
# save_windogrande()



def task_master(): 
    data_name = "task_master"
    with open("/home/sejune/Taskmaster/TM-1-2019/self-dialogs.json","r") as f:
        data = json.load(f)

    for idx in range(10):
        result = []
        temp = PATTERNS[data_name][idx]
        source = temp[0]
        target = temp[1]
        for _data in data:
            # options = f"[Options]\n-{_data['candidates'][0]}\n-{_data['candidates'][1]}"
            utt = _data['utterances']
            utt_idx = random.randint(1,min(len(utt)-1,20))
            e_formatted_d = {
            "config" : "none",
            "task" : data_name,
            "prompt" : str(idx),
            "source" : source.format(dialog_="\n".join([utt[i]['text'] for i in range(utt_idx)])),
            "target" : target.format(answer= utt[utt_idx]['text']),
            }
            result.append(e_formatted_d)
        with open(f"{DATA_DIR}/dump/{idx}_{data_name}.json","w") as f:
            json.dump(result,f,indent=4)


def task_master_inversion(): 
    data_name = "task_master_input_inversion"
    with open("/home/sejune/Taskmaster/TM-1-2019/self-dialogs.json","r") as f:
        data = json.load(f)

    for idx in range(10):
        result = []
        temp = PATTERNS[data_name][idx]
        source = temp[0]
        target = temp[1]
        for _data in data:
            # options = f"[Options]\n-{_data['candidates'][0]}\n-{_data['candidates'][1]}"
            utt = _data['utterances']
            utt_idx = random.randint(1,min(len(utt)-1,20))
            e_formatted_d = {
            "config" : "none",
            "task" : data_name,
            "prompt" : str(idx),
            "source" : source.format(answer= utt[utt_idx]['text']),
            "target" : target.format(dialog_="\n".join([utt[i]['text'] for i in range(utt_idx)])),
            }
            result.append(e_formatted_d)
        with open(f"{DATA_DIR}/dump/{idx}_{data_name}.json","w") as f:
            json.dump(result,f,indent=4)


# task_master()
# task_master_inversion()


def dr_repair():
    data_name = "program_synthesis_dr_repair"
    # data = load_dataset("dr_repair",split="train")
    with open("/home/ubuntu/LKLab-storage-texas/sejune/code_data/dr_repair/train.json","r") as f:
        data = json.load(f)
    data = data[:1000]
    for idx in range(10):
        result = []
        temp = PATTERNS[data_name][idx]
        source = temp[0]
        target = temp[1]
        for _data in data:
            e_formatted_d = {
            "config" : "none",
            "task" : data_name,
            "prompt" : str(idx),
            "source" : source.format(question=_data['mod_code']),
            "target" : target.format(answer=_data['code']),
            }
            result.append(e_formatted_d)
        with open(f"{DATA_DIR}/dump/{idx}_{data_name}.json","w") as f:
            json.dump(result,f,indent=4)

def dr_repair_comment():
    data_name = "program_synthesis_dr_repair_error_comments"
    # data = load_dataset("dr_repair",split="train")
    with open("/home/ubuntu/LKLab-storage-texas/sejune/code_data/dr_repair/train.json","r") as f:
        data = json.load(f)
    data = data[:1000]
    for idx in range(10):
        result = []
        temp = PATTERNS[data_name][idx]
        source = temp[0]
        target = temp[1]
        for _data in data:
            e_formatted_d = {
            "config" : "none",
            "task" : data_name,
            "prompt" : str(idx),
            "source" : source.format(question=_data['err_msg']),
            "target" : target.format(answer=_data['code']),
            }
            result.append(e_formatted_d)
        with open(f"{DATA_DIR}/dump/{idx}_{data_name}.json","w") as f:
            json.dump(result,f,indent=4)

def dr_repair_inver():
    data_name = "program_synthesis_dr_repair_input_inversion"
    with open("/home/ubuntu/LKLab-storage-texas/sejune/code_data/dr_repair/train.json","r") as f:
        data = json.load(f)
    data = data[:1000]
    for idx in range(10):
        result = []
        temp = PATTERNS[data_name][idx]
        source = temp[0]
        target = temp[1]
        for _data in data:
            e_formatted_d = {
            "config" : "none",
            "task" : data_name,
            "prompt" : str(idx),
            "source" : source.format(answer=_data['code']),
            "target" : target.format(question=_data['mod_code']),
            }
            result.append(e_formatted_d)
        with open(f"{DATA_DIR}/dump/{idx}_{data_name}.json","w") as f:
            json.dump(result,f,indent=4)


def dmcc():
    data_name = "program_synthesis_dmcc_python"
    with open("/home/ubuntu/LKLab-storage-texas/sejune/code_data/dmcc/train.json","r") as f:
        data = json.load(f)
    data = random.sample(data,1000)
    for idx in range(10):
        result = []
        temp = PATTERNS[data_name][idx]
        source = temp[0]
        target = temp[1]
        for _data in data:
            e_formatted_d = {
            "config" : "none",
            "task" : data_name,
            "prompt" : str(idx),
            "source" : source.format(question=_data['question']),
            "target" : target.format(answer=_data['answer']),
            }
            result.append(e_formatted_d)
        with open(f"{DATA_DIR}/dump/{idx}_{data_name}.json","w") as f:
            json.dump(result,f,indent=4)

def dmcc_inver():
    data_name = "program_synthesis_dmcc_python_input_inversion"
    with open("/home/ubuntu/LKLab-storage-texas/sejune/code_data/dmcc/train.json","r") as f:
        data = json.load(f)
    data = random.sample(data,1000)
    for idx in range(10):
        result = []
        temp = PATTERNS[data_name][idx]
        source = temp[0]
        target = temp[1]
        for _data in data:
            e_formatted_d = {
            "config" : "none",
            "task" : data_name,
            "prompt" : str(idx),
            "source" : source.format(answer=_data['answer']),
            "target" : target.format(question=_data['question']),
            }
            result.append(e_formatted_d)
        with open(f"{DATA_DIR}/dump/{idx}_{data_name}.json","w") as f:
            json.dump(result,f,indent=4)

# dr_repair()
# print("dr_repair done")
# dr_repair_comment()
# print("dr_repair_comment done")
# dr_repair_inver()
# print("dr_repair_inver done")
dmcc()
print("dmcc done")
dmcc_inver()
print("dmcc_inver done")