import json
from utils import *
import json
from dataloader import *
from tqdm import tqdm
size = 600

DIR = "/home/ubuntu/LKLab-storage-texas/sejune"

for category,tasks in tqdm(CODE_DATA.items()):
    data_dict = {}
    for task in tasks:
        task_dict = {}
        for i in range(10):
            with open(f"{DIR}/dump/{i}_{task}.json","r") as f:
                data = json.load(f)
            task_dict[str(i)] = parse_data(data,size=size)
        data_dict[task] = task_dict
    with open(f"{DIR}/flan/{category}_train_phase2_{size}.json","w") as f:
        json.dump(data_dict,f,indent=4)
