from flan.v2 import mixtures
from flan.v2.templates import PATTERNS
import seqio
from dataloader import *
from seqio import TaskRegistry
from utils import *
IDX = 9
from tqdm import tqdm


for category, tasks in tqdm(DIALOG_DATA.items()):    
    for task in tasks:
        # try:
        if task == "mnli":
            t = f"mnli_matched_template_{IDX}_zero_shot"
        else:
            t= f"{task}_template_{IDX}_zero_shot"
        dataset = seqio.get_mixture_or_task(t).get_dataset(sequence_length={"inputs": 256, "targets": 256}) 
        save_data(dataset,task,IDX)
        # except Exception as e:
        #     print(e)
        #     print(f"Failed to load {task}")
        

