from flan.v2 import mixtures
from flan.v2.templates import PATTERNS
import seqio
from dataloader import *
from seqio import TaskRegistry


t = "wiki_dialog_template_0to10_zero_shot"
dataset = seqio.get_mixture_or_task(t).get_dataset(sequence_length={"inputs": 256, "targets": 256}) 
