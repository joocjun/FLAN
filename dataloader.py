DATA = {
    "nli":["rte","cb","anli_r1","anli_r2","anli_r3","mnli","qnli","wnli","snli"],
    "sentiment":["yelp_polarity_reviews","sentiment140","sst2"],
    "paraphrase":["stsb"],
    "cbqa":["natural_questions"],
    "rc":["drop","squad_v1"],
    "coref":["definite_pronoun_resolution","winogrande"],#
    "misc":["quac","coqa","cola","math_dataset","fix_punct"],
    "s2t":["dart","e2e_nlg","web_nlg_en"],
    "summarization":["aeslc","wiki_lingua_english_en","ag_news_subset"]
}

DATA_EXTRACT = {
    "nli":["rte","cb","anli_r1","anli_r2","anli_r3","mnli","qnli","wnli","snli"],
    "sentiment":["yelp_polarity_reviews","sentiment140","sst2"],
    "paraphrase":["stsb"],
    "cbqa":["natural_questions"],
    "rc":["drop","squad_v1"],
    # "coref":["definite_pronoun_resolution"],
    "misc":["quac","coqa","cola","math_dataset","fix_punct"],
    "s2t":["dart","e2e_nlg","web_nlg_en"],
    "summarization":["wiki_lingua_english_en","ag_news_subset"]
}

DATA_RAT = {
    "nli":["rte","cb","anli_r1","anli_r2","anli_r3","mnli","qnli","wnli","snli"],
    "cbqa":["natural_questions"],
    "rc":["drop","squad_v1"],
    "misc":["quac","cola","math_dataset"],
}

DIALOG_DATA = {"dialog":['wiki_dialog',"wiki_dialog_input_inversion",'qrecc',"qrecc_input_inversion","task_master","task_master_input_inversion"]} #, 