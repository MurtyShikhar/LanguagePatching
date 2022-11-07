from datasets import load_dataset

# setting path
from checklist.editor import Editor
from collections import defaultdict as ddict
import random
import pandas as pd

editor = Editor()

import re

### Data slice from FewRel
def get_fewrel_data():
    neg_relations = [
        "P3373",
        "P22",
        "P6",
        "P57",
        "P674",
        "P1344",
        "P22",
        "P991",
        "P106",
        "P463",
        "P40",
        "P25",
        "P108",
    ]
    pos_relations = ["P26"]
    dataset = load_dataset("few_rel", "default")
    keys = ["train_wiki", "val_wiki", "val_semeval"]

    def get_all(data, relation_types):
        filtered = []
        for ex in data:
            if ex["relation"] in relation_types:
                filtered.append(ex)
        return filtered

    def process(ex):
        if ex[0][-1] != ".":
            ex[0] = "{}.".format(ex[0])
        return "{} Entity1: {}. Entity2: {}".format(ex[0], ex[1], ex[2])

    positives = []
    negatives = []
    for key in keys:
        negatives += get_all(dataset[key], neg_relations)
    for key in keys:
        positives += get_all(dataset[key], pos_relations)

    positive_data = [
        process([" ".join(l["tokens"]), l["head"]["text"], l["tail"]["text"]])
        for l in positives
    ]
    negative_data = [
        process([" ".join(l["tokens"]), l["head"]["text"], l["tail"]["text"]])
        for l in negatives
    ]
    return positive_data, negative_data
