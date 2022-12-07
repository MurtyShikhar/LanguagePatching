from hydra.utils import get_original_cwd
import json
import os
import random

from helpers import get_spouse_data

from patch_dataset import PatchDataset, SimpleDataset
from collections import defaultdict as ddict


from datasets import load_dataset
from helpers import prompt_styles, verbalize_examples


def construct_sentiment_dataset(dataset, chosen_prompt_style):
    all_data = []
    prompt_func = prompt_styles[chosen_prompt_style]
    for ex in dataset:
        sentence = ex["sentence"]
        label = ex["label"]
        curr_input = prompt_func("", sentence)
        label = int(label > 0.5)
        all_data.append((curr_input, label))
    return all_data


### For every patch, get the set of examples corresponding to it
def build_patch2examples_dict(all_texts, all_labels, all_patches):
    all_texts_unique = list(set(all_texts))
    random.shuffle(all_texts_unique)
    train_split = int(len(all_texts_unique) * 0.8)
    train_texts = set(all_texts_unique[:train_split])

    patch2examples_dict_train = ddict(list)
    patch2examples_dict_validation = ddict(list)
    for idx, patch in enumerate(all_patches):
        corresponding_text = all_texts[idx]
        patch = patch.strip()
        if corresponding_text in train_texts:
            patch2examples_dict_train[patch].append(idx)
        else:
            patch2examples_dict_validation[patch].append(idx)
    return patch2examples_dict_train, patch2examples_dict_validation


def get_patch_data_helper(
    cfg, orig_dir, patch_json, tokenizer, use_negatives=True, get_hard_negs=False
):
    print("Reading patches from {}".format(patch_json))
    with open(os.path.join(orig_dir, patch_json)) as reader:
        data = json.load(reader)
    all_texts, all_labels, all_patches = (
        data["instances"],
        data["labels"],
        data["explanations"],
    )
    (
        patch2examples_dict_train,
        patch2examples_dict_validation,
    ) = build_patch2examples_dict(all_texts, all_labels, all_patches)

    kwargs = {
        "use_negatives": use_negatives,
        "get_hard_negs": get_hard_negs,
    }
    train_dataset = PatchDataset(
        patch2examples_dict_train, all_texts, all_labels, tokenizer, **kwargs
    )
    val_dataset = PatchDataset(
        patch2examples_dict_validation, all_texts, all_labels, tokenizer, **kwargs
    )
    return train_dataset, val_dataset


def get_data_patch_finetuning_re(cfg, tokenizer):
    try:
        orig_dir = get_original_cwd()
    except:
        orig_dir = ""
    patch_dir = cfg.get("patch_type", "patch_re")

    if cfg.get("learnt_interpreter", False):
        patch_json = "PATCH_DIR/{}/synthetic_data.json".format(patch_dir)
        train_dataset, val_dataset = get_patch_data_helper(
            cfg, orig_dir, patch_json, tokenizer, use_negatives=False
        )
        all_datasets = {"task_data_patch": train_dataset}
    else:
        patch_json = "PATCH_DIR/{}/synthetic_data_applies.json".format(patch_dir)
        train_dataset, val_dataset = get_patch_data_helper(
            cfg, orig_dir, patch_json, tokenizer, get_hard_negs=False
        )
        all_datasets = {"patch_grounding_data": train_dataset}

    if cfg.get("multitask_re", False):
        print("Using spouse data!")
        use_percent = cfg.get("use_percent", 0.1)
        verbalized_spouse_train = get_spouse_data(
            "train", prompt_style="p1", use_percent=use_percent
        )
        re_train_dataset = SimpleDataset(
            verbalized_spouse_train,
            tokenizer,
            as_lm="gpt" in cfg.model_type or "t5" in cfg.model_type,
        )
        all_datasets["task_data"] = re_train_dataset

    return all_datasets, val_dataset


def get_data_patch_finetuning_sentiment(cfg, tokenizer):
    try:
        orig_dir = get_original_cwd()
    except:
        orig_dir = ""
    patch_type = cfg.get("patch_type", "patch1")
    # this is for training p(patch | x) => T5[prompt(patch, x)] => gating head
    patch_json = "PATCH_DIR/{}/synthetic_data_applies.json".format(patch_type)
    train_dataset, val_dataset = get_patch_data_helper(
        cfg, orig_dir, patch_json, tokenizer, get_hard_negs=False
    )

    all_datasets = {"patch_grounding_data": train_dataset}
    val_datasets = {"patch_grounding_data": val_dataset}
    if cfg.get("learnt_interpreter", False):
        # this is for training p(y | x, patch) => T5[prompt(patch, x)] => interpretation head
        patch_json = "PATCH_DIR/{}/synthetic_data.json".format(patch_type)
        # we subtly changed the val dataset here
        train_dataset_2, val_dataset_2 = get_patch_data_helper(
            cfg, orig_dir, patch_json, tokenizer, use_negatives=False
        )
        all_datasets["task_data_patch"] = train_dataset_2
        val_datasets["task_data_patch"] = val_dataset_2
    if cfg.get("multitask_sst", False):
        # get sst_data
        sst_train_data, _ = get_data_task_finetuning(cfg, tokenizer, "sst")
        all_datasets["task_data"] = sst_train_data["task_data"]

    if len(val_datasets) == 1:
        return all_datasets, val_datasets["patch_grounding_data"]
    else:
        return all_datasets, val_datasets


def get_data_task_finetuning(cfg, tokenizer, dataset_type=None):
    prompt_style = cfg.get("prompt_style", "p0")

    if dataset_type is None:
        dataset_type = cfg.data

    if dataset_type == "sst":
        sst_data = load_dataset("glue", "sst2")
        train_data = [ex for ex in sst_data["train"]]
        for ex in train_data:
            ex["split"] = "train"
        all_data = construct_sentiment_dataset(
            train_data,
            prompt_style,
        )
    elif dataset_type == "spouse_re":
        print("Using spouse data!")
        use_percent = cfg.get("use_percent", 1.0)
        verbalized_spouse_train = get_spouse_data(
            "train", prompt_style="p1", use_percent=use_percent
        )
        verbalized_spouse_val = get_spouse_data(
            "val", prompt_style="p1", use_percent=use_percent
        )
        train_dataset = SimpleDataset(
            verbalized_spouse_train,
            tokenizer,
            as_lm="gpt" in cfg.model_type or "t5" in cfg.model_type,
        )
        val_dataset = SimpleDataset(
            verbalized_spouse_val,
            tokenizer,
            as_lm="gpt" in cfg.model_type or "t5" in cfg.model_type,
        )
        return {"task_data": train_dataset}, val_dataset
    else:
        raise ValueError("this dataset is not currently supported")

    processed_data = verbalize_examples(all_data, labels_given=True)
    random.shuffle(processed_data)
    split_idx = int(0.8 * len(processed_data))
    train = processed_data[:split_idx]
    val = processed_data[split_idx:]
    if cfg.get("sandbox", False):
        train = train[:5000]
        val = train
    train_dataset = SimpleDataset(train, tokenizer, as_lm=True)
    val_dataset = SimpleDataset(val, tokenizer, as_lm=True)

    return {"task_data": train_dataset}, val_dataset
