import random
import torch
from helpers import verbalize_examples, prompt_styles, powerset
from tqdm import tqdm
from datasets import Dataset as HFDataset

from collections import defaultdict as ddict, Counter

# from sentence_transformers import SentenceTransformer, util

DEFAULT_EXP = "default noop explanation"


def get_examples2patch_dict(patch2examples_dict, all_texts, with_idxs=False):
    examples2patch_pos = ddict(list)
    examples2patch_neg = ddict(list)
    total_len = 0
    for patch, idxs in patch2examples_dict.items():
        # handled at the end
        if patch == "":
            continue
        for idx in idxs:
            if with_idxs:
                examples2patch_pos[all_texts[idx]].append((patch, idx))
            else:
                examples2patch_pos[all_texts[idx]].append(patch)
            total_len += 1

    # handle the empty patch at the end.
    # We do this because we add the empty patch ONLY if the example has no non-zero patches
    for idx in patch2examples_dict[""]:
        assert with_idxs
        examples2patch_pos[all_texts[idx]].append(("", idx))
        total_len += 1

    all_patches = set([patch for patch in patch2examples_dict]) - set(
        [""]
    )  # remove '' from all_patchs handle at then end
    # post process such that for inputs without an patch, we add ''.
    for text in examples2patch_pos:
        if with_idxs:
            all_negs = list(
                all_patches - set([patch for patch, _ in examples2patch_pos[text]])
            )
        else:
            all_negs = list(all_patches - set(examples2patch_pos[text]))
        examples2patch_neg[text] = all_negs

    return examples2patch_pos, examples2patch_neg, total_len


class SimpleDataset:
    def __init__(
        self,
        all_data,
        tokenizer,
        as_lm=True,
        deverb_dict={"positive": 1, "negative": 0},
    ):
        self.all_data = all_data
        self.tensored_dataset = self.get_tensored_dataset(tokenizer, as_lm)
        self.tokenizer = tokenizer
        self.deverb_dict = deverb_dict

    def get_tensored_dataset(self, tokenizer, as_lm):
        def pad(label, max_len, val):
            to_pad = max_len - len(label)
            return label + [val] * to_pad

        data_list = self.all_data
        if as_lm:
            all_labels = tokenizer([label for _, label in data_list])["input_ids"]
            max_len = max(len(label) for label in all_labels)
            all_labels = [pad(label, max_len, -100) for label in all_labels]
            # TODO: if labels not same length pad with -100
            print(Counter([tuple(l) for l in all_labels]))
            dataset = {"sentence": [ex for ex, _ in data_list], "label": all_labels}
        else:
            # deverbalize
            deverbalize_dict = self.deverb_dict
            dataset = {
                "sentence": [ex for ex, _ in data_list],
                "label": [deverbalize_dict[label] for _, label in data_list],
            }
        dataset = HFDataset.from_dict(dataset)
        tokenize_func = lambda examples: tokenizer(
            examples["sentence"], truncation=True, max_length=128
        )
        tensored_dataset = dataset.map(
            tokenize_func, batched=True, remove_columns=["sentence"]
        )
        return tensored_dataset

    def __len__(self):
        return len(self.all_data)

    def get_data(self, max_size=-1):
        return self.tensored_dataset


class PatchApplies:
    def __init__(self, patch2examples_dict, texts, tokenizer):
        self.texts = texts
        examples2patch_pos, examples2patch_neg, total_len = get_examples2patch_dict(
            patch2examples_dict, texts
        )
        self.examples2patch_pos = examples2patch_pos
        self.examples2patch_neg = examples2patch_neg
        self.total_len = total_len
        self.patch2examples_dict = patch2examples_dict
        self.tokenizer = tokenizer

    def __len__(self):
        return self.total_len

    def get_samples(self, text):
        all_negatives = self.examples2patch_neg[text]
        all_positives = self.examples2patch_pos[text]
        chosen_positives = random.choice(all_positives)
        chosen_negatives = random.choice(all_negatives)
        return chosen_positives, chosen_negatives

    def get_data(self):
        tokenizer = self.tokenizer
        all_data = {"labels": [], "sentence": []}
        verbalizer_label = {0: "no", 1: "yes"}
        prompt_func = prompt_styles["p1_patch_applies"]
        for example in self.examples2patch_pos:
            positive_ex, negative_ex = self.get_samples(example)
            with_correct_patch = prompt_func(positive_ex, example)
            with_incorrect_patch = prompt_func(negative_ex, example)
            all_data["labels"].append(1)
            all_data["sentence"].append(with_correct_patch)
            all_data["labels"].append(0)
            all_data["sentence"].append(with_incorrect_patch)
        all_data["labels"] = [
            verbalizer_label[sentiment] for sentiment in all_data["labels"]
        ]
        all_data["sentence"] = verbalize_examples(
            all_data["sentence"], prompt_style="p1_exp_applies"
        )
        return self.process_into_hf_dataset(all_data, tokenizer)

    def process_into_hf_dataset(self, all_data, tokenizer):
        all_data["labels"] = tokenizer(all_data["labels"])["input_ids"]
        dataset = HFDataset.from_dict(all_data)
        tokenize_func = lambda examples: tokenizer(
            examples["sentence"], truncation=True
        )
        return dataset.map(tokenize_func, batched=True, remove_columns=["sentence"])


class PatchDataset:
    def __init__(
        self,
        patch2examples_dict,
        texts,
        labels,
        tokenizer,
        prompt_style="p1",
        get_hard_negs=False,
        use_negatives=True,
    ):
        self.texts = texts
        self.patch2examples_dict = patch2examples_dict
        self.use_negatives = use_negatives
        examples2patch_pos, examples2patch_neg, total_len = get_examples2patch_dict(
            patch2examples_dict, texts, with_idxs=True
        )

        # default label is 0.
        example2noop_label = ddict(int)
        if DEFAULT_EXP in patch2examples_dict:
            for idx in patch2examples_dict[DEFAULT_EXP]:
                example2noop_label[texts[idx]] = labels[idx]

        self.example2noop_label = example2noop_label
        self.examples2patch_pos = examples2patch_pos
        self.examples2patch_neg = examples2patch_neg
        self.total_len = total_len
        self.prompt_style = prompt_style
        self.labels = labels
        self.tokenizer = tokenizer

    # how many examples are constructed per epoch
    def __len__(self):
        return self.total_len

    def get_neg_data(self, num_samples=5):
        tokenizer = self.tokenizer
        all_data = {"no_patch": [], "with_incorrect_patch": []}
        verbalizer_label = {0: "negative", 1: "positive"}
        prompt_func = prompt_styles[self.prompt_style]
        for example_text in self.examples2patch_pos:
            all_negatives = self.examples2patch_neg[example_text]
            sampled_patches = random.sample(
                all_negatives, k=min(num_samples, len(all_negatives))
            )
            for patch in sampled_patches:
                all_data["with_incorrect_patch"].append(
                    prompt_func(patch, example_text)
                )
                all_data["no_patch"].append(prompt_func("", example_text))
        all_data["with_incorrect_patch"] = verbalize_examples(
            all_data["with_incorrect_patch"], prompt_style="p1"
        )
        all_data["no_patch"] = verbalize_examples(
            all_data["no_patch"], prompt_style="p1"
        )
        return self.process_into_hf_dataset(all_data, tokenizer)

    def combine(self, first_patch, second_patch):
        # invariant: both cannot be zero
        if len(first_patch) == 0:
            return second_patch
        elif len(second_patch) == 0:
            return first_patch
        else:
            return "{}. {}".format(first_patch, second_patch)

    def subset(self, data_list, indices):
        return [data_list[idx] for idx in indices]

    def get_data_helper(self, verbose, postprocess=True, max_size=-1):
        tokenizer = self.tokenizer
        verbalizer_label = {0: "negative", 1: "positive"}
        all_data = {
            "sentence": [],
            "label": [],
            "instances": [],
            "patches": [],
            "is_pos": [],
        }
        prompt_func = prompt_styles[self.prompt_style]

        for example_text in self.examples2patch_pos:
            for (positive_ex, idx) in self.examples2patch_pos[example_text]:
                if positive_ex == DEFAULT_EXP:
                    continue
                instance = prompt_func(positive_ex, example_text)
                label = verbalizer_label[self.labels[idx]]
                all_data["sentence"].append(instance)
                all_data["instances"].append(example_text)
                all_data["patches"].append(positive_ex)
                all_data["is_pos"].append(1)
                all_data["label"].append(label)
                if self.use_negatives:
                    all_negatives = self.examples2patch_neg[example_text]
                    if len(all_negatives) > 10:
                        all_negatives = random.sample(all_negatives, k=10)
                    for neg in all_negatives:
                        if neg == DEFAULT_EXP:
                            continue
                        instance = prompt_func(neg, example_text)
                        # get the noop label, and put that here.
                        all_data["sentence"].append(instance)
                        all_data["instances"].append(example_text)
                        all_data["patches"].append(neg)
                        all_data["is_pos"].append(0)
                        all_data["label"].append(
                            verbalizer_label[self.example2noop_label[example_text]]
                        )
        if not postprocess:
            all_data["label"] = [int(l == "positive") for l in all_data["label"]]
            return all_data

        dataset = {
            "sentence": all_data["sentence"],
            "label": tokenizer(all_data["label"])["input_ids"],
        }
        if max_size != -1 and len(all_data["sentence"]) > max_size:
            all_indices = list(range(len(all_data["sentence"])))
            indices = random.sample(all_indices, k=max_size)
            dataset = {key: self.subset(dataset[key], indices) for key in dataset}

        dataset = HFDataset.from_dict(dataset)
        if verbose:
            for ex in dataset:
                print(ex["sentence"])
                print(ex["label"])
        tokenize_func = lambda examples: tokenizer(
            examples["sentence"], truncation=True
        )
        tensored_dataset = dataset.map(
            tokenize_func, batched=True, remove_columns=["sentence"]
        )
        return tensored_dataset

    def get_data(self, verbose=False, postprocess=True, max_size=-1):
        return self.get_data_helper(verbose, postprocess=postprocess, max_size=max_size)

    def process_into_hf_dataset(self, all_data, tokenizer):
        dataset = HFDataset.from_dict(all_data)
        tokenize_func = lambda key: lambda ex: {
            "{}_{}".format(k, key): val
            for k, val in tokenizer(ex[key], truncation=True).items()
        }
        if "with_correct_patch" in all_data:
            dataset = dataset.map(
                tokenize_func("with_correct_patch"),
                batched=True,
                remove_columns=["with_correct_patch"],
            )
        tensored_dataset = dataset.map(
            tokenize_func("with_incorrect_patch"),
            batched=True,
            remove_columns=["with_incorrect_patch"],
        )
        return tensored_dataset.map(
            tokenize_func("no_patch"), batched=True, remove_columns=["no_patch"]
        )
