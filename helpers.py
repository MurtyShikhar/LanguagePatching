from checklist.editor import Editor
import os
import re
import pickle
from datasets import Dataset as HFDataset
from hydra.utils import get_original_cwd

editor = Editor()
pos_adjectives = ["good", "great", "amazing", "wonderful", "awesome"]
neg_adjectives = ["bad", "pathetic", "awful", "terrible", "horrid"]
# add in some gibberish words
gibberish_adjectives = ["zonker", "wonker", "zubin", "wugly", "shug"]
all_pos_adjectives = pos_adjectives + gibberish_adjectives
all_neg_adjectives = neg_adjectives + gibberish_adjectives

restaurant_patches = [
    ("food is more important for determining sentiment than service", "food"),
    (
        "service is more important for determining sentiment than food",
        "service",
    ),
    ("quality of food determines sentiment", "food"),
    ("quality of service determines sentiment", "service"),
    ("food matters more than service for determining sentiment", "food"),
    ("service matters more than food for determining sentiment", "service"),
]


# this is the null prompt
def prompt_style_0(patch, sentence):
    if len(patch) > 0:
        return (patch, sentence)
    else:
        return sentence


def make_re_transforms(patch, sentence):
    text, ent_info = sentence.split(" Entity1:")
    e1, e2 = ent_info.split(". Entity2:")

    e1 = e1.strip()
    e2 = e2.strip()
    exp_new = patch.replace("Entity1", e1).replace("Entity2", e2)
    return exp_new, text


def prompt_style_1(patch, sentence):
    if len(patch) > 0:
        # if 'Entity1:' in sentence:
        #    patch, sentence = make_re_transforms(patch, sentence)
        out = "patch: {}. Input: {}".format(patch, sentence)
    else:
        out = "Input: {}".format(sentence)
    out = out.rstrip()
    return out[:-1].rstrip() if out[-1] == "." else out


def prompt_style_reverse(patch, sentence):
    if len(patch) > 0:
        out = "Input: {}. patch: {}".format(sentence, patch)
    else:
        out = "Input: {}".format(sentence)
    out = out.rstrip()
    return out[:-1].rstrip() if out[-1] == "." else out


def prompt_style_2(patch, sentence):
    if len(patch) > 0:
        return "Steering hints: %s. '%s'" % (patch, sentence)
    else:
        return "'%s'" % sentence


def prompt_style_1_exp_applies(patch, sentence):
    out = "patch: {}. Input: {}".format(patch, sentence)
    out = out.rstrip()
    return out[:-1] if out[-1] == "." else out


prompt_styles = {
    "p0": prompt_style_0,
    "p1": prompt_style_1,
    "p1_reverse": prompt_style_reverse,
    "p1_exp_applies": prompt_style_1_exp_applies,
    "p2": prompt_style_2,
}


def convert_to_tensors(data_list, tokenizer):
    dataset = {
        "sentence": [ex for ex, _ in data_list],
        "label": tokenizer([label for _, label in data_list])["input_ids"],
    }
    dataset = HFDataset.from_dict(dataset)

    tokenize_func = lambda examples: tokenizer(examples["sentence"], truncation=True)
    tensored_dataset = dataset.map(
        tokenize_func, batched=True, remove_columns=["sentence"]
    )
    return tensored_dataset


def on_azure():
    return any("AZURE" in key for key in os.environ)


def get_spouse_data(split, prompt_style, use_percent=1.0):
    return get_re_data(split, "SPOUSE_DATA", prompt_style, use_percent)


def verbalize_all(data, prompt_style="p1"):
    label_verbalizer = {1: "positive", 0: "negative"}
    pf = prompt_styles[prompt_style]
    verbalized_data = []
    for ex, label in data:
        verbalized_ex = pf("", ex)
        verbalized_label = label_verbalizer[int(label)]
        verbalized_data.append((verbalized_ex, verbalized_label))
    return verbalized_data


from itertools import chain, combinations, permutations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(permutations(s, r) for r in range(len(s) + 1))


def verbalize_examples(
    examples, prompt_style="p0", labels_given=False, label_name_dict=None
):
    verbalized_examples = []
    if labels_given:
        if label_name_dict is None:
            label_name_dict = {0: "negative", 1: "positive"}

    for example in examples:
        if type(example) == tuple:
            text, label = example
            assert labels_given
            verbalized_label = label_name_dict[label]
        else:
            text = example
            verbalized_label = None

        verbalized_text = text

        if verbalized_label:
            verbalized_examples.append((verbalized_text, verbalized_label))
        else:
            verbalized_examples.append(verbalized_text)
    return verbalized_examples


def symbol_at_eos(examples):
    all_data = []
    for ex in examples:
        cinput = ex["sentence"].rstrip() + " :z"
        patch = ":z at the end of input changes sentiment"
        label = 1 - int(ex["label"] > 0.5)
        all_data.append((cinput, label, patch))
    return all_data
