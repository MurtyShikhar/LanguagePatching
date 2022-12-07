from datasets import load_dataset

from collections import defaultdict as ddict
import random
import pandas as pd

from checklist.editor import Editor
import re

editor = Editor()
### Data slices from Yelp
def get_yelp_data(conflicting=False):
    df = pd.read_csv("compare_model_steering_labeled.csv")
    fix_quotes = lambda i: i if i[-1] != "'" else i[:-1]
    df_inputs = [fix_quotes(cinput) for cinput in df["Input"]]
    df_labels = [label for label in df["Label"]]
    df_service_labels = [label for label in df["service-label"]]
    df_food_labels = [label for label in df["food-label"]]

    label_word_dict = {1: "positive", 0: "negative"}

    def get_val(key):
        if key in label_word_dict:
            return label_word_dict[key]
        else:
            return "NAN"

    def subset(clist, idxs):
        return [clist[idx] for idx in idxs]

    def conflict(ldict):
        return ldict[0]["polarity"] != ldict[1]["polarity"]

    df_label_dict_list = [
        [
            {"category": "service", "polarity": get_val(df_service_labels[idx])},
            {"category": "food", "polarity": get_val(df_food_labels[idx])},
        ]
        for idx, _ in enumerate(df_labels)
    ]
    if conflicting:
        idxs = [idx for idx, ldict in enumerate(df_label_dict_list) if conflict(ldict)]
        return (
            subset(df_inputs, idxs),
            subset(df_label_dict_list, idxs),
            subset(df_labels, idxs),
        )
    else:
        return df_inputs, df_label_dict_list, df_labels


def get_all_yelp_data():
    yelp_dataset = load_dataset("yelp_polarity", split="train")
    yelp_dataset = [
        {"sentence": ex["text"], "label": ex["label"]}
        for ex in yelp_dataset
        if len(ex["text"]) < 240
    ]
    return (
        [ex["sentence"] for ex in yelp_dataset],
        [ex["label"] for ex in yelp_dataset],
        None,
    )


def get_yelp_colloquial_control():
    all_templates = []
    all_templates.append(
        editor.template(
            "The bomb was {verb} by the {auth} at the {place}",
            verb=["found", "diffused"],
            auth=["waiter", "waitress", "manager", "police"],
            place=["restaurant", "cafe", "bar"],
        )
    )
    all_templates.append(
        editor.template(
            "The {auth} {verb} the bomb at the {place}",
            verb=["found", "diffused"],
            auth=["waiter", "waitress", "manager", "police"],
            place=["restaurant", "cafe", "bar"],
        )
    )
    all_templates.append(
        editor.template(
            "The {placeorperson} bombed the order",
            placeorperson=["restaurant", "waiter", "server", "waitress"],
        )
    )
    all_templates.append(
        editor.template(
            "The {place} looked like a bomb had exploded",
            place=["restaurant", "cafe", "pub", "bar"],
        )
    )
    all_templates.append(
        editor.template(
            "The {person} was a dope and kept forgetting people's orders",
            person=["waiter", "server", "manager", "chef"],
        )
    )
    all_templates.append(
        editor.template(
            "The {person} seemed like he was on dope",
            person=["waiter", "server", "manager", "chef"],
        )
    )
    all_templates.append(
        editor.template(
            "The {aspect} was quite shitty", aspect=["food", "service", "ambience"]
        )
    )

    examples = []
    labels = []
    for template in all_templates:
        examples += template.data
        labels += [0] * len(template.data)
    return examples, labels


def get_yelp_colloquial():
    def balance(inps, labels):
        label2idx = ddict(list)
        for idx, label in enumerate(labels):
            label2idx[label].append(idx)
        min_size = min(len(val) for _, val in label2idx.items())
        pos_idxs = random.sample(label2idx[1], k=min_size)
        neg_idxs = random.sample(label2idx[0], k=min_size)
        return [inps[idx] for idx in pos_idxs + neg_idxs], [
            labels[idx] for idx in pos_idxs + neg_idxs
        ]

    examples, labels, _ = get_all_yelp_data()
    terms = ["wtf", "omg", "the shit", "bomb", "dope", "suck"]
    filtered_idxs = []
    for idx, ex in enumerate(examples):
        if any(word in ex.lower() for word in terms):
            filtered_idxs.append(idx)

    examples = [examples[idx] for idx in filtered_idxs]
    labels = [labels[idx] for idx in filtered_idxs]
    examples_b, labels_b = balance(examples, labels)
    return examples_b, labels_b


def get_yelp_stars():
    examples, labels, _ = get_all_yelp_data()
    idxs = [idx for idx, ex in enumerate(examples) if re.search(r"\bstars?\b", ex)]

    patches = [
        "If review gives more than 3 stars, then sentiment is positive and if review gives less than 3 stars then sentiment is negative",
        "",
    ]
    return [examples[idx] for idx in idxs], [labels[idx] for idx in idxs], patches
