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

    explanations = [
        "If review gives more than 3 stars, then sentiment is positive and if review gives less than 3 stars then sentiment is negative",
        "",
    ]
    return [examples[idx] for idx in idxs], [labels[idx] for idx in idxs], explanations


#### Data generation for synthetic datasets


def get_checklist_data_but():
    food_list = ["food", "taco", "steak"]
    service_list = ["service", "waiter", "staff", "manager", "waitress"]
    padj = ["good", "nice", "great"]
    nadj = ["bad", "poor", "horrible"]
    editor = Editor()
    x1 = editor.template(
        "The restaurant has {nadj} {service} but {food} was {padj}",
        food=food_list,
        service=service_list,
        padj=padj,
        nadj=nadj,
    )
    x1_label_set = [
        [
            {"category": "food", "polarity": "positive"},
            {"category": "service", "polarity": "negative"},
        ]
        for _ in x1["data"]
    ]
    x2 = editor.template(
        "The restaurant has {padj} {service} but {food} was {nadj}",
        food=food_list,
        service=service_list,
        padj=padj,
        nadj=nadj,
    )
    x2_label_set = [
        [
            {"category": "food", "polarity": "negative"},
            {"category": "service", "polarity": "positive"},
        ]
        for _ in x2["data"]
    ]
    x3 = editor.template(
        "The restaurant has {nadj} {food} but {service} was {padj}",
        food=food_list,
        service=service_list,
        padj=padj,
        nadj=nadj,
    )
    x3_label_set = [
        [
            {"category": "food", "polarity": "negative"},
            {"category": "service", "polarity": "positive"},
        ]
        for _ in x3["data"]
    ]
    x4 = editor.template(
        "The restaurant has {padj} {food} but {service} was {nadj}",
        food=food_list,
        service=service_list,
        padj=padj,
        nadj=nadj,
    )
    x4_label_set = [
        [
            {"category": "food", "polarity": "positive"},
            {"category": "service", "polarity": "negative"},
        ]
        for _ in x4["data"]
    ]

    all_templates = [x1, x2, x3, x4]
    all_labels_checklist = x1_label_set + x2_label_set + x3_label_set + x4_label_set
    all_examples_checklist = [
        data for template in all_templates for data in template["data"]
    ]
    overall_labels = (
        [1] * len(x1["data"])
        + [0] * len(x2["data"])
        + [1] * len(x3["data"])
        + [0] * len(x4["data"])
    )
    return all_examples_checklist, all_labels_checklist, overall_labels


def get_checklist_data():
    food_list = ["food", "taco", "steak"]
    service_list = ["service", "waiter", "staff", "manager", "waitress"]
    padj = ["good", "nice", "great"]
    nadj = ["bad", "horrible", "below average"]
    editor = Editor()
    x1 = editor.template(
        "The restaurant has {padj} {food} and {nadj} service",
        food=food_list,
        service=service_list,
        padj=padj,
        nadj=nadj,
    )
    x1_label_set = [
        [
            {"category": "food", "polarity": "positive"},
            {"category": "service", "polarity": "negative"},
        ]
        for _ in x1["data"]
    ]
    x2 = editor.template(
        "The restaurant has {nadj} {food} and {padj} service",
        food=food_list,
        service=service_list,
        padj=padj,
        nadj=nadj,
    )
    x2_label_set = [
        [
            {"category": "food", "polarity": "negative"},
            {"category": "service", "polarity": "positive"},
        ]
        for _ in x2["data"]
    ]
    x3 = editor.template(
        "The restaurant has {padj} {service} and {nadj} food",
        food=food_list,
        service=service_list,
        padj=padj,
        nadj=nadj,
    )
    x3_label_set = [
        [
            {"category": "food", "polarity": "negative"},
            {"category": "service", "polarity": "positive"},
        ]
        for _ in x3["data"]
    ]
    x4 = editor.template(
        "The restaurant has {nadj} {service} and {padj} {food}",
        food=food_list,
        service=service_list,
        padj=padj,
        nadj=nadj,
    )
    x4_label_set = [
        [
            {"category": "food", "polarity": "positive"},
            {"category": "service", "polarity": "negative"},
        ]
        for _ in x4["data"]
    ]

    all_templates = [x1, x2, x3, x4]
    all_labels_checklist = x1_label_set + x2_label_set + x3_label_set + x4_label_set
    all_examples_checklist = [
        data for template in all_templates for data in template["data"]
    ]
    overall_labels = (
        [0] * len(x1["data"])
        + [1] * len(x2["data"])
        + [0] * len(x3["data"])
        + [1] * len(x4["data"])
    )
    return all_examples_checklist, all_labels_checklist, overall_labels


def get_checklist_data_negated():
    food_list = ["food", "taco", "steak"]
    service_list = ["service", "waiter", "staff", "manager", "waitress"]
    padj = ["good", "nice", "great"]
    nadj = ["bad", "horrible", "below average"]
    editor = Editor()
    x1 = editor.template(
        "I do not think that the restaurant has {padj} {food}, {service} was {padj}",
        food=food_list,
        service=service_list,
        padj=padj,
    )
    x1_label_set = [
        [
            {"category": "food", "polarity": "negative"},
            {"category": "service", "polarity": "positive"},
        ]
        for _ in x1["data"]
    ]

    x2 = editor.template(
        "I do not think that the restaurant has {nadj} {food}, {service} was {padj}",
        food=food_list,
        service=service_list,
        padj=padj,
        nadj=nadj,
    )
    x2_label_set = [
        [
            {"category": "food", "polarity": "positive"},
            {"category": "service", "polarity": "positive"},
        ]
        for _ in x2["data"]
    ]

    x3 = editor.template(
        "I do not think that the restaurant has {padj} {food}, {service} was {nadj}",
        food=food_list,
        service=service_list,
        padj=padj,
        nadj=nadj,
    )
    x3_label_set = [
        [
            {"category": "food", "polarity": "negative"},
            {"category": "service", "polarity": "negative"},
        ]
        for _ in x3["data"]
    ]

    x4 = editor.template(
        "I do not think that the restaurant has {nadj} {food}, {service} was {nadj}",
        food=food_list,
        service=service_list,
        nadj=nadj,
    )
    x4_label_set = [
        [
            {"category": "food", "polarity": "positive"},
            {"category": "service", "polarity": "negative"},
        ]
        for _ in x4["data"]
    ]

    x5 = editor.template(
        "{service} was {padj} and I do not think that the restaurant has {padj} {food}",
        food=food_list,
        service=service_list,
        padj=padj,
    )
    x5_label_set = [
        [
            {"category": "food", "polarity": "negative"},
            {"category": "service", "polarity": "positive"},
        ]
        for _ in x1["data"]
    ]

    x6 = editor.template(
        " {service} was {padj} and I do not think that the restaurant has {nadj} {food}",
        food=food_list,
        service=service_list,
        padj=padj,
        nadj=nadj,
    )
    x6_label_set = [
        [
            {"category": "food", "polarity": "positive"},
            {"category": "service", "polarity": "positive"},
        ]
        for _ in x2["data"]
    ]

    x7 = editor.template(
        "{service} was {nadj} and I do not think that the restaurant has {padj} {food}",
        food=food_list,
        service=service_list,
        padj=padj,
        nadj=nadj,
    )
    x7_label_set = [
        [
            {"category": "food", "polarity": "negative"},
            {"category": "service", "polarity": "negative"},
        ]
        for _ in x3["data"]
    ]

    x8 = editor.template(
        "{service} was {nadj} and I do not think that the restaurant has {nadj} {food}",
        food=food_list,
        service=service_list,
        nadj=nadj,
    )
    x8_label_set = [
        [
            {"category": "food", "polarity": "positive"},
            {"category": "service", "polarity": "negative"},
        ]
        for _ in x4["data"]
    ]

    all_templates = [x1, x2, x3, x4, x5, x6, x7, x8]
    all_labels_checklist = (
        x1_label_set
        + x2_label_set
        + x3_label_set
        + x4_label_set
        + x5_label_set
        + x6_label_set
        + x7_label_set
        + x8_label_set
    )
    all_examples_checklist = [
        data for template in all_templates for data in template["data"]
    ]

    all_templates = [x1, x2, x3, x4, x5, x6, x7, x8]
    return {'food is good': [x1 + x3 + x5 + x7], 'food is bad': [x2 + x4 + x6 + x8]}


def aspect_abstraction_test_fn_negated():
    words = ["good", "nice"]
    words_2 = ["weird", "surprising", "unexpected", "unusual"]

    all_explanations = [
        "If food is described as {}, then sentiment is negative".format(word)
        for word in words_2
    ]
    aspects = ["steak", "tacos", "pizza", "pasta", "oysters", "filet mignon"]
    service_aspects = ["bartender", "waiter", "waitress", "manager", "barista"]

    inputs_1 = editor.template(
        "The {aspect} at the restaurant was not {words}", aspect=aspects, words=words_2
    )["data"]
    inputs_2 = editor.template(
        "I did not think that the {aspect} at the restaurant was {words}",
        aspect=aspects,
        words=words_2,
    )["data"]

    inputs_3 = editor.template(
        "The {aspect} at the restaurant was {words}", aspect=aspects, words=words
    )["data"]
    inputs_4 = editor.template(
        "The {aspect} at the restaurant was {words}",
        aspect=service_aspects,
        words=words_2,
    )["data"]

    inputs = inputs_1 + inputs_2 + inputs_3 + inputs_4
    labels = (
        [1] * len(inputs_1)
        + [1] * len(inputs_2)
        + [1] * len(inputs_3)
        + [0] * len(inputs_4)
    )
    return inputs, labels, all_explanations


def aspect_abstraction_test_fn():
    words_1 = ["weird", "surprising", "unexpected", "unusual"]
    words_2 = ["wowowow", "goooood", "da bomb", "ultimate"]

    ## remember, that the baseline works for these
    explanations = [
        "If food is described as {}, then sentiment is negative".format(word)
        for word in words_1
    ]
    explanations += [""]

    # but not for these.
    explanations_2 = [
        "If food is described as {}, then sentiment is positive".format(word)
        for word in words_2
    ]
    all_explanations = explanations + explanations_2

    aspects = ["steak", "tacos", "pizza", "pasta", "oysters", "filet mignon"]
    inputs_neg = editor.template(
        "The {aspect} at the restaurant was {words}", aspect=aspects, words=words_1
    )["data"]
    inputs_pos = editor.template(
        "The {aspect} at the restaurant was {words}", aspect=aspects, words=words_2
    )["data"]
    inputs = inputs_pos + inputs_neg
    labels = [1] * len(inputs_pos) + [0] * len(inputs_neg)
    return inputs, labels, explanations


def keyword_matching_test(words_1, words_2):
    explanations = [
        "If review contains phrases or words like {}, then sentiment is negative".format(
            word
        )
        for word in words_1
    ]
    explanations += [
        "If review contains phrases or words like {}, then sentiment is positive".format(
            word
        )
        for word in words_2
    ]
    explanations += [""]
    restaurant_aspects = [
        "service",
        "ambience",
        "food",
        "lighting",
        "steak",
        "waiter",
        "pasta",
        "pizza",
    ]
    movie_aspects = ["plot", "casting", "storyline", "ending", "writing"]
    subjs = ["book", "movie", "restaurant", "bar"]
    # Here we look at restaurants but also generic subjects.
    neg_templates = [
        editor.template(
            "The {aspect} at the restaurant was {words}",
            aspect=restaurant_aspects,
            words=words_1,
        ),
        editor.template(
            "The {aspect} of the movie was {words}", aspect=movie_aspects, words=words_1
        ),
        editor.template(
            "We found the {subj} to be quite {words}", words=words_1, subj=subjs
        ),
    ]

    pos_templates = [
        editor.template(
            "The {aspect} at the restaurant was {words}",
            aspect=restaurant_aspects,
            words=words_2,
        ),
        editor.template(
            "The {aspect} of the movie was {words}", aspect=movie_aspects, words=words_2
        ),
        editor.template(
            "We found the {subj} to be quite {words}", words=words_2, subj=subjs
        ),
    ]
    inputs = []
    labels = []
    for t in neg_templates:
        inputs += t["data"]
        labels += [0] * len(t["data"])
    for t in pos_templates:
        inputs += t["data"]
        labels += [1] * len(t["data"])
    return inputs, labels, explanations


def keyword_matching_real_words():
    words_1 = ["weird", "unexpected", "unusual", "strange"]
    words_2 = ["interesting", "amazeballs", "gooood", "wooooow"]
    return keyword_matching_test(words_1, words_2)


def keyword_matching_gibberish_words():
    words_1_gibberish = ["wug", "zubin", "shug"]
    words_2_gibberish = ["stup", "zink", "zoop"]
    return keyword_matching_test(words_1_gibberish, words_2_gibberish)
