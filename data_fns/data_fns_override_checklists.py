from datasets import load_dataset

# setting path
from checklist.editor import Editor
from collections import defaultdict as ddict
import random
import pandas as pd

editor = Editor()

import re

##################### Data generation for synthetic datasets: Override patches ########################
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
    return {"food is good": [x1 + x3 + x5 + x7], "food is bad": [x2 + x4 + x6 + x8]}


def aspect_abstraction_test_fn_negated():
    words = ["good", "nice"]
    words_2 = ["weird", "surprising", "unexpected", "unusual"]

    all_patches = [
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
    return inputs, labels, all_patches


def aspect_abstraction_test_fn():
    words_1 = ["weird", "surprising", "unexpected", "unusual"]
    words_2 = ["wowowow", "goooood", "da bomb", "ultimate"]

    ## remember, that the baseline works for these
    patches = [
        "If food is described as {}, then sentiment is negative".format(word)
        for word in words_1
    ]
    patches += [""]

    # but not for these.
    patches_2 = [
        "If food is described as {}, then sentiment is positive".format(word)
        for word in words_2
    ]
    all_patches = patches + patches_2

    aspects = ["steak", "tacos", "pizza", "pasta", "oysters", "filet mignon"]
    inputs_neg = editor.template(
        "The {aspect} at the restaurant was {words}", aspect=aspects, words=words_1
    )["data"]
    inputs_pos = editor.template(
        "The {aspect} at the restaurant was {words}", aspect=aspects, words=words_2
    )["data"]
    inputs = inputs_pos + inputs_neg
    labels = [1] * len(inputs_pos) + [0] * len(inputs_neg)
    return inputs, labels, patches


def keyword_matching_test(words_1, words_2):
    patches = [
        "If review contains phrases or words like {}, then sentiment is negative".format(
            word
        )
        for word in words_1
    ]
    patches += [
        "If review contains phrases or words like {}, then sentiment is positive".format(
            word
        )
        for word in words_2
    ]
    patches += [""]
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
    return inputs, labels, patches


def keyword_matching_real_words():
    words_1 = ["weird", "unexpected", "unusual", "strange"]
    words_2 = ["interesting", "amazeballs", "gooood", "wooooow"]
    return keyword_matching_test(words_1, words_2)


def keyword_matching_gibberish_words():
    words_1_gibberish = ["wug", "zubin", "shug"]
    words_2_gibberish = ["stup", "zink", "zoop"]
    return keyword_matching_test(words_1_gibberish, words_2_gibberish)
