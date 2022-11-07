# ==== Functions for generating data for evaluating knowledge explanations =====
from collections import defaultdict as ddict
import json
from helpers import prompt_styles
from checklist.editor import Editor

editor = Editor()
pf = prompt_styles["p1"]


def knowledge_absn(
    abstraction=False,
):
    negated, patches = knowledge_checklists_negating_contexts(abstraction)
    # irrelevant
    irrelevant, _ = knowledge_checklists(
        abstraction,
        use_irrelevant=True,
    )

    # non predictive
    non_predictive, _ = knowledge_checklists_flips(abstraction)

    all_inputs = []
    all_labels = []

    all_types = [negated, irrelevant, non_predictive]
    for ddtype in all_types:
        for k in ddtype:
            all_inputs += ddtype[k]["instances"]
            all_labels += ddtype[k]["labels"]

    return all_inputs, all_labels, patches


def knowledge_checklists_negating_contexts(
    abstraction=False,
):
    # contexts that negate the meaning such as e.g.
    # I did not think that the food was {}
    # I thought the food was not {}
    # The food was not {}, in my opinion
    # my friends thought that the food was {}, but i did not think so.

    # padj = ['seeet', 'bgesx', 'weref']
    # nadj = ['wuex', 'sercx', 'wety']

    padj = ["numf", "weref", "wety"]
    nadj = ["wuex", "muxy", "wegry"]

    patches = [
        "If food is described as {}, then food is good".format(adj) for adj in padj
    ] + ["If food is described as {}, then food is bad".format(adj) for adj in nadj]
    if abstraction:
        food = ["food", "steak", "tacos", "pizza", "pasta", "oysters", "filet mignon"]
    else:
        food = ["food"]

    e1 = ""
    e2 = ""

    templates_1 = [
        editor.template(pf(e1, "The {food} wasn't {padj}"), food=food, padj=padj),
        editor.template(pf(e2, "The {food} wasn't {nadj}"), food=food, nadj=nadj),
    ]
    labels_1 = [0, 1]
    templates_2 = [
        editor.template(
            pf(e1, "I did not think that the {food} was {padj}"), food=food, padj=padj
        ),
        editor.template(
            pf(e2, "I did not think that the {food} was {nadj}"), food=food, nadj=nadj
        ),
    ]
    labels_2 = [0, 1]
    templates_3 = [
        editor.template(
            pf(e1, "The {food} was not {padj}, in my opinion"), food=food, padj=padj
        ),
        editor.template(
            pf(e2, "The {food} was not {nadj}, in my opinion"), food=food, nadj=nadj
        ),
    ]
    labels_3 = [0, 1]

    return {
        "d1": get_metadata(templates_1, labels_1),
        "d2": get_metadata(templates_2, labels_2),
        "d3": get_metadata(templates_3, labels_3),
    }, patches


def knowledge_checklists_flips(
    abstraction=False,
):
    # performing well here indicates that the model isn't just copying
    padj = ["numf", "weref", "wety"]
    nadj = ["wuex", "muxy", "wegry"]
    patches = [
        "If food is described as {}, then food is good".format(adj) for adj in padj
    ] + ["If food is described as {}, then food is bad".format(adj) for adj in nadj]
    if abstraction:
        food = ["steak", "tacos", "pizza", "pasta", "oysters", "filet mignon"]
    else:
        food = ["food"]

    e1 = ""
    e2 = ""
    # e1 = '{padj} is a good word'
    # e2 = '{nadj} is a bad word'

    templates_1 = [
        editor.template(
            pf(e1, "The {food} was {padj}, but everything else was really {o_nadj}"),
            padj=padj,
            food=food,
            o_nadj=["bad", "poor", "pathetic"],
        ),
        editor.template(
            pf(e2, "The {food} was {nadj}, but everything else was really {o_padj}"),
            nadj=nadj,
            food=food,
            o_padj=["amazing", "wonderful"],
        ),
    ]
    labels_1 = [
        0,
        1,
    ]  # not 0 and 1 but the model preds for food was good, but everything else was really

    templates_2 = [
        editor.template(
            pf(
                e1,
                "Unfortunately everything else was really {o_nadj} even though the {food} was {padj}",
            ),
            food=food,
            padj=padj,
            o_nadj=["bad", "poor", "pathetic"],
        ),
        editor.template(
            pf(
                e2,
                "Fortunately, everything else was really {o_padj} even though the {food} was {nadj}",
            ),
            food=food,
            nadj=nadj,
            o_padj=["amazing", "wonderful"],
        ),
    ]
    labels_2 = [0, 1]

    return {
        "d1": get_metadata(templates_1, labels_1),
        "d2": get_metadata(templates_2, labels_2),
    }, patches


def knowledge_checklists(
    abstraction=False,
    use_irrelevant=False,
):
    easy_labels = [1, 0, 1, 0]
    padj = ["the bomb", "the shizz"]
    nadj = ["unusual", "strange"]
    s_padj = padj
    s_nadj = nadj
    o_padj = ["good", "decent"]
    o_nadj = ["bad"]

    if abstraction:
        food = ["steak", "tacos", "pizza", "pasta", "oysters", "filet mignon"]
        service = ["bartender", "server", "barista", "host"]
    else:
        service = ["service"]
        food = ["food"]

    patches = [f"if food is described as {adj}, then food is good" for adj in padj]
    patches += [f"if food is described as {adj}, then food is bad" for adj in nadj]
    patches += [
        f"if service is described as {adj}, then service is good" for adj in padj
    ]
    patches += [
        f"if service is described as {adj}, then service is bad" for adj in nadj
    ]

    e1 = e2 = e3 = e4 = ""

    simple_templates = [
        editor.template(
            pf(e1, "The restaurant has {padj} {food}"), food=food, padj=padj
        ),
        editor.template(
            pf(e2, "The restaurant has {nadj} {food}"), food=food, nadj=nadj
        ),
    ]

    irrelevant_simple = [
        editor.template(
            pf(e3, "The restaurant has a {padj} {service}"),
            service=service,
            padj=s_padj,
        ),
        editor.template(
            pf(e4, "The restaurant has a {nadj} {service}"),
            service=service,
            nadj=s_nadj,
        ),
    ]

    compound_templates = [
        editor.template(
            pf(
                e1, "The restaurant has a {nadj} {service} but {food} was really {padj}"
            ),
            nadj=o_nadj,
            padj=padj,
            food=food,
            service=service,
        ),
        editor.template(
            pf(
                e2, "The restaurant has a {padj} {service} but {food} was really {nadj}"
            ),
            padj=o_padj,
            nadj=nadj,
            food=food,
            service=service,
        ),
    ]

    irrelevant_compound = [
        editor.template(
            pf(
                e3,
                "The restaurant has {nadj} {food} but the {service} was really {padj}",
            ),
            nadj=o_nadj,
            padj=s_padj,
            food=food,
            service=service,
        ),
        editor.template(
            pf(
                e4,
                "The restaurant has {padj} {food} but the {service} was really {nadj}",
            ),
            padj=o_padj,
            nadj=s_nadj,
            food=food,
            service=service,
        ),
    ]

    if use_irrelevant:
        return {
            "simple": get_metadata(irrelevant_simple, easy_labels),
            "compound": get_metadata(irrelevant_compound, easy_labels),
        }, patches
    else:
        return {
            "simple": get_metadata(simple_templates, easy_labels),
            "compound": get_metadata(compound_templates, easy_labels),
        }, patches


def subset(metadata, idxs):
    metadata_subset = {}
    for key in metadata:
        metadata_subset[key] = [metadata[key][idx] for idx in idxs]
    return metadata_subset


def deconstruct(explanation_and_instance_list):
    explanations = []
    instances = []
    for eandi in explanation_and_instance_list:
        if len(eandi.split(".")) > 1:
            explanation = eandi.split(".")[0].split(":")[-1].strip()
            instance = eandi.split(".")[1].split(":")[-1].strip()
        else:
            instance = eandi.split(":")[-1].strip()
            explanation = ""
        explanations.append(explanation)
        instances.append(instance)

    return instances, explanations


def get_metadata(all_data, all_labels):
    labels = [
        label for label, template in zip(all_labels, all_data) for _ in template["data"]
    ]
    try:
        all_instances, all_explanations = deconstruct(
            [ex for template in all_data for ex in template["data"]]
        )
    except:
        all_instances = [ex for template in all_data for ex in template["data"]]
        all_explanations = ["" for _ in all_instances]
    return {
        "instances": all_instances,
        "explanations": all_explanations,
        "labels": labels,
    }
