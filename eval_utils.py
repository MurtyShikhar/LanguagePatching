# load in the tokenizer
import os, sys
from torch.nn import functional as F
import numpy as np
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from munch import Munch

from transformers import T5ForConditionalGeneration, AutoTokenizer
from model import T5ForConditionalGenerationMultipleHeads, T5Interpeter
import torch
import itertools
from helpers import convert_to_tensors, prompt_styles, verbalize_examples
from transformers.data.data_collator import DataCollatorWithPadding
from patch_dataset import SimpleDataset

from training_utils import train_loop_fixed_steps


def apply_patch_soft(patch_applies_probs, baseline_probs, conditioned_probs):
    applies_prob = patch_applies_probs[:, 1].reshape(-1, 1)
    return (applies_prob * conditioned_probs) + (1 - applies_prob) * baseline_probs


def dissect(patch):
    cond, consequence = patch.split(",")
    cond = " ".join(cond.split(" ")[1:])
    consequence = " ".join(consequence.split(" ")[2:])
    print(cond, consequence)
    return cond, consequence


def get_scores_multiple_patches_hard(model_obj, data, patch_list, silent=False):
    no_exps = [("", ex) for ex in data[0]]
    no_exp_probs = predict_stuff(
        no_exps,
        [0] * len(no_exps),
        model_obj,
        "p1",
        verbose=False,
        mode="task_predictor",
    )
    if not silent:
        print(np.mean(no_exp_probs.argmax(axis=1) == data[1]))
    cond_probs = []
    all_patched_probs = []
    for idx, patch in enumerate(patch_list):
        if patch == "":
            continue

        cond, consequence = dissect(patch)
        contextualized = [(cond, ex) for ex in data[0]]
        gating_probs = predict_stuff(
            contextualized, itertools.repeat(0), model_obj, "p1", verbose=False
        )
        cond_probs.append(np.log(gating_probs[:, 1]))  # log(p(c | x))

        conditioning_examples = [(consequence, ex) for ex in data[0]]
        conditioned_probs = predict_stuff(
            conditioning_examples,
            itertools.repeat(0),
            model_obj,
            "p1",
            verbose=True,
            mode="task_predictor",
        )

        patched_probs = apply_patch_soft(gating_probs, no_exp_probs, conditioned_probs)

        if not silent:
            print("Applying patch {}".format(cond))
        all_patched_probs.append(patched_probs[:, 1])
    # how much should each be weighted by?
    # pick best patch and apply it!
    all_patched_probs = np.stack(all_patched_probs, axis=1)  # D x P
    cond_probs = np.stack(cond_probs, axis=1)  # D x P
    best_patches = np.argmax(cond_probs, axis=1)  # D x l

    ptrue = np.array([p[idx] for p, idx in zip(all_patched_probs, best_patches)])
    pfalse = 1.0 - ptrue
    return no_exp_probs, np.stack([pfalse, ptrue]).T


def get_data(tuple_list, tokenizer, prompt_style="p1"):
    inputs, labels = tuple_list
    prompt_func = prompt_styles[prompt_style]
    verbalizer_label = {0: "negative", 1: "positive"}
    all_data = []
    for inp, label in zip(inputs, labels):
        ex = prompt_func("", inp)
        all_data.append((ex, verbalizer_label[label]))

    return SimpleDataset(all_data, tokenizer, as_lm=True)


def fewshot_finetune(path_name, update_steps, train_tuple_list, val_tuple_list, metric):
    # load the model in
    model_obj = load_model(path_name)
    train_data = get_data(train_tuple_list, model_obj.tokenizer)

    if type(val_tuple_list) == dict:
        val_data = {
            key: get_data(_val, model_obj.tokenizer)
            for key, _val in val_tuple_list.items()
        }
    else:
        val_data = get_data(val_tuple_list, model_obj.tokenizer)

    # TODO: figure out a way to get the config.
    cfg = Munch(
        num_warmup_steps=0,
        lr=1e-4,
        train_batch_size=4,
        accum_steps=4,
        eval_batch_size=256,
    )
    return train_loop_fixed_steps(
        model_obj, cfg, {"task_data": train_data}, val_data, update_steps, metric
    )


def load_model(path_name, primary_mode="task_predictor", device_idx=0):
    if "t5" in path_name:
        tokenizer = AutoTokenizer.from_pretrained("t5-large")
        try:
            base_model = T5ForConditionalGenerationMultipleHeads.from_pretrained(
                "t5-large"
            )
            model_obj = T5Interpeter(
                base_model, tokenizer, primary_mode=primary_mode, train_multihead=True
            )
            # don't set strict to true here because we want all keys to match!
            model_obj.load_state_dict(torch.load(path_name, map_location="cpu"))
        except RuntimeError:
            print("only loading base model!")
            base_model = T5ForConditionalGeneration.from_pretrained("t5-large")
            base_model.load_state_dict(
                torch.load(path_name, map_location="cpu"), strict="False"
            )
            model_obj = T5Interpeter(
                base_model, tokenizer, primary_mode=primary_mode, train_multihead=False
            )
        # except:
        #    print("loading base model with multiple heads")
        #    base_model = T5ForConditionalGenerationMultipleHeads.from_pretrained('t5-large')
        #    model_obj = T5Interpeter(base_model, tokenizer, primary_mode=primary_mode, train_multihead=False)
        #    model_obj.load_state_dict(torch.load(path_name, map_location='cpu'))
    else:
        print("model not supported")
        sys.exit(1)

    if torch.cuda.is_available():
        if device_idx:
            device = torch.device("cuda:{}".format(device_idx))
        else:
            device = torch.device("cuda")
        model_obj.to(device)
    else:
        print("No cuda!!")
    model_obj.eval()
    return model_obj


def predict_stuff_helper(
    model,
    dataset,
    verbose,
    interchange=True,
    data_collator_to_use=None,
    batch_size=64,
    mode=None,
    ret_result=False,
):
    if data_collator_to_use is None:
        tokenizer = model.tokenizer
        data_collator_to_use = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size,
        collate_fn=data_collator_to_use,
    )
    result = model.evaluator(dataloader, verbose=verbose, mode=mode)
    if ret_result:
        return result
    else:
        pp = F.softmax(result["logits"], dim=1).numpy()
        if interchange:
            pp = np.hstack((pp[:, 1:], pp[:, 0:1]))
        return pp


def predict_stuff(
    examples,
    labels,
    model,
    prompt_style,
    verbose=False,
    interchange=True,
    verbalize=True,
    batch_size=64,
    mode=None,
):
    prompt_func = prompt_styles[prompt_style]
    tokenizer = model.tokenizer
    examples = [x if type(x) == str else prompt_func(x[0], x[1]) for x in examples]
    if verbalize:
        verbalized_examples = verbalize_examples(
            [(x, label) for (x, label) in zip(examples, labels)],
            prompt_style,
            labels_given=True,
        )
    else:
        verbalized_examples = [(x, label) for (x, label) in zip(examples, labels)]
    if verbose:
        print(verbalized_examples[0])
    test_dataset = convert_to_tensors(verbalized_examples, tokenizer)
    return predict_stuff_helper(
        model,
        test_dataset,
        verbose,
        interchange=interchange,
        batch_size=batch_size,
        mode=mode,
    )


def get_predictions(patches, inputs, model_dict, prompt_style=None):
    model2preds = {}
    for model_name in model_dict:
        preds = {}
        try:
            model_obj = load_model(model_dict[model_name], None)
        except:
            continue
        if not prompt_style:
            if "p2" in model_name:
                prompt_style = "p2"
            else:
                prompt_style = "p1"

        for patch in patches:
            if len(patch) == 0:
                input_examples = inputs
            else:
                input_examples = [(patch, cinput) for cinput in inputs]
            preds[patch] = predict_stuff(
                input_examples, [0] * len(inputs), model_obj, prompt_style
            )
        model2preds[model_name] = preds
    return model2preds
