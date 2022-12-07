# Entry script: Train language models so that they can be patched with natural language at test time

import random
import os
import torch
import logging
import numpy as np
from transformers import AutoTokenizer

import json
from training_utils import train_loop

# for managing experiments
import hydra
from hydra.utils import get_original_cwd


from model import T5Interpeter, T5ForConditionalGenerationMultipleHeads
from build_datasets import (
    get_data_task_finetuning,
    get_data_patch_finetuning_sentiment,
    get_data_patch_finetuning_re,
)

import wandb

### SET THIS AS YOUR OWN PROJECT!
wandb.init(project="patches", entity="shikharmurty")


@hydra.main(config_path="config", config_name="task-finetuning")
def main(cfg):
    log = logging.getLogger(__name__)
    # save the config to wandb run
    wandb.config = cfg
    # make the model name descriptive enough so it doubles as a run name
    wandb.run.name = cfg.train.save_path
    wandb.run.save()
    # set seed
    random.seed(cfg.get("seed", 42))
    orig_working_dir = get_original_cwd()

    model_type = cfg.model_type
    model = T5ForConditionalGenerationMultipleHeads.from_pretrained(model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    data_protocol = cfg.get("protocol", "simple")
    if "patch" in data_protocol and not cfg.get("learnt_interpreter", False):
        primary_mode = "patch_applies_predictor"
    else:
        primary_mode = "task_predictor"
    if cfg.train.get("load_path", None):
        load_path = cfg.train.load_path
        log.info("loading a checkpoint from {}".format(load_path))
        try:
            model.load_state_dict(torch.load(os.path.join(orig_working_dir, load_path)))
            model_obj = T5Interpeter(
                model, tokenizer, primary_mode=primary_mode, train_multihead=True
            )
        except:
            model_obj = T5Interpeter(
                model, tokenizer, primary_mode=primary_mode, train_multihead=True
            )
            model_obj.load_state_dict(
                torch.load(os.path.join(orig_working_dir, load_path)), strict=False
            )

    else:
        model_obj = T5Interpeter(
            model, tokenizer, primary_mode=primary_mode, train_multihead=True
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model_obj.to(device)

    ### Get data for Task Finetuning stage
    if data_protocol == "simple":
        train_data, val_data = get_data_task_finetuning(cfg, tokenizer)
    ### Get data for Patch Finetuning stage
    elif data_protocol == "patch_finetune_conds":
        train_data, val_data = get_data_patch_finetuning_sentiment(cfg, tokenizer)
    elif data_protocol == "patch_re":
        train_data, val_data = get_data_patch_finetuning_re(cfg, tokenizer)

    wandb.watch(model_obj)
    if cfg.data == "spouse_re" or data_protocol == "patch_re":
        metric = "f1"
    elif cfg.get("learnt_interpreter", False):
        metric = "task_data_patch_acc"
    else:
        metric = "acc"
    train_loop(model_obj, log, cfg.train, train_data, val_data, metric)


if __name__ == "__main__":
    main()
