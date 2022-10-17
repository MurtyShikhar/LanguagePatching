import torch
from tqdm import tqdm
import os
import wandb
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from hydra.utils import get_original_cwd

from transformers.data.data_collator import DataCollatorWithPadding
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)

def working_dir():
    USER = os.environ["USER"]
    dir_name = f'/scr/biggest'
    if os.path.exists(dir_name):
        sub_dir = '{}/{}/models'.format(dir_name, USER)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        return sub_dir
    else:
        try:
            return get_original_cwd()
        except:
            return ''

def construct_collate_exp_finetune(tokenizer, neg_only=False):
    def collate_fn(feature_list):
        process_fn = lambda key: [{'input_ids': ex['input_ids_{}'.format(key)], 'attention_mask': ex['attention_mask_{}'.format(key)]} for ex in feature_list]
        ret_dict = {}
        if not neg_only:
            b1 = tokenizer.pad(process_fn('with_correct_exp'), padding=True, max_length=None, pad_to_multiple_of=None, return_tensors='pt')
            for key, val in b1.items():
                ret_dict[key] = val
            ret_dict['labels'] = torch.tensor([ex['labels'] for ex in feature_list])
            ret_dict['probs'] = torch.tensor([ex['prob'] for ex in feature_list])
        b2 = tokenizer.pad(process_fn('with_incorrect_exp'), padding=True, max_length=None, pad_to_multiple_of=None, return_tensors='pt')
        b3 = tokenizer.pad(process_fn('no_exp'), padding=True, max_length=None, pad_to_multiple_of=None, return_tensors='pt')
        # exp vs explanation inconsistency... 
        for key, val in b2.items(): 
            ret_dict['{}_with_incorrect_explanation'.format(key)] = val
        for key, val in b3.items():
            ret_dict['{}_no_exp'.format(key)] = val
        return ret_dict
    return collate_fn
    

def construct_collate(tokenizer):
    def collate_fn(feature_list):
        # process 
        process_fn = lambda key: [{'input_ids': ex['input_ids_{}'.format(key)], 'attention_mask': ex['attention_mask_{}'.format(key)]} for ex in feature_list]
        b1 = tokenizer.pad(process_fn('without_explanation'), padding=True, max_length=None, pad_to_multiple_of=None, return_tensors='pt')
        b2 = tokenizer.pad(process_fn('with_explanation'), padding=True, max_length=None, pad_to_multiple_of=None, return_tensors='pt')
        ret_dict = {}
        for key, val in b1.items():
            ret_dict['{}_without_explanation'.format(key)] = val
        for key, val in b2.items(): 
            ret_dict['{}_with_explanation'.format(key)] = val
        labels = torch.tensor([ex['labels'] for ex in feature_list])
        exp_applies = torch.tensor([ex['exp_applies'] for ex in feature_list])
        is_gold = torch.tensor([ex['gold'] for ex in feature_list])
        ret_dict['labels'] = labels
        ret_dict['exp_applies'] = exp_applies
        ret_dict['is_gold'] = is_gold
        return ret_dict

    return collate_fn

def get_opt(cfg, model):
    no_decay = ["bias", "LayerNorm.weight"]
    weight_decay = cfg.get("weight_decay", 0.0)
    adam_epsilon = cfg.get("adam_epsilon", 1e-7)
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=cfg.get("lr", 1e-4),
        eps=adam_epsilon,
    )
    return optimizer


def get_scheduler(cfg, opt, t_total):
    num_warmup_steps = cfg.get("num_warmup_steps", 500)
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )
    return scheduler


def eval_func(cfg, model, val_data_dict, collator, log, best_metric=None, metric='acc'):
    model.eval()
    to_log = {}

    if type(val_data_dict) == dict:
        for key, val_data in val_data_dict.items():
            val_data_curr = val_data.get_data(max_size = 30000)
            validation = DataLoader(
                val_data_curr,
                sampler=SequentialSampler(val_data_curr),
                batch_size=cfg.eval_batch_size,
                collate_fn=collator
            )

            if key == 'exp_grounding_data':
                result = model.evaluator(validation, mode='exp_applies_predictor')
            else:
                result = model.evaluator(validation)
            to_log['{}_f1'.format(key)] = result['f1']
            to_log['{}_acc'.format(key)] = result['acc']
    else:
        val_data_curr = val_data_dict.get_data(max_size=30000)
        validation = DataLoader(
            val_data_curr,
            sampler=SequentialSampler(val_data_curr),
            batch_size=cfg.eval_batch_size,
            collate_fn=collator
        )
        result = model.evaluator(validation)
        to_log['f1'] = result['f1']
        to_log['acc'] = result['acc']

    orig_working_dir = working_dir()

    if log:
        result_str = " ".join(["\n{}: {:.2f}".format(key, val) for key, val in to_log.items()])
        log.info(result_str)
    try:
        wandb.log(to_log)
    except:
        pass

    if best_metric is None:
        if metric in to_log:
            return to_log[metric]
        else:
            # if val_data is a dict
            keys = [key for key in to_log if metric in key]
            return [to_log[key] for key in keys]
        
    elif to_log[metric] > best_metric:
        best_metric = to_log[metric]
        log.info(
            "Saving model at {}".format(os.path.join(orig_working_dir, cfg.save_path))
        )
        torch.save(
            model.state_dict(),
            "{}/{}".format(orig_working_dir, cfg.save_path),
        )
    return best_metric


def train_loop_fixed_steps(model, cfg, train_data_dict, val_data, t_total, metric):
    accum_steps = cfg.get("accum_steps", 1)
    opt = get_opt(cfg, model)
    scheduler = get_scheduler(cfg, opt, t_total)
    num_steps = 0

    tokenizer = model.tokenizer
    train_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    val_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    # number of total
    pbar = tqdm(total = t_total)
    while num_steps < t_total:
        train_dataloaders = {}
        total_train_sz = []
        for key, train_data in train_data_dict.items():
            train_data_curr = train_data.get_data()
            total_train_sz.append(len(train_data_curr))
            train = DataLoader(
                train_data_curr,
                sampler=RandomSampler(train_data_curr),
                batch_size=cfg.train_batch_size,
                collate_fn=train_data_collator
            )
            train_dataloaders[key] = train
        with torch.enable_grad():
            losses = []
            all_keys = list(train_dataloaders.keys())
            for all_batches in zip(*train_dataloaders.values()):
                curr_batch_dict = dict(zip(all_keys, all_batches))
                model.train()
                loss_curr = model.get_loss(curr_batch_dict)
                loss_curr /= accum_steps
                loss_curr.backward()
                losses.append(loss_curr.item())
                if len(losses) == accum_steps:
                    num_steps += 1
                    pbar.update(1)
                    opt.step()
                    scheduler.step()
                    model.zero_grad()
                    losses = []
            
                if num_steps == t_total:
                    break
            if losses:
                num_steps +=1 
                pbar.update(1)
                opt.step()
                scheduler.step()
                model.zero_grad()
                losses = []
            # Evaluate on this epoch
    pbar.close()

    #print("Evaluating on Train Data")
    #train_data_curr = train_data_dict['task_data'].get_data()
    #train = DataLoader(
    #    train_data_curr,
    #    sampler=SequentialSampler(train_data_curr),
    #    batch_size=cfg.eval_batch_size,
    #    collate_fn=train_data_collator
    #)
    #final_train_acc =  eval_func(cfg, model, train_data_dict['task_data'], train_data_collator, None)
    #print("Final Train Acc: {}".format(final_train_acc))


    print("Evaluating on Test Data.")
    return eval_func(cfg, model, val_data, val_data_collator, None, metric=metric)
    


def train_loop(model, log, cfg, train_data_dict, val_data, metric='acc'):
    num_epochs = cfg.num_epochs
    accum_steps = cfg.get("accum_steps", 1)
    eval_every = cfg.get('eval_every', None)
    max_grad_norm = cfg.get('max_grad_norm', 5)
    opt = get_opt(cfg, model)
    t_total = num_epochs * (min(len(train_data_dict[key]) for key in train_data_dict) // accum_steps * cfg.train_batch_size)
    scheduler = get_scheduler(cfg, opt, t_total)
    num_steps = 0
    best_acc = 0
    orig_working_dir = working_dir()

    tokenizer = model.tokenizer
    train_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    val_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # evaluate once at the beginning to see if evaluation pipeline is A-ok
    for epoch in range(num_epochs):
        # Evaluate on this epoch
        train_dataloaders = {}
        total_train_sz = []
        for key, train_data in train_data_dict.items():
            train_data_curr = train_data.get_data()
            total_train_sz.append(len(train_data_curr))
            train = DataLoader(
                train_data_curr,
                sampler=RandomSampler(train_data_curr),
                batch_size=cfg.train_batch_size,
                collate_fn=train_data_collator
            )
            train_dataloaders[key] = train

        log.info("Epoch: {}".format(epoch))
        with torch.enable_grad(), tqdm(total=min(total_train_sz)) as progress_bar:
            # Train on this epoch
            # TODO: implement an eval-every
            losses = []
            all_keys = list(train_dataloaders.keys())
            canon_key = all_keys[0]
            for all_batches in zip(*train_dataloaders.values()):
                curr_batch_dict = dict(zip(all_keys, all_batches))
                model.train()
                loss_curr = model.get_loss(curr_batch_dict)
                progress_bar.update(len(curr_batch_dict[canon_key]['input_ids']))
                loss_curr /= accum_steps
                loss_curr.backward()
                losses.append(loss_curr.item())
                if len(losses) == accum_steps:
                    num_steps += 1
                    progress_bar.set_postfix({"loss": sum(losses) / len(losses), "num_steps": num_steps})
                    opt.step()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scheduler.step()
                    model.zero_grad()
                    losses = []
                    if eval_every and num_steps % eval_every == 0:
                        log.info("Evaluating at step {}".format(num_steps))
                        best_acc = eval_func(cfg, model, val_data, val_data_collator, log, best_acc, metric)
        # evaluate at the end of the epoch.
        if not eval_every:
            log.info("Evaluating at step {}".format(num_steps))
            best_acc = eval_func(cfg, model, val_data, val_data_collator, log, best_acc, metric)
    return
