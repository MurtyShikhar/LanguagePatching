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

from transformers import GPT2LMHeadModel, T5ForConditionalGeneration,  AutoTokenizer
from model import T5ForConditionalGenerationMultipleHeads, T5Interpeter 
import torch
import itertools
from helpers import convert_to_tensors, prompt_styles, verbalize_examples
from transformers.data.data_collator import DataCollatorWithPadding
from explanation_dataset import SimpleDataset

from training_utils import train_loop_fixed_steps


def apply_patch_soft(exp_applies_probs, baseline_probs, conditioned_probs):    
    applies_prob = exp_applies_probs[:, 1].reshape(-1, 1)
    return (applies_prob * conditioned_probs) + (1 - applies_prob) * baseline_probs
    
def dissect(patch):
    cond, consequence = patch.split(',')
    cond = ' '.join(cond.split(' ')[1:])
    consequence = ' '.join(consequence.split(' ')[2:])
    print(cond, consequence)
    return cond, consequence


def get_scores_multiple_patches_hard(model_obj, data, patch_list, examine=False, silent=False):
    no_exps = [('', ex) for ex in data[0]]
    no_exp_probs = predict_stuff(no_exps, [0]*len(no_exps), model_obj, 'p1', verbose=False, mode='task_predictor')
    if not silent:
        print(np.mean(no_exp_probs.argmax(axis=1)==data[1]))
    cond_probs = []
    interpret_probs = []    
    all_patched_probs = []
    for idx, patch in enumerate(patch_list):
        if patch == '':
            continue
                    
        cond, consequence = dissect(patch)
        contextualized = [(cond, ex) for ex in data[0]]
        gating_probs = predict_stuff(contextualized, itertools.repeat(0), model_obj, 'p1', verbose=False)
        cond_probs.append(np.log(gating_probs[:, 1])) # log(p(c | x))
        

        conditioning_examples = [(consequence, ex) for ex in data[0]]
        conditioned_probs = predict_stuff(conditioning_examples, itertools.repeat(0), 
                                          model_obj, 'p1', verbose=True, mode='task_predictor')


        patched_probs = apply_patch_soft(gating_probs, no_exp_probs, conditioned_probs)

        if not silent:
            print("Applying patch {}".format(cond))
            print(np.mean(patched_probs.argmax(axis=1) == data[1]))
        all_patched_probs.append(patched_probs[:, 1])
    # how much should each be weighted by?
    # pick best patch and apply it! 
    all_patched_probs = np.stack(all_patched_probs, axis=1) # D x P
    cond_probs = np.stack(cond_probs, axis=1) # D x P
    best_patches = np.argmax(cond_probs, axis=1) # D x l
    
    ptrue = np.array([p[idx] for p, idx in zip(all_patched_probs, best_patches)])
    pfalse = 1.0 - ptrue
    return no_exp_probs, np.stack([pfalse, ptrue]).T



def get_data(tuple_list, tokenizer, prompt_style='p1'):
    inputs, labels = tuple_list
    prompt_func = prompt_styles[prompt_style]
    verbalizer_label = {0: 'negative', 1: 'positive'}
    all_data = []
    for inp, label in zip(inputs, labels):
        ex = prompt_func('', inp)
        all_data.append((ex, verbalizer_label[label]))

    return SimpleDataset(all_data, tokenizer, as_lm=True)

def fewshot_finetune(path_name, update_steps, train_tuple_list, val_tuple_list, metric):
    # load the model in
    model_obj = load_model(path_name)
    train_data = get_data(train_tuple_list, model_obj.tokenizer)

    if type(val_tuple_list) == dict:
        val_data = {key: get_data(_val, model_obj.tokenizer) for key, _val in val_tuple_list.items()}
    else:
        val_data = get_data(val_tuple_list, model_obj.tokenizer)
    
    # TODO: figure out a way to get the config.
    cfg = Munch(num_warmup_steps=0, lr=1e-4, train_batch_size=4, accum_steps=4, eval_batch_size=256)
    return train_loop_fixed_steps(model_obj, cfg, {'task_data': train_data}, val_data, update_steps, metric)


def load_model(path_name, primary_mode='task_predictor', device_idx=0):
    if 't5' in path_name:
        tokenizer = AutoTokenizer.from_pretrained('t5-large')
        try:
            base_model = T5ForConditionalGenerationMultipleHeads.from_pretrained('t5-large')
            model_obj = T5Interpeter(base_model, tokenizer, primary_mode=primary_mode, train_multihead=True)
            # don't set strict to true here because we want all keys to match!
            model_obj.load_state_dict(torch.load(path_name, map_location='cpu'))
        except RuntimeError:
           print("only loading base model!")
           base_model = T5ForConditionalGeneration.from_pretrained('t5-large')
           base_model.load_state_dict(torch.load(path_name, map_location='cpu'), strict='False')
           model_obj = T5Interpeter(base_model, tokenizer, primary_mode=primary_mode, train_multihead=False)
        #except:
        #    print("loading base model with multiple heads")
        #    base_model = T5ForConditionalGenerationMultipleHeads.from_pretrained('t5-large')
        #    model_obj = T5Interpeter(base_model, tokenizer, primary_mode=primary_mode, train_multihead=False)
        #    model_obj.load_state_dict(torch.load(path_name, map_location='cpu'))
    else:
        print("model not supported")
        sys.exit(1)

    if torch.cuda.is_available():
        if device_idx:
            device = torch.device('cuda:{}'.format(device_idx))
        else:
            device = torch.device('cuda')
        model_obj.to(device)
    else:
        print('No cuda!!')
    model_obj.eval();
    return model_obj


def predict_stuff_helper(model, dataset, verbose, 
                         interchange=True, data_collator_to_use=None, 
                         batch_size=64, mode=None, ret_result=False):
    if data_collator_to_use is None:
        tokenizer = model.tokenizer
        data_collator_to_use = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size, collate_fn=data_collator_to_use)
    result = model.evaluator(dataloader, verbose=verbose, mode=mode)
    if ret_result:
        return result
    else:
        pp = F.softmax(result['logits'], dim=1).numpy()
        if interchange:
            pp = np.hstack((pp[:, 1:], pp[:, 0:1]))
        return pp


def predict_stuff(examples, labels, model, prompt_style, 
                  verbose=False, interchange=True, 
                  verbalize=True, batch_size=64, mode=None):
    prompt_func = prompt_styles[prompt_style]
    tokenizer = model.tokenizer
    examples = [x if type(x) == str else prompt_func(x[0], x[1]) for x in examples]
    if verbalize:
        verbalized_examples = verbalize_examples([(x, label) for (x, label) in zip(examples, labels)], prompt_style, labels_given=True)
    else:
        verbalized_examples = [(x, label) for (x, label) in zip(examples, labels)]
    if verbose:
        print(verbalized_examples[0])
    test_dataset = convert_to_tensors(verbalized_examples, tokenizer)
    return predict_stuff_helper(model, test_dataset, verbose, 
                                interchange=interchange, batch_size=batch_size, mode=mode)

def get_predictions(explanations, inputs, model_dict, prompt_style=None):
    model2preds = {}
    for model_name in model_dict:
        preds = {}
        try:
            model_obj = load_model(model_dict[model_name], None)
        except:
            continue
        if not prompt_style:
            if 'p2' in model_name:
                prompt_style='p2'
            else:
                prompt_style='p1'

        for explanation in explanations:
            if len(explanation) == 0:
                input_examples = inputs
            else:
                input_examples = [(explanation, cinput) for cinput in inputs]
            preds[explanation] = predict_stuff(input_examples, [0]*len(inputs), model_obj, prompt_style)
        model2preds[model_name] = preds
    return model2preds   


# ==== Functions for generating data for evaluating knowledge explanations ===== 
from collections import defaultdict as ddict 
import json 
from explanation_dataset import ExplanationDataset_2
from helpers import prompt_styles
from checklist.editor import Editor
editor = Editor()
pf = prompt_styles['p1']

# ==== Helpers for evaluating =====
def get_acc(preds, labels):
    acc = 0.0
    total = 0.0
    for pred, label in zip(preds, labels):
        if label == -1:
            continue
        else:
            acc += (pred == label)
            total += 1
    if total == 0:
        return -1.0
    else:
        return (acc / total, total)

def get_predictions_with_exp(model, dataset, batch_size):    
    preds = predict_stuff_helper(model, dataset, verbose=True, batch_size=batch_size)
    try:
        label_oidx_to_idx = {24561: 1, 31591: 0}
        return preds, [label_oidx_to_idx[label[0]] for label in dataset['label']]
    except:
        # T5. 
        label_oidx_to_idx = {1465: 1, 2841: 0}
        return preds, [label_oidx_to_idx[label[0]] for label in dataset['label']]

# === Some helpers ==== 
def deconstruct(explanation_and_instance_list):
    explanations = []
    instances = []
    for eandi in explanation_and_instance_list:
        if len(eandi.split('.')) > 1:
            explanation = eandi.split('.')[0].split(':')[-1].strip()
            instance = eandi.split('.')[1].split(':')[-1].strip()
        else:
            instance = eandi.split(':')[-1].strip()
            explanation = ''
        explanations.append(explanation)
        instances.append(instance)
    
    return instances, explanations


def get_metadata(all_data, all_labels):
    labels = [label for label, template in zip(all_labels, all_data) for _ in template['data']]
    try:
        all_instances, all_explanations = deconstruct([ex for template in all_data for ex in template['data']]) 
    except:
        all_instances = [ex for template in all_data for ex in template['data']]
        all_explanations = ['' for _ in all_instances]
    return {'instances': all_instances, 'explanations': all_explanations, 'labels': labels}


def build_exp2examples_dict(all_texts, all_labels, all_explanations):
    exp2examples_dict = ddict(list)
    for idx, explanation in enumerate(all_explanations):
        if len(explanation) > 0:
            explanation = explanation[:-1].rstrip() if explanation[-1] == '.' else explanation.rstrip()
        for exp in explanation.split("."):
            exp = exp.strip()
            exp2examples_dict[exp].append(idx)
    return exp2examples_dict


def get_applies_metadata(metadata):
    print('getting applies metadata')
    def get_cond(explanation):
        return ' '.join(explanation.split(',')[0].split(' ')[1:])

    new_data = {key: [] for key in metadata}
    from distractor_lib import DEFAULT_EXP
    for exp, instance, label in zip(metadata['explanations'], metadata['instances'], metadata['labels']):
        new_data['labels'].append(1)
        cond = get_cond(exp) if 'if' in exp else exp
        new_data['instances'].append(instance)
        new_data['explanations'].append(cond)

        new_data['labels'].append(0) # this is because we give all negative explanations the label of the noop...
        new_data['instances'].append(instance)
        new_data['explanations'].append(DEFAULT_EXP)
    return new_data


def get_dataset(metadata, tokenizer, distractor_fn='default', postprocess=True, with_explanations=True, measure_applies_acc=False):
    if with_explanations:
        if measure_applies_acc:
            metadata = get_applies_metadata(metadata)
        exp2examples_dict = build_exp2examples_dict(metadata['instances'], metadata['labels'], metadata['explanations'])
        dataset = ExplanationDataset_2(exp2examples_dict, metadata['instances'], metadata['labels'], 
                                       tokenizer, use_distractors=False)
        hf_data = dataset.get_data(verbose=False, postprocess=postprocess)
        return hf_data
    else:
        return {'instances': metadata['instances'], 'labels':  metadata['labels']}



# TODO: add a flag to be able to do abstractions!
def knowledge_checklists_type_2(tokenizer, with_explanations=True, measure_applies_acc=False, metadata_only=False, abstraction=False):
    # doing well here means the model can incorporate the explanation in some context that doesn't negate.
    #padj = ['seeet', 'bgesx', 'weref']
    #nadj = ['wuex', 'sercx', 'wety']

    padj = ['numf', 'weref', 'wety']
    nadj = ['wuex', 'muxy', 'wegry']
    if abstraction:
        food = ['steak', 'tacos', 'pizza', 'pasta', 'oysters', 'filet mignon']
    else:
        food = ['food']


    patches = ['If food is described as {}, then food is good'.format(adj) for adj in padj] + ['If food is described as {}, then food is bad'.format(adj) for adj in nadj]

    e1 = ''
    e2 = ''
    #e1 = '{padj} is a good word'
    #e2 = '{nadj} is a bad word'


    templates_1 = [editor.template(pf(e2, 'The {food} was {nadj} and service was {o_nadj}'), food=food, nadj=nadj, o_nadj=['bad', 'poor', 'pathetic']),
                  editor.template(pf(e1, 'The {food} was {padj} and service was {o_padj}'), food=food, padj=padj, o_padj=['good', 'amazing', 'wonderful'])]
    labels_1 = [0, 1]

    templates_2 = [editor.template(pf(e2, 'The service was {o_nadj} and the {food} was {nadj}'), food=food, nadj=nadj, o_nadj=['bad', 'poor', 'pathetic']),
                  editor.template(pf(e1, 'The service was {o_padj} and the {food} was {padj}'), food=food, padj=padj, o_padj=['good', 'amazing', 'wonderful'])]
    labels_2 = [0, 1]

    #templates_3 = [editor.template(pf(e1, 'The {food} was {padj} and {neutral_phrase}'), food=food, padj=padj, neutral_phrase=['I met my friends there', 'I took the subway to get there', 'it was raining that day', 'I met my cousins there']),
    #              editor.template(pf(e2, 'The {food} was {nadj} and {neutral_phrase}'), food=food, nadj=nadj, neutral_phrase=['I met my friends there', 'I took the subway to get there', 'it was raining that day', 'I met my cousins there'])]
    #labels_3 = [1, 0]
    #'d3_type2': get_metadata(templates_3, labels_3)
    if metadata_only:
        return {'d1_type2': get_metadata(templates_1, labels_1), 
                'd2_type2': get_metadata(templates_2, labels_2)}, patches
    else:
        dataset_1 = get_dataset(get_metadata(templates_1, labels_1), tokenizer, with_explanations, measure_applies_acc=measure_applies_acc)
        dataset_2 = get_dataset(get_metadata(templates_2, labels_2), tokenizer, with_explanations, measure_applies_acc=measure_applies_acc)
        dataset_3 = get_dataset(get_metadata(templates_3, labels_3), tokenizer, with_explanations, measure_applies_acc=measure_applies_acc)
        return {'d1_type2': dataset_1, 'd2_type2': dataset_2, 'd3_type2': dataset_3} 


def knowledge_absn(tokenizer, with_explanations=True, measure_applies_acc=False, metadata_only=False, abstraction=False):
    negated, patches = knowledge_checklists_negating_contexts(tokenizer, with_explanations, measure_applies_acc, metadata_only, abstraction)
    # irrelevant
    irrelevant, _ = knowledge_checklists(tokenizer, with_explanations, measure_applies_acc, metadata_only, abstraction, use_irrelevant=True)

    # non predictive
    non_predictive, _ = knowledge_checklists_flips(tokenizer, with_explanations, measure_applies_acc, metadata_only, abstraction)

    all_inputs = []
    all_labels = []

    all_types = [negated, irrelevant, non_predictive]
    for ddtype in all_types:
        for k in ddtype:
            all_inputs += ddtype[k]['instances']
            all_labels += ddtype[k]['labels']

    return all_inputs, all_labels, patches

def knowledge_checklists_negating_contexts(tokenizer, with_explanations=True, measure_applies_acc=False, metadata_only=False, abstraction=False):
    # contexts that negate the meaning such as e.g. 
    # I did not think that the food was {}
    # I thought the food was not {}
    # The food was not {}, in my opinion
    # my friends thought that the food was {}, but i did not think so. 
    
    # padj = ['seeet', 'bgesx', 'weref']
    # nadj = ['wuex', 'sercx', 'wety']

    padj = ['numf', 'weref', 'wety']
    nadj = ['wuex', 'muxy', 'wegry']

    patches = ['If food is described as {}, then food is good'.format(adj) for adj in padj] + ['If food is described as {}, then food is bad'.format(adj) for adj in nadj]
    if abstraction:
        food = ['food', 'steak', 'tacos', 'pizza', 'pasta', 'oysters', 'filet mignon']
    else:
        food = ['food']


    e1 = ''
    e2 = ''

    templates_1 = [editor.template(pf(e1, "The {food} wasn't {padj}"), food=food, padj=padj),
                   editor.template(pf(e2, "The {food} wasn't {nadj}"), food=food, nadj=nadj)]
    labels_1 = [0, 1]
    templates_2 = [editor.template(pf(e1, "I did not think that the {food} was {padj}"), food=food, padj=padj),
                   editor.template(pf(e2, "I did not think that the {food} was {nadj}"), food=food, nadj=nadj)]
    labels_2 = [0, 1]
    templates_3 = [editor.template(pf(e1, "The {food} was not {padj}, in my opinion"), food=food, padj=padj),
                   editor.template(pf(e2, "The {food} was not {nadj}, in my opinion"), food=food, nadj=nadj)]
    labels_3 = [0, 1]


    if metadata_only:
        return {'d1': get_metadata(templates_1, labels_1), 
                'd2': get_metadata(templates_2, labels_2), 
                'd3': get_metadata(templates_3, labels_3)}, patches
    else:
        return None


def knowledge_checklists_flips(tokenizer, with_explanations=True, measure_applies_acc=False, metadata_only=False, abstraction=False):
    # performing well here indicates that the model isn't just copying
    padj = ['numf', 'weref', 'wety']
    nadj = ['wuex', 'muxy', 'wegry']
    patches = ['If food is described as {}, then food is good'.format(adj) for adj in padj] + ['If food is described as {}, then food is bad'.format(adj) for adj in nadj]
    if abstraction:
        food = ['steak', 'tacos', 'pizza', 'pasta', 'oysters', 'filet mignon']
    else:
        food = ['food']

    e1 = ''
    e2 = ''
    #e1 = '{padj} is a good word'
    #e2 = '{nadj} is a bad word'


    templates_1 = [editor.template(pf(e1, 'The {food} was {padj}, but everything else was really {o_nadj}'), padj=padj, food=food,  o_nadj=['bad', 'poor', 'pathetic']),
                  editor.template(pf(e2, 'The {food} was {nadj}, but everything else was really {o_padj}'), nadj=nadj, food=food, o_padj=['amazing', 'wonderful'])]
    labels_1 = [0, 1] # not 0 and 1 but the model preds for food was good, but everything else was really 

    templates_2 = [editor.template(pf(e1, 'Unfortunately everything else was really {o_nadj} even though the {food} was {padj}'), food=food, padj=padj, o_nadj=['bad', 'poor', 'pathetic']),
                  editor.template(pf(e2, 'Fortunately, everything else was really {o_padj} even though the {food} was {nadj}'), food=food, nadj=nadj, o_padj=['amazing', 'wonderful'])]
    labels_2= [0, 1]
    
    if metadata_only:
        return {'d1': get_metadata(templates_1, labels_1),
         'd2': get_metadata(templates_2, labels_2)}, patches
    else:
        dataset_1 = get_dataset(get_metadata(templates_1, labels_1), tokenizer, with_explanations, measure_applies_acc=measure_applies_acc)
        dataset_2 = get_dataset(get_metadata(templates_2, labels_2), tokenizer, with_explanations, measure_applies_acc=measure_applies_acc)
        dataset_3 = get_dataset(get_metadata(templates_3, labels_3), tokenizer, with_explanations, measure_applies_acc=measure_applies_acc)
        return {'d1': dataset_1, 'd2': dataset_2, 'd3': dataset_3}


def knowledge_checklists(tokenizer, with_explanations=True, measure_applies_acc=False, metadata_only=False, abstraction=False, use_irrelevant=False):    
    easy_labels = [1, 0, 1, 0]
    padj = ['the bomb', 'the shizz']
    nadj = ['unusual', 'strange']
    #padj = ['numf', 'weref', 'wety']
    #nadj = ['wuex', 'muxy', 'wegry']
    #padj = ['zubin', 'wug', 'shug'] 
    #nadj = ['wuf', 'numf', 'zoox']
    #nadj = padj #['wuex', 'sercx', 'wety']
    s_padj = padj #['muxy', 'wegry', 'seyl']
    s_nadj = nadj # ['saery', 'mumfy', 'grup']
    o_padj = ['good', 'decent']
    o_nadj = ['bad']

        
    if abstraction:
        food = ['steak', 'tacos', 'pizza', 'pasta', 'oysters', 'filet mignon']
        service = ['bartender', 'server', 'barista', 'host']
    else:
        service = ['service']
        food = ['food']

    patches = [f'if food is described as {adj}, then food is good' for adj in padj]
    patches += [f'if food is described as {adj}, then food is bad' for adj in nadj]
    patches += [f'if service is described as {adj}, then service is good' for adj in padj]
    patches += [f'if service is described as {adj}, then service is bad' for adj in nadj]

    
    
    e1 = e2 = e3 = e4 = ''
    
    simple_templates = [editor.template(pf(e1, 'The restaurant has {padj} {food}'), food=food, padj=padj),
                      editor.template(pf(e2, 'The restaurant has {nadj} {food}'), food=food, nadj=nadj)]

    irrelevant_simple = [editor.template(pf(e3, 'The restaurant has a {padj} {service}'), service=service, padj=s_padj),
                        editor.template(pf(e4, 'The restaurant has a {nadj} {service}'), service=service, nadj=s_nadj)]

    compound_templates = [editor.template(pf(e1, 'The restaurant has a {nadj} {service} but {food} was really {padj}'), nadj=o_nadj, padj=padj, food=food, service=service),
                      editor.template(pf(e2, 'The restaurant has a {padj} {service} but {food} was really {nadj}'), padj=o_padj, nadj=nadj, food=food, service=service)]


    irrelevant_compound = [editor.template(pf(e3, 'The restaurant has {nadj} {food} but the {service} was really {padj}'), nadj=o_nadj, padj=s_padj, food=food, service=service),
                           editor.template(pf(e4, 'The restaurant has {padj} {food} but the {service} was really {nadj}'), padj=o_padj, nadj=s_nadj, food=food, service=service)]

    

    if metadata_only:
        if use_irrelevant:
            return {'simple': get_metadata(irrelevant_simple, easy_labels),
                    'compound': get_metadata(irrelevant_compound, easy_labels)}, patches
        else:
            return {'simple': get_metadata(simple_templates, easy_labels),
                    'compound': get_metadata(compound_templates, easy_labels)}, patches


    else:
        easy_dataset = get_dataset(get_metadata(easy_templates, easy_labels), tokenizer, with_explanations, measure_applies_acc=measure_applies_acc)
        neg_easy_dataset = get_dataset(get_metadata(neg_easy_templates, neg_labels), tokenizer, with_explanations, measure_applies_acc=measure_applies_acc)
        # actually, this is arguably HARD...
        neg_hard_dataset = get_dataset(get_metadata(neg_hard_templates, neg_labels), tokenizer, with_explanations, measure_applies_acc=measure_applies_acc)
        hard_dataset = get_dataset(get_metadata(hard_templates, easy_labels), tokenizer, with_explanations, measure_applies_acc=measure_applies_acc)
        movies_dataset = get_dataset(get_metadata(templates_movies, easy_labels), tokenizer, with_explanations, measure_applies_acc=measure_applies_acc)
        laptop_dataset = get_dataset(get_metadata(templates_laptop, easy_labels), tokenizer, with_explanations, measure_applies_acc=measure_applies_acc)
        return {'easy': easy_dataset,
                'neg_easy': neg_easy_dataset,
                'neg_hard': neg_hard_dataset,
                'hard': hard_dataset,
                'movies': movies_dataset,
                'laptop': laptop_dataset}



def subset(metadata, idxs):
    metadata_subset = {}
    for key in metadata:
        metadata_subset[key] = [metadata[key][idx] for idx in idxs]
    return metadata_subset

# dataset surgery, do this lazily
# helper functions for dataset surgery
def is_good(exp):
    all_words_exp = exp.split(" ")
    return 'good' in all_words_exp

# replace a word with its more abstract form to see if we can still use an explanation. 
def perform_replacements(metadata, attribs):
    def replace_exp(exp):
        exp_split = exp.split(" ")
        exp_split[1] = attribs[exp_split[1]] 
        return " ".join(exp_split)
        
    metadata_replaced = {'instances': metadata['instances'], 'labels': metadata['labels']}
    explanations = [replace_exp(exp) for exp in metadata['explanations']]
    metadata_replaced['explanations'] = explanations
    return metadata_replaced



def knowledge(file_name, tokenizer, with_explanations=True, metadata_only=False):    
    def fn():    
        with open(file_name, 'r') as reader:
            metadata = json.load(reader)
        flip_idxs = [idx for idx, (exp, label) in enumerate(zip(metadata['explanations'], metadata['labels'])) if is_good(exp) != label]
        no_flip_idxs = [idx for idx, (exp, label) in enumerate(zip(metadata['explanations'], metadata['labels'])) if is_good(exp) == label]
        metadata_flips = subset(metadata, flip_idxs)
        metadata_no_flips = subset(metadata, no_flip_idxs)
        
        if metadata_only:
            return {'overall': metadata, 'flips': metadata_flips, 'no_flips': metadata_no_flips}
        else:
            return {'overall': get_dataset(metadata, tokenizer, with_explanations),
                    'flips': get_dataset(metadata_flips, tokenizer, with_explanations),
                    'no_flips': get_dataset(metadata_no_flips, tokenizer, with_explanations)}  
    return fn