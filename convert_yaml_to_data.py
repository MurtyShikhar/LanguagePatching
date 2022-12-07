from curses import meta
from distutils.spawn import find_executable
import yaml
import argparse
import json
import random
from checklist.editor import Editor
from collections import Counter
from copy import deepcopy
from eval_utils import load_model, predict_stuff
from itertools import product
from collections import defaultdict as ddict
import numpy as np

editor = Editor()


def prompt_style_0(explanation, sentence):
    return "Explanation: %s.\nInput: %s." % (explanation, sentence)

def prompt_style_1(explanation, sentence):
    if len(explanation) > 0:
        out =  "Explanation: {}. Input: {}".format(explanation, sentence)
    else:
        out = "Input: {}".format(sentence)
    out = out.rstrip()
    return out[:-1].rstrip() if out[-1] == '.' else out

def prompt_style_2(explanation, sentence):
    return "Steering hints: %s. '%s'" %(explanation, sentence)

def deconstruct(explanation_and_instance_list):
    explanations = []
    instances = []
    for eandi in explanation_and_instance_list:
        try:
            split_idx = eandi.find('Input')
            explanation = ' '.join(eandi[:split_idx].split(':')[1:]).strip()
            instance = ' '.join(eandi[split_idx:].split(':')[1:]).strip()
        except:
            instance = ' '.join(eandi.split(':')[1:]).strip()
            explanation = ''
        if explanation[-1] == '.':
            explanation = explanation[:-1]
        explanations.append(explanation)
        instances.append(instance)
    return explanations, instances

def get_default_labels(instances, mode='sentiment'):
    instances_with_exp = [('', instance) for instance in instances]
    if mode == 'sentiment':
        model_obj = load_model('models/t5-large-sst-no-exp')
    else:
        model_obj = load_model('models/t5-large-spouse_re_0.1')

    import pdb; pdb.set_trace();
    model_out = predict_stuff(instances_with_exp, [0]*len(instances), model_obj, verbose=True, prompt_style='p1')
    return model_out.argmax(axis=1)


prompt_styles = {"p0": prompt_style_0, "p1": prompt_style_1, "p2": prompt_style_2}
chosen_prompt_style = "p1"


def create_data(gpt_input, vals, key):
    # replace all occurences
    num_occurences = gpt_input.count(key)
    # consider the cartesian product num_occurences times
    replace_tuple_list = list(product(vals, repeat=num_occurences))
    all_data = []
    offset = len(key)
    for replace_tuple in replace_tuple_list:
        i = 0
        f = 0
        gpt_input_curr = deepcopy(gpt_input)
        while f < num_occurences:
            if gpt_input_curr[i : i + offset] == key:
                gpt_input_curr = "%s%s%s" % (
                    gpt_input_curr[:i],
                    replace_tuple[f],
                    gpt_input_curr[i + offset :],
                )
                f += 1
            i += 1
        all_data.append(gpt_input_curr)
    return {"data": all_data}


def gender(person):
    male_names = ['Bob', 'Stephen', 'Lee', 'Tao']
    female_names = ['Mary', 'Alice', 'Stacy']
    if person in male_names:
        return 'male'
    elif person in female_names:
        return 'female'
    else:
        return 'unknown'

def subsample(data_dict, sample_size):
    new_data_dict = {key: random.sample(data_dict[key], k=sample_size) for key in data_dict}
    return new_data_dict

def read_data_zsb(file_name, add_noop=True):
    data = {'explanations': [], 'instances': [], 'labels': []}
    with open(file_name, 'r') as stream:
        yaml_dict = yaml.load(stream)
        all_explanations = [val for _, val in yaml_dict['Explanations'].items()]
        all_template_sets = [val for _, val in yaml_dict['Templates'].items()]
        for _, (exps, template_set) in enumerate(zip(all_explanations, all_template_sets)):
            if type(exps) != list:
                exps = [exps]
            for exp in exps:
                for template in template_set:
                    label = template[-1]
                    sentence = template[0]
                    all_args = {arg: yaml_dict["Fillers"][arg] for arg in template[1:-1]}
                    print(sentence)
                    all_examples = editor.template(sentence, **all_args, remove_duplicates=True, meta=True)
                    data['explanations'] += [exp] * len(all_examples.data)
                    data['instances'] += all_examples.data 
                    data['labels'] += [label] * len(all_examples.data)

    print(Counter(data['labels']))
    if add_noop:
        instance2data = ddict(list)
        for idx, instance in enumerate(data['instances']):
            label = data['labels'][idx]
            instance = data['instances'][idx]
            exp = data['explanations'][idx]
            #instance2data[instance].append((label, instance, exp))
            # TODO: TRAIN AN APPLIES CLASSIFIER!
            instance2data[instance].append((1, instance, exp))

        all_instances = list(set(data['instances']))



        print("getting default labels for {} instances".format(len(all_instances)))
        #labels = get_default_labels(all_instances, mode='re')
        labels = [0]*len(all_instances)
        # baseline out...
        idx2labels = ddict(list)
        for idx, label in enumerate(labels):
            idx2labels[label].append(idx)


        #if len(idx2labels[1]) <= len(idx2labels[0]):
        #    chosen_negs = random.sample(idx2labels[0], k = len(idx2labels[1]))
        #    chosen_idxs = [(1, idx) for idx in idx2labels[1]] + [(0, idx) for idx in chosen_negs]
        #else:
        chosen_idxs = [(1, idx) for idx in idx2labels[1]] + [(0, idx) for idx in idx2labels[0]]
        #print("baseline acc: {}".format(baseline_acc))
        print(len(chosen_idxs), len(idx2labels[1]), len(idx2labels[0]))
        #for label, instance in zip(labels, all_instances):

        # we need to keep examples that have a different label compared to the noop label! 
        # if that is satisfied, we are good to go! 

        new_data = {'explanations': [], 'labels': [], 'instances': []}
        for label, idx in chosen_idxs:
            instance = all_instances[idx]
            old_data_list = instance2data[instance]
            to_use = False 
            for l, i, exp in old_data_list:
                #if l != label:
                if True:
                    to_use = True
                    new_data['explanations'].append(exp)
                    new_data['instances'].append(i)
                    new_data['labels'].append(l)
            if to_use:
                new_data['explanations'].append('default noop explanation')
                #new_data['labels'].append(int(label))
                #TODO:  TRAIN AN APPLIES CLASSIFIER!
                new_data['labels'].append(0)
                new_data['instances'].append(instance)
            else:
                pass
                # TODO: throw this into a rejected pile, and later, maybe accept some!
            # now add the old inputs here! 
    # subsample 10k examples
    return new_data


def read_data_re3(file_name):
    def get_only_unique(data):
        new_data = {'explanations': [], 'instances': [], 'labels': []}
        seen = set()
        for exp, instance, label in zip(data['explanations'], data['instances'], data['labels']):
            if (exp, instance) not in seen:
                new_data['explanations'].append(exp)
                new_data['instances'].append(instance)
                new_data['labels'].append(label)
                seen.add((exp, instance))

        print(len(data['instances']), len(new_data['instances']))
        return new_data 
    def get_instances(template, patch, entity_key, fillers):
        editor_temp = '%s\t%s' %(patch, template)
        all_args = {arg: fillers[arg] for arg in fillers if '{%s'%arg in editor_temp}
        all_examples = editor.template(editor_temp, **all_args, remove_duplicates=True, meta=True)
        instances = []
        patches = []
        

        person_1_key, person_2_key = entity_key.split('_')
        for inp, metadata in zip(all_examples.data, all_examples.meta):
            patch_curr, inp_curr = inp.split('\t')
            try:
                p1 = metadata[person_1_key]
                p2 = metadata[person_2_key]
            except:
                import pdb; pdb.set_trace();
            instances.append('{}. Entity1: {}. Entity2: {}'.format(inp_curr, p1, p2))
            patches.append(patch_curr)
        return instances, patches
    # for feature based patches on spouse
    data = {'explanations': [], 'instances': [], 'labels': []}
    with open(file_name, 'r') as stream:
        yaml_dict = yaml.load(stream)
        fillers = yaml_dict['FILLERS']
        for key in yaml_dict['Templates']:
            templates = yaml_dict['Templates'][key]
            
            patches = yaml_dict['Patches'][key] # get corresponding patches
            for template in templates:
                for patch in patches:
                    all_labels = yaml_dict['Labels'][key][0]
                    for entity_key in all_labels:
                        instances_curr, patches_curr = get_instances(template, patch, entity_key, fillers)
                        # also add entity2_entity?
                        label_curr = all_labels[entity_key]
                        data['instances'] += instances_curr
                        data['explanations'] += patches_curr
                        data['labels'] +=[label_curr]*len(instances_curr)

                        p1, p2 = entity_key.split('_')
                        instances_curr, patches_curr = get_instances(template, patch, '{}_{}'.format(p2, p1), fillers)
                        # also add entity2_entity1
                        label_curr = all_labels[entity_key]
                        data['instances'] += instances_curr
                        data['explanations'] += patches_curr
                        data['labels'] +=[label_curr]*len(instances_curr)


    data = get_only_unique(data)
    return data



def read_data_re2(file_name):
    inverses = {'e1': 'e2', 
                'e2': 'e1', 
                'e3': 'e3', 
                'e4': 'e4', 
                'e5': 'e5', 
                'e6': 'e6'}
    def get_instances(template, patch, entity_key, fillers):
        editor_temp = '%s\t%s' %(patch, template)
        all_args = {arg: fillers[arg] for arg in fillers if '{%s'%arg in editor_temp}
        all_examples = editor.template(editor_temp, **all_args, remove_duplicates=True, meta=True)
        #if '{location}' in template:
        #all_examples = editor.template(template, p=fillers['p'], location=fillers['location'], remove_duplicates=True, meta=True)
        #else:
        #all_examples = editor.template(template, p=fillers['p'], remove_duplicates=True, meta=True)
        instances = []
        patches = []
        

        person_1_key, person_2_key = entity_key.split('_')
        for inp, metadata in zip(all_examples.data, all_examples.meta):
            patch_curr, inp_curr = inp.split('\t')
            try:
                p1 = metadata[person_1_key]
                p2 = metadata[person_2_key]
            except:
                import pdb; pdb.set_trace();
            # create biased synthetic data
            # if label == 1 and gender(p1) == gender(p2):
            #     continue
            instances.append('{}. Entity1: {}. Entity2: {}'.format(inp_curr, p1, p2))
            patches.append(patch_curr)
        return instances, patches

    def get_only_unique(data):
        new_data = {'explanations': [], 'instances': [], 'labels': []}
        seen = set()
        for exp, instance, label in zip(data['explanations'], data['instances'], data['labels']):
            if (exp, instance) not in seen:
                new_data['explanations'].append(exp)
                new_data['instances'].append(instance)
                new_data['labels'].append(label)
                seen.add((exp, instance))

        print(len(data['instances']), len(new_data['instances']))
        return new_data 

    data = {'explanations': [], 'instances': [], 'labels': []}
    with open(file_name, 'r') as stream:
        yaml_dict = yaml.load(stream)
        all_explanations = {key: val[0] for key, val in yaml_dict['Explanations'].items()}
        for key in yaml_dict['Templates']:
            templates = yaml_dict['Templates'][key]
            labels = yaml_dict['LABELS'][key]
            for entity_key_dict in labels:
                entity_key = list(entity_key_dict.keys())[0]
                labels = entity_key_dict[entity_key].split(', ')
                print(len(labels))
                print(len(templates))
                for idx, template in enumerate(templates):
                    if labels[idx] == '_':
                        continue
                    positive_patch = all_explanations[labels[idx]]
                    instances, pos_patches = get_instances(template, positive_patch, entity_key, yaml_dict['FILLERS'])
                    data['instances'] += instances
                    data['explanations'] += pos_patches
                    data['labels'] += [1]*len(instances)

                    if labels[idx] in inverses:
                        inverse_patch = all_explanations[inverses[labels[idx]]]
                        e1, e2 = entity_key.split('_')
                        instances_2, inverse_patches = get_instances(template, inverse_patch, '{}_{}'.format(e2, e1), yaml_dict['FILLERS'])
                        data['instances'] += instances_2
                        data['explanations'] += inverse_patches
                        data['labels'] += [1]*len(instances_2)


    data = get_only_unique(data)
    return data


def read_data_re(file_name, add_noop=True):
    data = {'explanations': [], 'instances': [], 'labels': []}
    with open(file_name, 'r') as stream:
        yaml_dict = yaml.load(stream)
        all_explanations = [val for _, val in yaml_dict['Explanations'].items()]
        all_template_sets = [val for _, val in yaml_dict['Templates'].items()]
        for _, (exps, template_set) in enumerate(zip(all_explanations, all_template_sets)):
            if type(exps) != list:
                exps = [exps]
            for exp in exps:
                for template in template_set:
                    label = template[-1]
                    person_2_key = template[-2]
                    person_1_key = template[-3]
                    sentence = template[0]
                    all_args = {arg: yaml_dict["Fillers"][arg] for arg in template[1:-3]}
                    all_examples = editor.template(sentence, **all_args, remove_duplicates=True, meta=True)
                    instances_curr = []
                    for inp, metadata in zip(all_examples.data, all_examples.meta):
                        p1 = metadata[person_1_key]
                        p2 = metadata[person_2_key]
                        # create biased synthetic data
                        if label == 1 and gender(p1) == gender(p2):
                            continue
                        instances_curr.append('{}. Entity1: {}. Entity2: {}'.format(inp, p1, p2)) 
                    data['explanations'] += [exp] * len(instances_curr)
                    data['instances'] += instances_curr
                    data['labels'] += [label] * len(instances_curr)

    print(Counter(data['labels']))
    #if len(data['labels']) > 10000:
    #    data = subsample(data, 10000)
    if add_noop:
        instance2data = ddict(list)
        for idx, instance in enumerate(data['instances']):
            label = data['labels'][idx]
            instance = data['instances'][idx]
            exp = data['explanations'][idx]
            #instance2data[instance].append((label, instance, exp))
            # TODO: TRAIN AN APPLIES CLASSIFIER!
            instance2data[instance].append((1, instance, exp))

        all_instances = list(set(data['instances']))



        print("getting default labels for {} instances".format(len(all_instances)))
        #labels = get_default_labels(all_instances, mode='re')
        labels = [0]*len(all_instances)
        # baseline out...
        idx2labels = ddict(list)
        for idx, label in enumerate(labels):
            idx2labels[label].append(idx)


        #if len(idx2labels[1]) <= len(idx2labels[0]):
        #    chosen_negs = random.sample(idx2labels[0], k = len(idx2labels[1]))
        #    chosen_idxs = [(1, idx) for idx in idx2labels[1]] + [(0, idx) for idx in chosen_negs]
        #else:
        chosen_idxs = [(1, idx) for idx in idx2labels[1]] + [(0, idx) for idx in idx2labels[0]]
        #print("baseline acc: {}".format(baseline_acc))
        print(len(chosen_idxs), len(idx2labels[1]), len(idx2labels[0]))
        #for label, instance in zip(labels, all_instances):

        # we need to keep examples that have a different label compared to the noop label! 
        # if that is satisfied, we are good to go! 

        new_data = {'explanations': [], 'labels': [], 'instances': []}
        for label, idx in chosen_idxs:
            instance = all_instances[idx]
            old_data_list = instance2data[instance]
            to_use = False 
            for l, i, exp in old_data_list:
                #if l != label:
                if True:
                    to_use = True
                    new_data['explanations'].append(exp)
                    new_data['instances'].append(i)
                    new_data['labels'].append(l)
            if to_use:
                new_data['explanations'].append('default noop explanation')
                #new_data['labels'].append(int(label))
                #TODO:  TRAIN AN APPLIES CLASSIFIER!
                new_data['labels'].append(0)
                new_data['instances'].append(instance)
            else:
                pass
                # TODO: throw this into a rejected pile, and later, maybe accept some!
            # now add the old inputs here! 
    # subsample 10k examples
        return new_data
    else:
        return data


def read_data(args, file_name):
    prompt_func = prompt_styles[chosen_prompt_style]
    data = {"examples": [], "labels": [], 'is_gold': []}
    with open(file_name, "r") as stream:
        yaml_dict = yaml.load(stream)
        all_explanations = [val for key, val in yaml_dict["Explanations"].items()]
        all_template_sets = [val for key, val in yaml_dict["Templates"].items()]
        for idx, (explanation_obj, template_set) in enumerate(zip(all_explanations, all_template_sets)):
            if type(explanation_obj) != list:
                explanation_obj = [explanation_obj]

            for explanation in explanation_obj:
                for template in template_set:
                    label = template[-1]
                    sentence = template[0]
                    gpt_input = prompt_func(explanation, sentence)
                    if len(template) > 2:
                        all_args = {
                            arg: yaml_dict["Fillers"][arg] for arg in template[1:-1]
                        }
                        all_padj = all(['padj' in arg for arg in all_args if 'adj' in arg])
                        all_nadj = all(['nadj' in arg for arg in all_args if 'adj' in arg])
                        try:
                            all_examples = editor.template(gpt_input, **all_args, remove_duplicates=True)
                        except:
                            import pdb; pdb.set_trace();
                        data["examples"] += all_examples["data"]
                        if type(label) == int:
                            data["labels"] += [label] * len(all_examples["data"])
                            if all_padj or all_nadj:
                                print(gpt_input)
                                data['is_gold'] += [1] * len(all_examples['data'])
                            else:
                                data['is_gold'] += [0] * len(all_examples['data'])
                        else:
                            idxs = list(range(len(all_examples["data"])))
                            chosen_idxs = random.sample(idxs, k=int(len(idxs) * label))
                            data["labels"] += [
                                (1 if idx in chosen_idxs else 0) for idx in idxs
                            ]
                    else:
                        data["examples"] += [gpt_input]
                        data["labels"] += [label]
    explanations, instances = deconstruct(data['examples'])
    data['explanations'] = [exp.lower() for exp in explanations]
    data['instances'] = instances
    ## for each explanation, we have positives...
    ## all the negatives can be read from NOOP?
    # now just need to change the labels!!
    default_idxs = [idx for idx, ex in enumerate(data['explanations']) if ex == 'default noop explanation']
    if len(default_idxs) == 0:
        print("No default explanation. Make sure this is the correct behavior.")
        all_instances = list(set(data['instances']))
        print("getting default labels for {} instances".format(len(all_instances)))
        labels = get_default_labels(all_instances)
        for instance, label in zip(all_instances, labels):
            data['explanations'].append('default noop explanation')
            data['labels'].append(int(label))
            data['instances'].append(instance)
    else:
        default_instances = [data['instances'][idx] for idx in default_idxs]
        labels = get_default_labels(default_instances)
        for oidx, idx in enumerate(default_idxs):
            data['labels'][idx] = int(labels[oidx])
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser("create data from yaml file")
    parser.add_argument("--exp_dir", type=str)
    parser.add_argument("--mode", type=str, default='sentiment')

    args = parser.parse_args()
    file_name = "{}/explanations.yaml".format(args.exp_dir)
    if args.mode == 'sentiment':
        data = read_data(args, file_name)
    elif args.mode == 're':
        data = read_data_re3(file_name)
    else:
        data = read_data_zsb(file_name, add_noop=True)
    print(Counter(data['labels']))
    with open("{}/synthetic_data.json".format(args.exp_dir), "w") as writer:
        json.dump(data, writer)
