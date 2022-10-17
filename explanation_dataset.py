import random
import torch
from helpers import verbalize_examples, prompt_styles, powerset
from tqdm import tqdm
from datasets import Dataset as HFDataset

from collections import defaultdict as ddict, Counter
from distractor_lib import distractor_fn_dict, DEFAULT_EXP
#from sentence_transformers import SentenceTransformer, util


def get_examples2exp_dict(exp2examples_dict, all_texts, with_idxs=False):
    examples2exp_pos = ddict(list)
    examples2exp_neg = ddict(list)
    total_len = 0
    for exp, idxs in exp2examples_dict.items():
        # handled at the end
        if exp == '':
            continue
        for idx in idxs:
            if with_idxs:
                examples2exp_pos[all_texts[idx]].append((exp, idx))
            else:
                examples2exp_pos[all_texts[idx]].append(exp)
            total_len += 1

    # handle the empty explanation at the end. 
    # We do this because we add the empty explanation ONLY if the example has no non-zero explanations
    for idx in exp2examples_dict['']:
        assert(with_idxs)
        examples2exp_pos[all_texts[idx]].append(('', idx))
        total_len += 1

    all_exps = set([exp for exp in exp2examples_dict]) - set(['']) # remove '' from all_exps handle at then end
    # post process such that for inputs without an explanation, we add ''. 
    for text in examples2exp_pos:
        if with_idxs:
            all_negs = list(all_exps - set([exp for exp, _ in examples2exp_pos[text]]))
        else:
            all_negs = list(all_exps - set(examples2exp_pos[text]))
        examples2exp_neg[text] = all_negs

    return examples2exp_pos, examples2exp_neg, total_len

class SimpleDataset():
    def __init__(self, all_data, tokenizer, as_lm=True, 
                 deverb_dict={'positive': 1, 'negative': 0}):
        self.all_data = all_data
        self.tensored_dataset = self.get_tensored_dataset(tokenizer, as_lm)
        self.tokenizer = tokenizer
        self.deverb_dict = deverb_dict

    def get_tensored_dataset(self, tokenizer, as_lm):

        def pad(label, max_len, val):
            to_pad = max_len - len(label)
            return label + [val] * to_pad

        data_list = self.all_data
        if as_lm:
            all_labels = tokenizer([label for _, label in data_list])['input_ids'] 
            max_len = max(len(label) for label in all_labels)
            all_labels = [pad(label, max_len, -100) for label in all_labels] 
            # TODO: if labels not same length pad with -100
            print(Counter([tuple(l) for l in all_labels]))
            dataset = {'sentence': [ex for ex, _ in data_list], 'label': all_labels}
        else:
            # deverbalize
            deverbalize_dict = self.deverb_dict 
            dataset = {'sentence': [ex for ex, _ in data_list], 'label': [deverbalize_dict[label] for _, label in data_list]}
        dataset = HFDataset.from_dict(dataset)
        tokenize_func = lambda examples: tokenizer(examples['sentence'], truncation=True, max_length=128)
        tensored_dataset = dataset.map(tokenize_func, batched=True, remove_columns=['sentence'])
        return tensored_dataset

    def __len__(self):
        return len(self.all_data)

    def get_data(self, max_size=-1):
        return self.tensored_dataset


class ExplanationApplies():
    def __init__(self, exp2examples_dict, texts, tokenizer):
        self.texts = texts
        examples2exp_pos, examples2exp_neg, total_len = get_examples2exp_dict(exp2examples_dict, texts)
        self.examples2exp_pos = examples2exp_pos
        self.examples2exp_neg = examples2exp_neg
        self.total_len = total_len
        self.exp2examples_dict = exp2examples_dict
        self.tokenizer = tokenizer


    def __len__(self):
        return self.total_len

    def get_samples(self, text):
        all_negatives = self.examples2exp_neg[text] 
        all_positives = self.examples2exp_pos[text] 
        chosen_positives = random.choice(all_positives)
        chosen_negatives = random.choice(all_negatives) 
        return chosen_positives, chosen_negatives

    def get_data(self):
        tokenizer = self.tokenizer
        all_data = {'labels': [], 'sentence': []}
        verbalizer_label = {0: 'no', 1: 'yes'}
        prompt_func = prompt_styles['p1_exp_applies']
        for example in self.examples2exp_pos:
            positive_ex, negative_ex = self.get_samples(example)
            with_correct_exp = prompt_func(positive_ex, example)
            with_incorrect_exp = prompt_func(negative_ex, example)
            all_data['labels'].append(1)
            all_data['sentence'].append(with_correct_exp)
            all_data['labels'].append(0)
            all_data['sentence'].append(with_incorrect_exp)
        all_data['labels'] = [verbalizer_label[sentiment] for sentiment in all_data['labels']]
        all_data['sentence'] = verbalize_examples(all_data['sentence'], prompt_style='p1_exp_applies')
        return self.process_into_hf_dataset(all_data, tokenizer)

    def process_into_hf_dataset(self, all_data, tokenizer):
        all_data['labels'] = tokenizer(all_data['labels'])['input_ids']
        dataset = HFDataset.from_dict(all_data)
        tokenize_func = lambda examples: tokenizer(examples['sentence'], truncation=True)
        return dataset.map(tokenize_func, batched=True, remove_columns=['sentence'])

class ExplanationDataset_2():
    def __init__(self, exp2examples_dict, texts, labels, tokenizer, prompt_style='p1', get_hard_negs=False,
                 use_distractors=False, distractor_fn_type='default', use_negatives=True):
        self.texts = texts
        self.exp2examples_dict = exp2examples_dict
        self.use_negatives = use_negatives
        examples2exp_pos, examples2exp_neg, total_len = get_examples2exp_dict(exp2examples_dict, texts, with_idxs=True, get_hard_negs=get_hard_negs)

        # default label is 0.
        example2noop_label = ddict(int)
        if DEFAULT_EXP in exp2examples_dict:
            for idx in exp2examples_dict[DEFAULT_EXP]:
                example2noop_label[texts[idx]] = labels[idx]
    
        self.example2noop_label = example2noop_label
        self.examples2exp_pos = examples2exp_pos
        self.examples2exp_neg = examples2exp_neg
        self.total_len = total_len
        self.prompt_style = prompt_style
        self.distractor_fn = distractor_fn_dict[distractor_fn_type]
        print("distractor_fn_type: {}".format(distractor_fn_type))
        print("get hard negs: {}".format(get_hard_negs))
        self.labels = labels
        self.tokenizer = tokenizer
        self.use_distractors = use_distractors

    # how many examples are constructed per epoch
    def __len__(self):
        return self.total_len

    def get_samples(self, text, pos_exp):
        all_negatives = self.examples2exp_neg[text]
        try:
            all_negatives = self.distractor_fn(pos_exp, all_negatives)
        except:
            import pdb; pdb.set_trace();
        return all_negatives

    def get_neg_data(self, num_samples=5):
        tokenizer = self.tokenizer
        all_data = {'no_exp': [], 'with_incorrect_exp': []}
        verbalizer_label = {0: 'negative', 1: 'positive'}
        prompt_func = prompt_styles[self.prompt_style]
        for example_text in self.examples2exp_pos:
            all_negatives = self.examples2exp_neg[example_text]
            sampled_explanations = random.sample(all_negatives, k = min(num_samples, len(all_negatives)))
            for explanation in sampled_explanations:
                all_data['with_incorrect_exp'].append(prompt_func(explanation, example_text))
                all_data['no_exp'].append(prompt_func('', example_text))
        all_data['with_incorrect_exp'] = verbalize_examples(all_data['with_incorrect_exp'], prompt_style='p1')
        all_data['no_exp'] = verbalize_examples(all_data['no_exp'], prompt_style='p1')
        return self.process_into_hf_dataset(all_data, tokenizer)

    def combine(self, first_exp, second_exp):
        # invariant: both cannot be zero
        if len(first_exp) == 0:
            return second_exp
        elif len(second_exp) == 0:
            return first_exp
        else:
            return '{}. {}'.format(first_exp, second_exp)

    def get_data_with_distractors(self, verbose, num_repeats=8, postprocess=True):
        tokenizer = self.tokenizer
        verbalizer_label = {0: 'negative', 1: 'positive'}
        data_list = []
        all_data = {'sentence': [], 'label': [], 'positive_explanation': [], 'negative_explanation': []}
        prompt_func = prompt_styles[self.prompt_style]
        for example_text in self.examples2exp_pos:
            for (positive_ex, idx) in self.examples2exp_pos[example_text]:
                negative_ex_all = self.get_samples(example_text, positive_ex)
                if not self.use_all_negs and len(negative_ex_all) > num_repeats:
                    negative_ex_all = random.sample(negative_ex_all, k=num_repeats)
                if len(negative_ex_all) == 0:
                    # no negatives.. this is a preference explanation.
                    explanation = positive_ex
                    instance = prompt_func(explanation, example_text)
                    label = verbalizer_label[self.labels[idx]]
                    all_data['positive_explanation'].append(positive_ex)
                    all_data['negative_explanation'].append(None)
                    all_data['sentence'].append(instance)
                    all_data['label'].append(label)

                # always sample the default explanation!
                if positive_ex !=  DEFAULT_EXP:
                    negative_ex_all.append(DEFAULT_EXP)

                # just the positive explanation
                label = verbalizer_label[self.labels[idx]]
                #all_data['sentence'].append(prompt_func(positive_ex, example_text))
                #all_data['label'].append(label)
                #all_data['positive_explanation'].append(positive_ex)
                #all_data['negative_explanation'].append('')
                for negative_ex in negative_ex_all:
                    if random.choice([0, 1]):
                        explanation = self.combine(positive_ex, negative_ex)
                    else:
                        explanation = self.combine(negative_ex, positive_ex)
                    instance = prompt_func(explanation, example_text)
                    all_data['positive_explanation'].append(positive_ex)
                    all_data['negative_explanation'].append(negative_ex)
                    all_data['sentence'].append(instance)
                    all_data['label'].append(label)
        if not postprocess:
            return all_data
        else:
            dataset = {'sentence': all_data['sentence'], 'label': tokenizer(all_data['label'])['input_ids']} 
            dataset = HFDataset.from_dict(dataset)
            if verbose:
                for ex in dataset:
                    print(ex['sentence'])
                    print(ex['label'])
            tokenize_func = lambda examples: tokenizer(examples['sentence'], truncation=True)
            tensored_dataset = dataset.map(tokenize_func, batched=True, remove_columns=['sentence'])
            return tensored_dataset


    def subset(self, data_list, indices):
        return [data_list[idx] for idx in indices]


    def get_data_helper(self, verbose, postprocess=True, max_size=-1):
        tokenizer = self.tokenizer
        verbalizer_label = {0: 'negative', 1: 'positive'}
        all_data = {'sentence': [], 'label': [], 'instances': [], 'explanations': [], 'is_pos': []}
        prompt_func = prompt_styles[self.prompt_style]


        for example_text in self.examples2exp_pos:
            for (positive_ex, idx) in self.examples2exp_pos[example_text]:
                # TODO: think about decision boundary here!!!
                if positive_ex == DEFAULT_EXP:
                    continue
                instance = prompt_func(positive_ex, example_text)
                label = verbalizer_label[self.labels[idx]]
                all_data['sentence'].append(instance)
                all_data['instances'].append(example_text)
                all_data['explanations'].append(positive_ex)
                all_data['is_pos'].append(1)
                all_data['label'].append(label)
                if self.use_negatives:
                    all_negatives = self.examples2exp_neg[example_text]
                    if len(all_negatives) > 10:
                        all_negatives = random.sample(all_negatives, k=10)
                    for neg in all_negatives:
                        if neg == DEFAULT_EXP:
                            continue
                        instance = prompt_func(neg, example_text)
                        # get the noop label, and put that here. 
                        all_data['sentence'].append(instance)
                        all_data['instances'].append(example_text)
                        all_data['explanations'].append(neg)
                        all_data['is_pos'].append(0)
                        all_data['label'].append(verbalizer_label[self.example2noop_label[example_text]])
        if not postprocess:
            all_data['label'] = [int(l == 'positive') for l in all_data['label']]
            return all_data

        dataset = {'sentence': all_data['sentence'], 'label': tokenizer(all_data['label'])['input_ids']}
        if max_size != -1 and len(all_data['sentence']) > max_size:
            all_indices = list(range(len(all_data['sentence'])))
            indices = random.sample(all_indices, k=max_size)
            dataset = {key: self.subset(dataset[key], indices) for key in dataset}



        dataset = HFDataset.from_dict(dataset)
        if verbose:
            for ex in dataset:
                print(ex['sentence'])
                print(ex['label'])
        tokenize_func = lambda examples: tokenizer(examples['sentence'], truncation=True)
        tensored_dataset = dataset.map(tokenize_func, batched=True, remove_columns=['sentence'])
        return tensored_dataset


    def get_data(self, verbose=False, postprocess=True, max_size=-1):
        if self.use_distractors:
            return self.get_data_with_distractors(verbose, postprocess=postprocess)
        else:
            return self.get_data_helper(verbose, postprocess=postprocess, max_size=max_size)

    def process_into_hf_dataset(self, all_data, tokenizer):
        dataset = HFDataset.from_dict(all_data)
        tokenize_func = lambda key: lambda ex: {'{}_{}'.format(k, key): val for k, val in  tokenizer(ex[key], truncation=True).items()}
        if 'with_correct_exp' in all_data:
            dataset = dataset.map(tokenize_func('with_correct_exp'), batched=True, remove_columns=['with_correct_exp'])
        tensored_dataset = dataset.map(tokenize_func('with_incorrect_exp'), batched=True, remove_columns=['with_incorrect_exp'])
        return tensored_dataset.map(tokenize_func('no_exp'), batched=True, remove_columns=['no_exp'])


class ExplanationDataset():
    def __init__(self, exp2examples_dict, texts, labels, gold, tokenizer, prompt_style, orig_bs = 2, per_batch=1):
        self.texts = texts
        self.exp2examples_dict = exp2examples_dict
        self.get_negative_dict()
        self.prompt_style = prompt_style
        self.labels = labels
        self.gold = gold
        self.tokenizer = tokenizer
        self.per_batch = 1 # limit batch size to 2, but accumulate gradients


        # k / (batch_size) is the original number of batches constructed. 
        # now p * exp \approx k / (batch_size) => 
        self.num_iters_per_epoch = int(len(texts) / (orig_bs * len(exp2examples_dict))) 


    def get_negative_dict(self):
        neg_dict = {}
        for exp in tqdm(self.exp2examples_dict):
            all_negatives = [idx for idx, _ in enumerate(self.texts) if idx not in self.exp2examples_dict[exp]]
            neg_dict[exp] = all_negatives
        self.exp2examples_neg_dict = neg_dict

    # how many examples are constructed per epoch
    def __len__(self):
        return self.per_batch * len(self.exp2examples_dict)

    def get_samples(self, exp):
        all_negatives = self.exp2examples_neg_dict[exp]
        all_positives = list(self.exp2examples_dict[exp])
        chosen_negatives = random.sample(all_negatives, k = self.per_batch) 
        chosen_positives = random.sample(all_positives, k = self.per_batch) 
        return chosen_positives, chosen_negatives

    def get_data(self):
        tokenizer = self.tokenizer

        all_data = {'with_explanation': [], 'without_explanation': [], 'labels': [], 'exp_applies': [], 'gold': []}
        verbalizer_label = {0: 'negative', 1: 'positive'}
        for _ in range(self.num_iters_per_epoch):
            prompt_func = prompt_styles[self.prompt_style]
            for exp in self.exp2examples_dict:
                chosen_positives, chosen_negatives = self.get_samples(exp)
                for instance_list in [chosen_positives, chosen_negatives]:
                    all_texts_with_exp = [prompt_func(exp, self.texts[idx]) for idx in instance_list]
                    all_texts_without_exp = [prompt_func('', self.texts[idx]) for idx in instance_list]
                    all_labels = [self.labels[idx] for idx in instance_list]            
                    all_data['with_explanation'] += all_texts_with_exp
                    all_data['without_explanation'] += all_texts_without_exp
                    all_data['labels'] += all_labels
                    all_data['gold'] +=  [self.gold[idx] for idx in instance_list]
                all_data['exp_applies'] += [1]*len(chosen_positives) + [0]*len(chosen_negatives)


        all_data['with_explanation'] = verbalize_examples(all_data['with_explanation'], prompt_style=self.prompt_style)
        all_data['without_explanation'] = verbalize_examples(all_data['without_explanation'], prompt_style=self.prompt_style)
        all_data['labels'] = [verbalizer_label[sentiment] for sentiment in all_data['labels']]
        return self.process_into_hf_dataset(all_data, tokenizer)

    def process_into_hf_dataset(self, all_data, tokenizer):
        all_data['labels'] = tokenizer(all_data['labels'])['input_ids']
        dataset = HFDataset.from_dict(all_data)
        tokenize_func = lambda key: lambda ex: {'{}_{}'.format(k, key): val for k, val in  tokenizer(ex[key], truncation=True).items()}

        tensored_dataset = dataset.map(tokenize_func('with_explanation'), batched=True, remove_columns=['with_explanation'])
        return tensored_dataset.map(tokenize_func('without_explanation'), batched=True, remove_columns=['without_explanation'])

