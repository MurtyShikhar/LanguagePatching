from checklist.editor import Editor
import os
import re
import pickle
from datasets import Dataset as HFDataset
from hydra.utils import get_original_cwd

editor = Editor()
pos_adjectives = ["good", "great", "amazing", "wonderful", "awesome"]
neg_adjectives = ["bad", "pathetic", "awful", "terrible", "horrid"]
# add in some gibberish words
gibberish_adjectives = ["zonker", "wonker", "zubin", "wugly", "shug"]
all_pos_adjectives = pos_adjectives + gibberish_adjectives
all_neg_adjectives = neg_adjectives + gibberish_adjectives

restaurant_explanations = [
    ("food is more important for determining sentiment than service", "food"),
    (
        "service is more important for determining sentiment than food",
        "service",
    ),
    ("quality of food determines sentiment", "food"),
    ("quality of service determines sentiment", "service"),
    ("food matters more than service for determining sentiment", "food"),
    ("service matters more than food for determining sentiment", "service"),
]


# this is the null prompt
def prompt_style_0(explanation, sentence):
    if len(explanation) > 0:
        return (explanation, sentence)
    else:
        return sentence

def make_re_transforms(explanation, sentence):
    text, ent_info = sentence.split(' Entity1:')
    e1, e2 = ent_info.split('. Entity2:')

    e1 = e1.strip()
    e2 = e2.strip()
    exp_new = explanation.replace('Entity1', e1).replace('Entity2', e2)
    return exp_new, text


def prompt_style_1(explanation, sentence):
    if len(explanation) > 0:
        #if 'Entity1:' in sentence:
        #    explanation, sentence = make_re_transforms(explanation, sentence)
        out =  "Explanation: {}. Input: {}".format(explanation, sentence)
    else:
        out = "Input: {}".format(sentence)
    out = out.rstrip()
    return out[:-1].rstrip() if out[-1] == '.' else out

def prompt_style_reverse(explanation, sentence):
    if len(explanation) > 0:
        out =  "Input: {}. Explanation: {}".format(sentence, explanation)
    else:
        out = "Input: {}".format(sentence)
    out = out.rstrip()
    return out[:-1].rstrip() if out[-1] == '.' else out

def prompt_style_2(explanation, sentence):
    if len(explanation) > 0:
        return "Steering hints: %s. '%s'" %(explanation, sentence)
    else:
        return "'%s'" %sentence


def prompt_style_1_exp_applies(explanation, sentence):
    out = 'Explanation: {}. Input: {}'.format(explanation, sentence)
    out = out.rstrip()
    return out[:-1] if out[-1] == '.' else out

prompt_styles = {'p0': prompt_style_0, 'p1': prompt_style_1, 'p1_reverse': prompt_style_reverse, 'p1_exp_applies': prompt_style_1_exp_applies, 'p2': prompt_style_2} 

def convert_to_tensors(data_list, tokenizer):
    dataset = {'sentence': [ex for ex, _ in data_list], 'label': tokenizer([label for _, label in data_list])['input_ids']}
    dataset = HFDataset.from_dict(dataset)

    tokenize_func = lambda examples: tokenizer(examples['sentence'], truncation=True)
    tensored_dataset = dataset.map(tokenize_func, batched=True, remove_columns=['sentence'])
    return tensored_dataset


def on_azure():
    return any("AZURE" in key for key in os.environ)



def get_zs_benchmarking_data(prompt_style, split='train'):
    try:
        base_dir = get_original_cwd()
    except:
        base_dir = ''
    data_dir = os.path.join(base_dir, 'META_TUNING_DATA')


    if split=='train':
        pickle_files = ['group7_BenchmarkingZeroshotSituationCrime.pkl', 
                    'group7_BenchmarkingZeroshotSituationMed.pkl', 
                    'group7_BenchmarkingZeroshotSituationEvac.pkl', 
                    'group7_BenchmarkingZeroshotSituationSearch.pkl', 
                    'group7_BenchmarkingZeroshotSituationTerr.pkl', 
                    'group7_BenchmarkingZeroshotSituationRegime.pkl']
    elif split=='development':
        # use errors on these to design explanation finetuning. 
        pickle_files = ['group7_BenchmarkingZeroshotSituationUtils.pkl', 'group7_BenchmarkingZeroshotSituationWater.pkl']
    elif split == 'test': 
        # eval on these for patching purposes
        pickle_files = ['group7_BenchmarkingZeroshotSituationShelter.pkl', 
                        'group7_BenchmarkingZeroshotSituationFood.pkl', 
                        'group7_BenchmarkingZeroshotSituationInfra.pkl']
    data = []
    pf = prompt_styles[prompt_style]
    
    def normalize(t):
        return re.sub("'(.+)'", r'\1', t.lower())

    def qc2input(d):
        normalized_inp =  normalize(d['q'] + " [SEP] " + d['c'])
        return pf('', normalized_inp)

    def process(data_list):
        return [(qc2input(d), d['a']) for d in data_list]        


    for pickle_file in pickle_files:
        with open('{}/{}'.format(data_dir,pickle_file), 'rb') as reader:
            data += process(pickle.load(reader))
    return data
    #label_verbalizer = {1: 'positive', 0: 'negative'}
    #pf = prompt_styles[prompt_style]


    #verbalized_data = []
    #for ex, label in data:
    #    verbalized_ex = pf('', ex)
    #    verbalized_label = label_verbalizer[int(label)]
    #    verbalized_data.append((verbalized_ex, verbalized_label))
    #return verbalized_data


def get_cc_data(split, prompt_style):
    from datasets import load_dataset
    dataset = load_dataset('jigsaw_toxicity_pred', data_dir='toxicity')[split]
    label_verbalizer = {1: 'positive', 0: 'negative'}
    pf = prompt_styles[prompt_style]
    verbalized_data = []
    for ex in dataset:
        verbalized_ex = pf('', ex['comment_text'])
        verbalized_label = label_verbalizer[ex['toxic']]
        verbalized_data.append((verbalized_ex, verbalized_label))
    return verbalized_data

def get_re_data(split, dirname, prompt_style, use_percent = 1.0):
    base_dir = get_original_cwd()

    if use_percent != 1.0 and split == 'train':
        split = 'train_{}'.format(use_percent)

    split2filename = {'train': '{}/{}/train.txt.processed'.format(base_dir, dirname), 
                      'train_0.1': '{}/{}/train.txt_0.1.processed'.format(base_dir, dirname), 
                      'train_0.5': '{}/{}/train.txt_0.5.processed'.format(base_dir, dirname), 
                      'val': '{}/{}/val.txt.processed'.format(base_dir, dirname),
                      'test': '{}/{}/test.txt.processed'.format(base_dir, dirname)}
    with open(split2filename[split], 'r') as reader:
        data = [line.strip().split('\t') for line in reader.readlines()]
    
    label_verbalizer = {1: 'positive', 0: 'negative'}
    pf = prompt_styles[prompt_style]
    verbalized_data = []
    for ex, label in data:
        verbalized_ex = pf('', ex)
        verbalized_label = label_verbalizer[int(label)]
        verbalized_data.append((verbalized_ex, verbalized_label))
    return verbalized_data

def get_spouse_data(split, prompt_style, use_percent= 1.0):
    return get_re_data(split, 'SPOUSE_DATA', prompt_style, use_percent)

def get_cdr_data(split, prompt_style, use_percent= 1.0):
    return get_re_data(split, 'CDR_DATA', prompt_style, use_percent)



def verbalize_all_qqp(data):
    label_verbalizer = {1: 'positive', 0: 'negative'}
    verbalized_data = []
    for ex in data:
        sent1 = ex['question1']
        sent2 = ex['question2']
        label = ex['label']
        verbalized_label = label_verbalizer[int(label)]
        # separated by question marks.
        verbalized_ex = 'Sentence1: {} Sentence2: {}'.format(sent1, sent2)
        verbalized_data.append((verbalized_ex, verbalized_label))
    return verbalized_data


def verbalize_all(data, prompt_style='p1'):
    label_verbalizer = {1: 'positive', 0: 'negative'}
    pf = prompt_styles[prompt_style]
    verbalized_data = []
    for ex, label in data:
        verbalized_ex = pf('', ex)
        verbalized_label = label_verbalizer[int(label)]
        verbalized_data.append((verbalized_ex, verbalized_label))
    return verbalized_data



def word_polarity_helper(
    pos_words,
    neg_words,
    word_templates,
    always_add=False,
    control_explanation=False,
    add_eos_explanation=False,
    no_exp=False
):
    all_data = []
    aux_exp = "negating a word changes sentiment" if not control_explanation else "blah"
    if always_add:
        neg_exp = "{adj} is a negative word. %s" % aux_exp
        pos_exp = "{adj} is a positive word. %s" % aux_exp
    else:
        neg_exp = "{adj} is a negative word"
        pos_exp = "{adj} is a positive word"
    if add_eos_explanation:
        neg_exp = "%s. :z at the end of input changes sentiment" % neg_exp
        pos_exp = "%s. :z at the end of input changes sentiment" % pos_exp
        word_templates = ["{} :z".format(template) for template in word_templates]


    def construct_input(word_template, explanation, negated_sense):
        if no_exp:
            return "'%s'" %word_template
        elif negated_sense:
            if always_add:
                curr_input = "Explanation: %s. '%s'" % (explanation, word_template)
            else:
                curr_input = "Explanation: %s. %s. '%s'" % (explanation, aux_exp, word_template)
            return curr_input
        else: 
            return "Explanation: %s. '%s'" % (explanation, word_template)

    if len(neg_words) > 0:
        curr_input = construct_input(word_templates[0], neg_exp, False)
        template = editor.template(
            curr_input,
            place=["restaurant", "food", "person", "movie"],
            iswas=["is", "was"],
            adj=neg_words,
        )
        all_data += [(example, int(add_eos_explanation)) for example in template["data"]]
        curr_input = construct_input(word_templates[1], neg_exp, True)
        template = editor.template(
            curr_input,
            place=["restaurant", "food", "person", "movie"],
            iswas=["is", "was"],
            adj=neg_words,
        )
        all_data += [
            (example, 1 - int(add_eos_explanation)) for example in template["data"]
        ]

    if len(pos_words) > 0:
        curr_input = construct_input(word_templates[0], pos_exp, False)
        template = editor.template(
            curr_input,
            place=["restaurant", "food", "person", "movie"],
            iswas=["is", "was"],
            adj=pos_words,
        )
        all_data += [
            (example, 1 - int(add_eos_explanation)) for example in template["data"]
        ]

        curr_input = construct_input(word_templates[1], pos_exp, True)
        template = editor.template(
            curr_input,
            place=["restaurant", "food", "person", "movie"],
            iswas=["is", "was"],
            adj=pos_words,
        )
        all_data += [(example, int(add_eos_explanation)) for example in template["data"]]
    return all_data


def word_polarity(
    pos_words,
    neg_words,
    always_add=False,
    control_explanation=False,
    add_eos_explanation=False,
    no_exp=False
):
    word_templates = [
        "The {place} {iswas} {adj}",
        "The {place} {iswas} not {adj}",
    ]
    return word_polarity_helper(
        pos_words,
        neg_words,
        word_templates=word_templates,
        always_add=always_add,
        control_explanation=control_explanation,
        add_eos_explanation=add_eos_explanation,
        no_exp=no_exp
    )


def type2_examples(
    place_type,
    attrib_1,
    attrib_2,
    explanation_list,
    pos_adj_list,
    neg_adj_list,
    use_symbol_at_eos=False,
):
    all_data = []
    for exp, exp_type in explanation_list:
        if use_symbol_at_eos:
            t1 = editor.template(
                "Explanation: %s. 'The %s had {padj} %s and {nadj} %s :z'"
                % (exp, place_type, attrib_1, attrib_2),
                padj=pos_adj_list,
                nadj=neg_adj_list,
            )
            t2 = editor.template(
                "Explanation: %s. 'The %s had {nadj} %s and {padj} %s :z'"
                % (exp, place_type, attrib_2, attrib_1),
                padj=pos_adj_list,
                nadj=neg_adj_list,
            )
            # The same but slightly different ordering
            t3 = editor.template(
                "Explanation: %s. 'The %s had {nadj} %s and {padj} %s :z'"
                % (exp, place_type, attrib_1, attrib_2),
                padj=pos_adj_list,
                nadj=neg_adj_list,
            )
            t4 = editor.template(
                "Explanation: %s. 'The %s had {padj} %s and {nadj} %s :z'"
                % (exp, place_type, attrib_2, attrib_1),
                padj=pos_adj_list,
                nadj=neg_adj_list,
            )
        else:
            t1 = editor.template(
                "Explanation: %s. 'The %s had {padj} %s and {nadj} %s'"
                % (exp, place_type, attrib_1, attrib_2),
                padj=pos_adj_list,
                nadj=neg_adj_list,
            )
            t2 = editor.template(
                "Explanation: %s. 'The %s had {nadj} %s and {padj} %s'"
                % (exp, place_type, attrib_2, attrib_1),
                padj=pos_adj_list,
                nadj=neg_adj_list,
            )
            # The same but slightly different ordering
            t3 = editor.template(
                "Explanation: %s. 'The %s had {nadj} %s and {padj} %s'"
                % (exp, place_type, attrib_1, attrib_2),
                padj=pos_adj_list,
                nadj=neg_adj_list,
            )
            t4 = editor.template(
                "Explanation: %s. 'The %s had {padj} %s and {nadj} %s'"
                % (exp, place_type, attrib_2, attrib_1),
                padj=pos_adj_list,
                nadj=neg_adj_list,
            )
        # good food is postive, bad food is negative
        if exp_type == attrib_1:
            label_set = [1, 1, 0, 0]
        else:
            label_set = [0, 0, 1, 1]
        if use_symbol_at_eos:
            label_set = [int(not label) for label in label_set]
        templates = [t1, t2, t3, t4]
        for ex, label in zip(templates, label_set):
            all_data += [(example, label) for example in ex["data"]]
    return all_data


def type3_examples(
    place_type,
    attrib_1,
    attrib_2,
    explanation_list,
    pos_adj_list,
    neg_adj_list,
):
    all_data = []
    for exp, exp_type in explanation_list:
        # The plot of the movie was not good while the cast was great
        # The food at the restaurant was not good while the service was great
        t1 = editor.template(
            "Explanation: %s. 'The %s %s was not {padj} while the %s was {padj}'"
            % (exp, attrib_1, place_type, attrib_2),
            padj=pos_adj_list,
        )
        # The cast of the movie was not good while the plot was great
        # The service at the restaurant was not good while the food was great
        t2 = editor.template(
            "Explanation: %s. 'The %s %s was not {padj} while the %s was {padj}'"
            % (exp, attrib_2, place_type, attrib_1),
            padj=pos_adj_list,
        )
        t3 = editor.template(
            "Explanation: %s. 'The %s %s was not {nadj} while the %s was {nadj}'"
            % (exp, attrib_1, place_type, attrib_2),
            nadj=neg_adj_list,
        )
        # The cast of the movie was not good while the plot was great
        # The service at the restaurant was not good while the food was great
        t4 = editor.template(
            "Explanation: %s. 'The %s %s was not {nadj} while the %s was {nadj}'"
            % (exp, attrib_2, place_type, attrib_1),
            nadj=neg_adj_list,
        )
        if exp_type == attrib_1:
            label_set = [0, 1, 1, 0]
        else:
            label_set = [1, 0, 0, 1]
        templates = [t1, t2, t3, t4]
        for ex, label in zip(templates, label_set):
            all_data += [(example, label) for example in ex["data"]]
    return all_data


def restaurant_2():
    return type3_examples(
        "at the restaurant",
        "food",
        "service",
        restaurant_explanations,
        pos_adjectives,
        neg_adjectives,
    )


def restaurant_and_neg_words(use_symbol_at_eos=False):
    composed_explanations = [
        ("{nadj} is a negative word. %s" % exp, kind)
        for (exp, kind) in restaurant_explanations
    ]
    if use_symbol_at_eos:
        composed_explanations = [
            ("%s. :z at the end of input changes sentiment" % exp, kind)
            for (exp, kind) in composed_explanations
        ]
    return type2_examples(
        "restaurant",
        "food",
        "service",
        composed_explanations,
        pos_adjectives,
        gibberish_adjectives,
        use_symbol_at_eos=use_symbol_at_eos,
    )


def restaurant_and_pos_words(use_symbol_at_eos=False):
    composed_explanations = [
        ("{padj} is a positive word. %s" % exp, etype)
        for (exp, etype) in restaurant_explanations
    ]
    if use_symbol_at_eos:
        composed_explanations = [
            ("%s. :z at the end of input changes sentiment" % exp, kind)
            for (exp, kind) in composed_explanations
        ]
    return type2_examples(
        "restaurant",
        "food",
        "service",
        composed_explanations,
        gibberish_adjectives,
        neg_adjectives,
        use_symbol_at_eos=use_symbol_at_eos,
    )


def restaurant_and_symbol_at_eos():
    composed_explanations = [
        ("%s. :z at the end of input changes sentiment" % exp, etype)
        for exp, etype in restaurant_explanations
    ]
    return type2_examples(
        "restaurant",
        "food",
        "service",
        composed_explanations,
        pos_adjectives,
        neg_adjectives,
        use_symbol_at_eos=True,
    )


def restaurant():
    return type2_examples(
        "restaurant",
        "food",
        "service",
        restaurant_explanations,
        pos_adjectives,
        neg_adjectives,
    )


def simple_examples():
    return word_polarity(all_pos_adjectives, all_neg_adjectives)

from itertools import chain, combinations, permutations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(permutations(s, r) for r in range(len(s)+1))


def verbalize_examples(examples, prompt_style='p0', labels_given=False, label_name_dict=None):
    verbalized_examples = []
    if labels_given:
        if label_name_dict is None:
            label_name_dict = {0: 'negative', 1: 'positive'}

    for example in examples:
        if type(example) == tuple:
            text, label = example
            assert(labels_given)
            verbalized_label = label_name_dict[label]
        else:
            text = example
            verbalized_label = None

        if prompt_style == 'p0':
            verbalized_text = text
        else:
            verbalized_text = text

        if verbalized_label:
            verbalized_examples.append((verbalized_text, verbalized_label))
        else:
            verbalized_examples.append(verbalized_text)
    return verbalized_examples

def symbol_at_eos(examples):
    all_data = []
    for ex in examples:
        cinput = ex["sentence"].rstrip() + " :z"
        explanation = ':z at the end of input changes sentiment'
        label = 1 - int(ex["label"] > 0.5)
        all_data.append((cinput, label, explanation))
    return all_data
