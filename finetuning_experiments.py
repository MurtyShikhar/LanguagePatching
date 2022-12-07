# ### How many labeled examples is patching worth? ####


from eval_utils import predict_stuff
from data_utils import (
    get_yelp_colloquial_control,
    get_yelp_stars,
    get_yelp_colloquial,
    get_fewrel_data,
)
from eval_utils import load_model, predict_stuff
import numpy as np
import argparse
import random
import re
import torch
from eval_utils import fewshot_finetune


def compare(inputs, labels, exp, path_name):
    model = load_model(path_name)
    inputs_without_exp = [("", inp) for inp in inputs]
    baseline_1_out = predict_stuff(
        inputs_without_exp, labels, model, verbose=True, prompt_style="p1"
    )
    baseline_1_labels = baseline_1_out.argmax(axis=1)

    inputs_with_exp = [(exp, inp) for inp in inputs]
    baseline_1_out_withexp = predict_stuff(
        inputs_with_exp, labels, model, verbose=True, prompt_style="p1"
    )
    baseline_1_labels_withexp = baseline_1_out_withexp.argmax(axis=1)

    print(np.mean(baseline_1_labels == np.array(labels)))
    print(np.mean(baseline_1_labels_withexp == np.array(labels)))


# keep 80% of yelp stars for testing, and let's say we get 20% for finetuning. what happens?
def get_sample(inputs, labels, sample_size):
    all_idxs = [idx for idx, _ in enumerate(inputs)]
    sampled_idxs = set(random.sample(all_idxs, k=sample_size))

    train_inputs = [inputs[idx] for idx, _ in enumerate(inputs) if idx in sampled_idxs]
    test_inputs = [
        inputs[idx] for idx, _ in enumerate(inputs) if idx not in sampled_idxs
    ]

    train_labels = [labels[idx] for idx, _ in enumerate(inputs) if idx in sampled_idxs]
    test_labels = [
        labels[idx] for idx, _ in enumerate(inputs) if idx not in sampled_idxs
    ]
    return (train_inputs, train_labels), (test_inputs, test_labels)


def get_fewshot_curve(train_data, test_data, path_name, metric="acc"):
    # for all tasks this is how we get subsets to stay balanced
    def get_subset_balanced(num_examples):
        shots = num_examples // 2
        pos_ex = [
            train_data[0][idx] for idx, label in enumerate(train_data[1]) if label == 1
        ]
        neg_ex = [
            train_data[0][idx] for idx, label in enumerate(train_data[1]) if label == 0
        ]
        chosen_pos = random.sample(pos_ex, k=shots)
        chosen_neg = random.sample(neg_ex, k=shots)
        labels = [1] * len(chosen_pos) + [0] * len(chosen_neg)
        return chosen_pos + chosen_neg, labels

    # for relation extraction though, we get subsets like this, since we don't want to balance the dataset!
    def get_subset_re(num_examples):
        all_idxs = [idx for idx, _ in enumerate(train_data[0])]
        sampled_idxs = random.sample(all_idxs, k=num_examples)
        chosen_inps = [train_data[0][idx] for idx in sampled_idxs]
        chosen_labels = [train_data[1][idx] for idx in sampled_idxs]
        return chosen_inps, chosen_labels

    all_acc_dict = {}
    for num_examples in [2, 4, 8, 16, 32, 64, 128]:
        try:
            if metric == "acc":
                train_subset = get_subset_balanced(num_examples)
            else:
                train_subset = get_subset_re(num_examples)
        except:
            break
        curr_acc = fewshot_finetune(
            path_name,
            update_steps=64,
            train_tuple_list=train_subset,
            val_tuple_list=test_data,
            metric=metric,
        )
        all_acc_dict[num_examples] = curr_acc
        print(curr_acc)
    return all_acc_dict


def set_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="stars")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--path_name", type=str)
    parser.add_argument("--sample_size", type=int, default=256)

    args = parser.parse_args()

    set_seed(args.seed)
    if args.type == "stars":
        star_inputs, star_labels, _ = get_yelp_stars()
        star_train_data, star_test_data = get_sample(
            star_inputs, star_labels, args.sample_size
        )
        all_data_dict = get_fewshot_curve(
            star_train_data, star_test_data, args.path_name
        )
        with open("finetuning_logs/{}_{}.txt".format(args.type, args.seed), "w") as f:
            f.write(str(all_data_dict))
    elif args.type == "spouse_nyt":
        pos, neg = get_fewrel_data()
        labels = [1] * len(pos) + [0] * len(neg)
        inputs = pos + neg
        train_data, test_data = get_sample(inputs, labels, args.sample_size)
        all_data_dict = get_fewshot_curve(
            train_data, test_data, args.path_name, metric="f1"
        )
        with open("finetuning_logs/{}_{}.txt".format(args.type, args.seed), "w") as f:
            f.write(str(all_data_dict))
    elif "yelp_colloquial" in args.type:
        inputs, labels = get_yelp_colloquial()
        print(len(inputs))
        train_data, test_data = get_sample(inputs, labels, args.sample_size)
        if "control" in args.type:
            test_data = get_yelp_colloquial_control()
        all_data_dict = get_fewshot_curve(train_data, test_data, args.path_name)
        with open("finetuning_logs/{}_{}.txt".format(args.type, args.seed), "w") as f:
            f.write(str(all_data_dict))


if __name__ == "__main__":
    main()
