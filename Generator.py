import os
import sys
import pickle
import random
import copy
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm, trange
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

from torch.optim import AdamW

from utils import Config, Logger, make_log_dir
from run_classifier_dataset_utils import processors, output_modes, compute_metrics
from data_loader import load_train_data, load_test_data

# import wandb

# wandb.login(key="6c0882235635cf99e30f943812fc1027c03a8883")

from torch.utils.data import Dataset

from run_classifier_dataset_utils import (
    convert_examples_to_two_features
)


class Test_Dataset(Dataset):
    def __init__(self, input_features):
        self.input_features = input_features

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, index):
        input_ids_2 = torch.tensor(self.input_features[index].input_ids_2, dtype=torch.long)

        return {
            'input_ids': input_ids_2
        }


class Train_Dataset(Dataset):
    def __init__(self, input_features):
        self.input_features = input_features

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.input_features[index].input_ids, dtype=torch.long)
        attention_mask = torch.tensor(self.input_features[index].mask_ids, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids
        }


def main():
    # read configs
    config = Config(main_conf_path="./")

    # apply system arguments if exist
    argv = sys.argv[1:]
    if len(argv) > 0:
        cmd_arg = OrderedDict()
        argvs = " ".join(sys.argv[1:]).split(" ")
        for i in range(0, len(argvs), 2):
            arg_name, arg_value = argvs[i], argvs[i + 1]
            arg_name = arg_name.strip("-")
            cmd_arg[arg_name] = arg_value
        config.update_params(cmd_arg)

    args = config
    print(args.__dict__)

    # logger
    if "saves" in args.bert_model:
        log_dir = args.bert_model
        logger = Logger(log_dir)
        config = Config(main_conf_path=log_dir)
        old_args = copy.deepcopy(args)
        args.__dict__.update(config.__dict__)

        args.bert_model = old_args.bert_model
        args.do_train = old_args.do_train
        args.data_dir = old_args.data_dir
        args.task_name = old_args.task_name

        # apply system arguments if exist
        argv = sys.argv[1:]
        if len(argv) > 0:
            cmd_arg = OrderedDict()
            argvs = " ".join(sys.argv[1:]).split(" ")
            for i in range(0, len(argvs), 2):
                arg_name, arg_value = argvs[i], argvs[i + 1]
                arg_name = arg_name.strip("-")
                cmd_arg[arg_name] = arg_value
            config.update_params(cmd_arg)
    else:
        if not os.path.exists("saves"):
            os.mkdir("saves")
        log_dir = make_log_dir(os.path.join("saves", args.bert_model))
        logger = Logger(log_dir)
        config.save(log_dir)
    args.log_dir = log_dir

    # set CUDA devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logger.info("device: {} n_gpu: {}".format(device, args.n_gpu))

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # get dataset and processor
    task_name = args.task_name.lower()
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    args.num_labels = len(label_list)

    tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_models/trofi_model_120')
    model = GPT2LMHeadModel.from_pretrained('./gpt2_models/trofi_model_120')

    # tokenizer = AutoTokenizer.from_pretrained("./gpt1_models/vua_model_50")
    # model = AutoModel.from_pretrained("./gpt1_models/vua_model_50")
    
    model.to(device)

    f = open("result.txt", "w")

    model.eval()

    all_guids, eval_dataloader = load_test_data(
        args, logger, processor, task_name, label_list, tokenizer, output_mode
    )

    for eval_batch in tqdm(eval_dataloader, desc="Evaluating"):
        eval_batch = tuple(t.to(args.device) for t in eval_batch)

        (
            input_ids,
            segment_ids,
            mask_ids,
            idx,
            input_ids_2,
            segment_ids_2,
            mask_ids_2
        ) = eval_batch

        with torch.no_grad():

            generated_text = model.generate(
                input_ids=input_ids_2,
                attention_mask=mask_ids_2,
                max_new_tokens=100,
                num_return_sequences=1
            )

            for g, l in zip(generated_text, segment_ids_2):
                text11 = tokenizer.decode(g[:args.max_seq_length], skip_special_tokens=True)
                text12 = tokenizer.decode(g[args.max_seq_length:], skip_special_tokens=True)
                text2 = tokenizer.decode(l, skip_special_tokens=True)

                f.write("\n")
                f.write("------------------------------------------------------")
                f.write("\n")
                f.write(text11)
                f.write("\n")
                f.write("pred:")
                f.write("\n")
                f.write(text12)
                f.write("\n")
                f.write("label:")
                f.write("\n")
                f.write(text2)
                f.write("\n")
                f.write("------------------------------------------------------")
                f.write("\n")


if __name__ == "__main__":
    main()
