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
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

from torch.optim import AdamW

from utils import Config, Logger, make_log_dir
from run_classifier_dataset_utils import processors, output_modes, compute_metrics
from data_loader import load_train_data, load_test_data

import wandb
wandb.login(key="6c0882235635cf99e30f943812fc1027c03a8883")


from torch.utils.data import Dataset

from run_classifier_dataset_utils import (
    convert_examples_to_two_features
)


CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
ARGS_NAME = "training_args.bin"


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

    tokenizer = GPT2Tokenizer.from_pretrained(args.bert_model)
    model = GPT2LMHeadModel.from_pretrained(args.bert_model)


    train_examples = processor.get_train_examples(args.data_dir)
    train_features = convert_examples_to_two_features(
        train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
    )

    eval_examples = processor.get_test_examples(args.data_dir)
    eval_features = convert_examples_to_two_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
    )

    train_dataset = Train_Dataset(train_features)
    test_dataset = Test_Dataset(eval_features)


    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=15,
        per_device_train_batch_size=35,
        per_device_eval_batch_size=35,
        metric_for_best_model='eval_samples_per_second',
        evaluation_strategy='steps',
        eval_steps=60,
        logging_dir='./logs',
        logging_steps=100,
        load_best_model_at_end=True,
        save_steps=120,
        save_total_limit=3,
        no_cuda=False,
        seed=42,
        learning_rate=5e-5,
        disable_tqdm=False,
        dataloader_drop_last=False,
    )



    # 定义Trainer并传入优化器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 开始微调
    trainer.train()


    model.to("cpu")
    f = open("example.txt", "a")

    for i, eval in enumerate(eval_features):

        if i > 50:
            break

        input_ids = torch.tensor(eval.input_ids_2, dtype=torch.long).unsqueeze(0)
        mask_ids = torch.tensor(eval.mask_ids_2, dtype=torch.long).unsqueeze(0)
        generated_text = model.generate(
            input_ids=input_ids,
            attention_mask=mask_ids,
            max_new_tokens=50,
            num_return_sequences=1
        )

        tokenizer.padding_side = "left"


        text1 = tokenizer.decode(generated_text[0], skip_special_tokens=True)
        text2 = tokenizer.decode(eval.segment_ids_2, skip_special_tokens=True)


        f.write("\n")
        f.write("------------------------------------------------------")
        f.write("\n")
        f.write("pred sent:")
        f.write("\n")
        f.write(text1)
        f.write("\n")
        f.write("true sent:")
        f.write("\n")
        f.write(text2)
        f.write("\n")
        f.write("------------------------------------------------------")
        f.write("\n")




def save_model(args, model, tokenizer):
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.log_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.log_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.log_dir)

    # Good practice: save your training arguments together with the trained model
    output_args_file = os.path.join(args.log_dir, ARGS_NAME)
    torch.save(args, output_args_file)


def load_trained_model(args, model, tokenizer):
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.log_dir, WEIGHTS_NAME)

    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(output_model_file))
    else:
        model.load_state_dict(torch.load(output_model_file))

    return model


if __name__ == "__main__":
    main()
