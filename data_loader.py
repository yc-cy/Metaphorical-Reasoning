import numpy as np

import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from run_classifier_dataset_utils import (
    convert_examples_to_two_features
)


def load_train_data(args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None):
    # Prepare data loader
    if task_name == "vua":
        train_examples = processor.get_train_examples(args.data_dir)
    elif task_name == "trofi":
        train_examples = processor.get_train_examples(args.data_dir, k)
    else:
        raise ("task_name should be 'vua' or 'trofi'!")


    train_features = convert_examples_to_two_features(
        train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
    )

    # make features into tensor
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_mask_ids = torch.tensor([f.mask_ids for f in train_features], dtype=torch.long)
    all_input_ids_2 = torch.tensor([f.input_ids_2 for f in train_features], dtype=torch.long)
    all_segment_ids_2 = torch.tensor([f.segment_ids_2 for f in train_features], dtype=torch.long)
    all_mask_ids_2 = torch.tensor([f.mask_ids_2 for f in train_features], dtype=torch.long)

    train_data = TensorDataset(
        all_input_ids,
        all_segment_ids,
        all_mask_ids,
        all_input_ids_2,
        all_segment_ids_2,
        all_mask_ids_2
        )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size
    )

    return train_dataloader



def load_test_data(args, logger, processor, task_name, label_list, tokenizer, output_mode, k=None):
    if task_name == "vua":
        eval_examples = processor.get_test_examples(args.data_dir)
    elif task_name == "trofi":
        eval_examples = processor.get_test_examples(args.data_dir, k)
    else:
        raise ("task_name should be 'vua' or 'trofi'!")


    eval_features = convert_examples_to_two_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, args
    )

    logger.info("***** Running evaluation *****")

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_mask_ids = torch.tensor([f.mask_ids for f in eval_features], dtype=torch.long)
    all_guids = [f.guid for f in eval_features]
    all_idx = torch.tensor([i for i in range(len(eval_features))], dtype=torch.long)

    all_input_ids_2 = torch.tensor([f.input_ids_2 for f in eval_features], dtype=torch.long)
    all_segment_ids_2 = torch.tensor([f.segment_ids_2 for f in eval_features], dtype=torch.long)
    all_mask_ids_2 = torch.tensor([f.mask_ids_2 for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(
        all_input_ids,
        all_segment_ids,
        all_mask_ids,
        all_idx,
        all_input_ids_2,
        all_segment_ids_2,
        all_mask_ids_2
    )

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    return all_guids, eval_dataloader
