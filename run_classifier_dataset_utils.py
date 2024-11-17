from __future__ import absolute_import, division, print_function

import copy
import csv
import logging
import os
import sys
import torch

from scipy.stats import pearsonr, spearmanr, truncnorm
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
)


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(
        self,
        guid,
        text_a,
        text_b,
        label,
        word,
        answer
    ):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.word = word
        self.answer = answer


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids,
        segment_ids,
        mask_ids,
        guid,
        input_ids_2,
        segment_ids_2,
        mask_ids_2
    ):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.mask_ids = mask_ids
        self.guid = guid
        self.input_ids_2 = input_ids_2
        self.segment_ids_2 = segment_ids_2
        self.mask_ids_2 = mask_ids_2


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                lines.append(line)
            return lines




class VUAProcessor(DataProcessor):
    """Processor for the VUA data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.csv")), "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.csv")), "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):

            # if i == 0:
            #     continue 

            guid = "%s-%s" % (set_type, line[0])
            text_a = line[3]
            label = line[4]
            word = line[1]
            index = line[0]
            answer = line[6]
            examples.append(
                InputExample(
                    guid=guid, text_a=text_a, text_b=index, label=label, word=word, answer=answer
                )
            )
        return examples



def convert_examples_to_two_features(
    examples, label_list, max_seq_length, tokenizer, output_mode, args
):

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        prompt = f'Is “{example.word}” used metaphorically in “{example.text_a}”?'

        label = example.answer


        tokens_a = tokenizer.tokenize(prompt)
        segment_ids = [1] * len(tokens_a)

        # ------------------------------------------

        test_token = copy.deepcopy(tokens_a)

        test_input_ids = tokenizer.convert_tokens_to_ids(test_token)

        if len(test_input_ids) > max_seq_length:
            continue

        # test_segment_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.chain_of_thought))
        test_segment_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(label))

        test_mask_ids = [1] * len(test_input_ids)

        padding = [tokenizer.eos_token_id] * (
            max_seq_length - len(test_input_ids)
        )
        test_input_ids = padding + test_input_ids
        test_mask_ids = [0] * len(padding) + test_mask_ids
        test_segment_ids = [tokenizer.eos_token_id] * (max_seq_length - len(test_segment_ids)) + test_segment_ids


        assert len(test_input_ids) == max_seq_length
        assert len(test_mask_ids) == max_seq_length
        assert len(test_segment_ids) == max_seq_length

        # ------------------------------------------

        tokens_label = tokenizer.tokenize(label)

        if len(tokens_a) + len(tokens_label) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length-len(tokens_label)]
            segment_ids = segment_ids[:max_seq_length-len(tokens_label)]

        tokens = tokens_a + tokens_label

        segment_ids = segment_ids + [2] * len(tokens_label)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)


        mask_ids = [1] * len(input_ids)
        padding = [tokenizer.eos_token_id] * (
            max_seq_length - len(input_ids)
        )
        input_ids += padding
        mask_ids += [0] * len(padding)
        segment_ids += [0] * len(padding)


        assert len(input_ids) == max_seq_length
        assert len(mask_ids) == max_seq_length
        assert len(segment_ids) == max_seq_length


        features.append(
            InputFeatures(
                input_ids=input_ids,
                mask_ids=mask_ids,
                segment_ids=segment_ids,
                guid=example.guid + " " + str(example.text_b),
                input_ids_2=test_input_ids,
                segment_ids_2=test_segment_ids,
                mask_ids_2=test_mask_ids
            )
        )
    print(len(features))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def seq_accuracy(preds, labels):
    acc = []
    for idx, pred in enumerate(preds):
        acc.append((pred == labels[idx]).mean())
    return acc.mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def all_metrics(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    pre = precision_score(y_true=labels, y_pred=preds)
    rec = recall_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
    }


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return all_metrics(preds, labels)


processors = {
    "vua": VUAProcessor,
}

output_modes = {
    "vua": "classification",
    "trofi": "classification",
}
