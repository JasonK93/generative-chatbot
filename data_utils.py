from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import json
import logging
import os
import random
import re
from io import open
import config
import numpy as np


# set the usual logging config I used before
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)


def make_dir(path):
    """
    make a directory if there is not one existing
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)


# Step1 : build vocab for the exist file, encoder and decoder have different vocab
# consider the condition
def build_vocal(filename):
    # input file path
    in_path = os.path.join(config.DATA_PATH, filename)

    # output file path
    out_path = os.path.join(config.DATA_PATH, "vocab.{}".format(filename[-3:]))

    # initial one diction for vocab
    vocab = {}

    # build vocab
    with open(in_path,encoding='utf-8') as f:
        for line in f.readlines():
            for token in basic_tokenizer(line):
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1
    # sort the vocab by frequence
    # sorted_vocab <type "list">
    sorted_vocab = sorted(vocab, key= vocab.get, reverse=True)

    with open(out_path, 'w', encoding = 'utf-8') as f:
        # <pad> means padding blank
        f.write("<pad>" + "\n")
        # <unk> means unknown token
        f.write("<unk>" + "\n")
        # <s> means start of the sentence
        f.write("<s>" + "\n")
        # <\s> means end of the sentence
        f.write("<\s>" + "\n")
        vocab_scale = 4

        # if the frequency less than the threshold, the word should be drop out
        for word in sorted_vocab:
            if vocab[word] < config.TERMINAL_OUTPUT:
                return vocab_scale
            f.write(word + "\n")
            vocab_scale +=1
        return vocab_scale
# Step1-1 : change every line into tokens which can be set into vocab
def basic_tokenizer(line, normalize_digits= True):
    # initial a list to store words in line
    words =[]
    _word_split = re.compile(u"([.,!?\"'-<>:;)(])")
    _digit_re = re.compile(r"\d")

    for fragment in line.strip().lower().split():
        for token in re.split(_word_split,fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_digit_re, b'#', token)
            words.append(token)
        return words


def process_data()
    logging.info("Processing raw data ......")
    logging.info("Step 1: Building vocabulary for encoder inputs ......")
    enc_vocab_size = build_vocab("train.enc")

if __name__ == '__main__':
    process_data()