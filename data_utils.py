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


# tool-1 make dir
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
def build_vocab(filename):
    # FIXME: write some rule to fix tokens such as didn -> didn't, or some single letter
    """
    Get a list of vocab which satisfied the threshold --->txt
    :param filename:
    :return: the scale of vocab
    """

    # input file path
    in_path = os.path.join(config.DATA_PATH, filename)

    # output file path
    out_path = os.path.join(config.DATA_PATH, "vocab.{}".format(filename[-3:]))

    # initial one dictionary for vocab
    vocab = {}

    # build vocab
    with open(in_path, encoding='utf-8') as f:
        for line in f.readlines():
            for token in basic_tokenizer(line):
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1
    # sort the vocab by frequency
    # sorted_vocab <type "list">
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)

    with open(out_path, 'w', encoding='utf-8') as f:
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
            if vocab[word] < config.THRESHOLD:
                return vocab_scale
            f.write(word + "\n")
            vocab_scale +=1
    return vocab_scale


# Step1-1 : change every line into tokens which can be set into vocab
def basic_tokenizer(line, normalize_digits= True):
    """
    change line into tokens list
    :param line: exp'I love U'
    :param normalize_digits: True means normalize
    :return:['I', 'love', 'U']
    """
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


# Step 2: token2id
def token2id(data, mode):

    vocab_path = 'vocab' + '.' + mode
    in_path = data + "." + mode
    out_path = data + "_ids." + mode

    # get the one to one pair
    _, vocab = load_vocab(os.path.join(config.DATA_PATH, vocab_path))
    in_file = open(os.path.join(config.DATA_PATH, in_path),
                   encoding="utf-8")
    out_file = open(os.path.join(config.DATA_PATH, out_path), "w")
    lines = in_file.read().splitlines()
    # `lines` is a list of sentence strings, e.g., ["hello!", "how are you?"]
    in_file.close()
    for line in lines:
        if mode == "dec":  # we only care about <s> and </s> in decoder
            ids = [vocab["<s>"]]
        else:
            ids = []
        ids.extend(sentence2id(vocab, line))
        if mode == "dec":
            ids.append(vocab["<\s>"])
        out_file.write(" ".join(str(id_) for id_ in ids) + "\n")
    out_file.close()


# Step2-1 : load vocab
def load_vocab(vocab_path):
    """
    Load vocabulary.
    Returns:
        `words` is a list of vocab strings.
        `word2id` is a dict of <word_str, word_id> pairs.
    """
    # all vocab words in one list
    with open(vocab_path, encoding="utf-8") as f:
        words = f.read().splitlines()
    # dict generator
    word2id = {words[i]: i for i in range(len(words))}
    return words, word2id


# Step2-2 : change sentence2id array
def sentence2id(vocab, line):
    """
    Convert a sentence string to word id list.
    :param vocab: word2id
    :type vocab: dict
    :param line: a raw sentence
    :return: list of word indices
    """
    # Get word index or get the index of <unk>.
    return [vocab.get(token, vocab["<unk>"]) for token in basic_tokenizer(line)]


# Step 1+2
def process_data():
    logging.info("Processing raw data ......")
    logging.info("Step 1: Building vocabulary for encoder inputs ......")
    enc_vocab_size = build_vocab("train.enc")
    logging.info("Step 2: Building vocabulary for decoder inputs ......")
    dec_vocab_size = build_vocab("train.dec")
    vocab_size = {"encoder": enc_vocab_size, "decoder": dec_vocab_size}
    with open(os.path.join(config.DATA_PATH, "vocab_size.json"),
              "w", encoding="utf-8") as f:
        f.write(json.dumps(vocab_size, ensure_ascii=False))

    logging.info("Step 3: Tokenizing encoder inputs of training data...")
    token2id("train", "enc")
    logging.info("Step 4:Tokenizing decoder inputs of training data...")
    token2id("train", "dec")
    logging.info("Step 5:Tokenizing encoder inputs of test data...")
    token2id("test", "enc")
    logging.info("Step 6:Tokenizing decoder inputs of test data...")
    token2id("test", "dec")


# GROUP data into buckets
def load_data(enc_filename, dec_filename):
    """
    Load data from *_ids.* files and group the data into buckets.
    Args:
        :param enc_filename: "train_ids.enc", etc.
        :param dec_filename: "train_ids.dec", etc.
    Return:
        `data_buckets` is a list of lists. Each list is a bucket,
        and each bucket contains many <context, response> pairs.
    """
    encode_file = open(os.path.join(config.DATA_PATH, enc_filename))
    decode_file = open(os.path.join(config.DATA_PATH, dec_filename))
    encode, decode = encode_file.readline(), decode_file.readline()
    data_buckets = [[] for _ in config.BUCKETS]
    i = 0
    while encode and decode:
        if (i + 1) % 10000 == 0:
            print("Bucketing conversation number", i + 1)
        # covert digit string to integer
        encode_ids = [int(id_) for id_ in encode.split()]
        decode_ids = [int(id_) for id_ in decode.split()]
        for bucket_id, (encode_max_size, decode_max_size) in enumerate(config.BUCKETS):
            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break
        # Pairs with too long context or utterance should be dropped.
        encode, decode = encode_file.readline(), decode_file.readline()
        i += 1
    return data_buckets


if __name__ == '__main__':
    process_data()
