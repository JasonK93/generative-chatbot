from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import random
import sys
import time
from io import open

import numpy as np
import tensorflow as tf

import config
import data_utils
from model import ChatBotModel

# set the usual congig I used before
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)


# give the log level without warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def _get_buckets():
    """
    Load the dataset into buckets based on their lengths.
    train_buckets_scale is the interval that"ll help us
    choose a random bucket later on.
    """
    # test set
    test_buckets = data_utils.load_data("test_ids.enc", "test_ids.dec")
    # training set
    data_buckets = data_utils.load_data("train_ids.enc", "train_ids.dec")
    # Count the number of conversation pairs for each bucket.
    train_bucket_sizes = [len(data_buckets[b]) for b in range(len(config.BUCKETS))]
    print("Number of samples in each bucket:\n", train_bucket_sizes)
    # Total number of conversation pairs.
    train_total_size = sum(train_bucket_sizes)
    # list of increasing numbers from 0 to 1 that we"ll use to select a bucket.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    print("Bucket scale:\n", train_buckets_scale)
    return test_buckets, data_buckets, train_buckets_scale

def train():
    """
    Train the bot.
    """
    # test_buckets, data_buckets: <type "list">:
    #     [[[[Context], [Response]], ], ]]
    #     test_buckets[0]: first bucket
    #     test_buckets[0][0]: first pair of the first bucket
    #     test_buckets[0][0][0], test_buckets[0][0][1]: Context and response
    #     test_buckets[0][0][0][0]: word index of the first words
    # train_buckets_scale: list of increasing numbers from 0 to 1 that
    #     we"ll use to select a bucket. len(train_buckets_scale) = len(BUCKETS)
    test_buckets, data_buckets, train_buckets_scale = _get_buckets()

    # in train mode, we need to create the backward path, so forward_only is False
    model = ChatBotModel(False, config.BATCH_SIZE)
    # build graph
    model.build_graph()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("Running session...")
        sess.run(tf.global_variables_initializer())
        check_restore_parameters(sess, saver)

        iteration = model.global_step.eval()
        total_loss = 0
        logging.info("Training...")
        try:
            while True:
                skip_step = _get_skip_step(iteration)
                bucket_id = _get_random_bucket(train_buckets_scale)
                encoder_inputs, decoder_inputs, decoder_masks = data_utils.get_batch(
                    data_buckets[bucket_id], bucket_id,
                    batch_size=config.BATCH_SIZE)
                start = time.time()
                _, step_loss, _ = run_step(
                    sess, model, encoder_inputs, decoder_inputs,
                    decoder_masks, bucket_id, False)
                total_loss += step_loss
                iteration += 1

                if iteration % skip_step == 0:
                    logging.info("Training @ iter {:d}: loss {:.4f}, time {:.4f}".format(
                        iteration, total_loss / skip_step, time.time() - start))
                    total_loss = 0
                    saver.save(sess, os.path.join(config.CPT_PATH, "chatbot"),
                               global_step=model.global_step)
                    if iteration % (10 * skip_step) == 0:
                        logging.info("Testing...")
                        # Run evals on development set and print their loss
                        _eval_test_set(sess, model, test_buckets)
                    sys.stdout.flush()
        except KeyboardInterrupt:
            logging.info("Training interrupted.")


def main():
    # set the different parser to change the mode of train/chat
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices={"train", "chat"},
                        default="train",
                        help="mode. if not specified, it's in the train mode")
    args = parser.parse_args()

    # check the data pre-processing
    if not os.path.exists(os.path.join(config.DATA_PATH, "test_ids.dec")):
        data_utils.process_data()
    logging.info("Data ready!")
    # create checkpoints folder if there isn't one already
    data_utils.make_dir(config.CPT_PATH)

    if args.mode == "train":
        train()
    elif args.mode == "chat":
        chat()



