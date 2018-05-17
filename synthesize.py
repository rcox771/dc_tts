# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
original code By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts

adapted by Russ Cox for easy text2speech conversion
'''

from __future__ import print_function

import os

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_data, load_vocab, text_normalize
from scipy.io.wavfile import write
from tqdm import tqdm
import time


class Synth:

    def __init__(self, log_dir="log_dir", sample_dir="samples"):
        self._sess_loaded = False
        self.log_dir = log_dir
        self.sample_dir = sample_dir
        char2idx, idx2char = load_vocab()
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.g = Graph(mode="synthesize")
        print("Graph loaded")
        self.load_session()


    def prep_text(self, text):
        lines = text.split('\n')
        sents = [text_normalize(line).strip() + "E" for line in lines]
        texts = np.zeros((len(sents), hp.max_N), np.int32)
        for i, sent in enumerate(sents):
            texts[i, :len(sent)] = [self.char2idx[char] for char in sent]
        return texts

    def load_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(self.sess,
                       tf.train.latest_checkpoint(os.path.join(self.log_dir, "{}-1".format("LJ01"))))
        print("Text2Mel Restored!")

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(self.sess,
                       tf.train.latest_checkpoint(os.path.join(self.log_dir, "{}-2".format("LJ01"))))
        print("SSRN Restored!")
        self._sess_loaded = True

    def close_session(self):
        self.sess.close()
        self._sess_loaded = False
        print('closed session')


    def synthesize(self, text):
        if not os.path.exists(self.sample_dir): os.makedirs(self.sample_dir)

        if not self._sess_loaded:
            self.load_session()

        text = text.strip().lower()
        fn = os.path.join(self.sample_dir, "{}.wav".format(str(hash(text)).replace('-','_')))

        if os.path.isfile(fn):
            return fn
        start = time.time()
        L = self.prep_text(text)

        # Feed Forward
        ## mel
        Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)
        for j in tqdm(range(hp.max_T)):
            _gs, _Y, _max_attentions, _alignments = \
                self.sess.run([self.g.global_step, self.g.Y, self.g.max_attentions, self.g.alignments],
                         {self.g.L                  : L,
                          self.g.mels               : Y,
                          self.g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]

        # Get magnitude
        Z = self.sess.run(self.g.Z, {self.g.Y: Y})

        # Generate wav files

        for i, mag in enumerate(Z):
            print("Working on file {}:{}".format(fn, i))
            wav = spectrogram2wav(mag)
            write(fn, hp.sr, wav)
        stop = time.time()
        print("{} seconds elapsed".format(stop-start))




if __name__ == '__main__':
    synth = Synth()


    import json

    with open(os.path.join("data", "common.json"), 'r') as f:
        data = json.loads(f.read())
    for k in data:
        for sub_k in data[k]:
            for quote in data[k][sub_k]:
                synth.synthesize(quote)

    synth.close_session()



