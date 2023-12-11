
import Queue
import time
from random import shuffle
from threading import Thread

import numpy as np
import tensorflow as tf

import config
import data

import random
random.seed(1234)


class Example(object):

  def __init__(self, beforefix, afterfix_sentences, label, vocab):  

    start_decoding = vocab.word2id(data.START_DECODING)
    stop_decoding = vocab.word2id(data.STOP_DECODING)

    beforefix_words = beforefix.split()
    if len(beforefix_words) > config.max_enc_steps: 
      beforefix_words = beforefix_words[:config.max_enc_steps]

    self.enc_len = len(beforefix_words) 
    self.enc_input = [vocab.word2id(w) for w in beforefix_words] 

    afterfix = ' '.join(afterfix_sentences) 
    afterfix_words = afterfix.split() 
    abs_ids = [vocab.word2id(w) for w in afterfix_words] 

    # Get the decoder input sequence and target sequence  
    self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding, stop_decoding)
    self.dec_len = len(self.dec_input)

    if config.pointer_gen:  
      self.enc_input_extend_vocab, self.beforefix_oovs = data.beforefix2ids(beforefix_words, vocab)

      abs_ids_extend_vocab = data.afterfix2ids(afterfix_words, vocab, self.beforefix_oovs)

      _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding, stop_decoding)

    # Store the original strings
    self.original_beforefix = beforefix
    self.original_afterfix = afterfix
    self.original_afterfix_sents = afterfix_sentences
    self.label = label


  def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len:
      inp = inp[:max_len]
      target = target[:max_len] 
    else: 
      target.append(stop_id) 
    assert len(inp) == len(target)
    return inp, target


  def pad_decoder_inp_targ(self, max_len, pad_id):
    while len(self.dec_input) < max_len:
      self.dec_input.append(pad_id)
    while len(self.target) < max_len:
      self.target.append(pad_id)


  def pad_encoder_input(self, max_len, pad_id):
    while len(self.enc_input) < max_len:
      self.enc_input.append(pad_id)
    if config.pointer_gen:
      while len(self.enc_input_extend_vocab) < max_len:
        self.enc_input_extend_vocab.append(pad_id)


class Batch(object):
  def __init__(self, example_list, vocab, batch_size):    
    self.batch_size = batch_size  
    self.pad_id = vocab.word2id(data.PAD_TOKEN) 
    self.init_encoder_seq(example_list) 
    self.init_decoder_seq(example_list) 
    self.store_orig_strings(example_list) 
    self.init_label_seq(example_list)

  def init_label_seq(self, example_list):    
    org_labels = [ex.label for ex in example_list]
    self.original_labels = org_labels


  def init_encoder_seq(self, example_list):
    max_enc_seq_len = max([ex.enc_len for ex in example_list])

    for ex in example_list:
      ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

    self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
    self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
    self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)

    for i, ex in enumerate(example_list):
      self.enc_batch[i, :] = ex.enc_input[:]
      self.enc_lens[i] = ex.enc_len
      for j in xrange(ex.enc_len):
        self.enc_padding_mask[i][j] = 1

    if config.pointer_gen:
      self.max_art_oovs = max([len(ex.beforefix_oovs) for ex in example_list])
      self.art_oovs = [ex.beforefix_oovs for ex in example_list]
      self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
      for i, ex in enumerate(example_list):
        self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

  def init_decoder_seq(self, example_list):
    for ex in example_list:
      ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

    self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
    self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
    self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
    self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

    for i, ex in enumerate(example_list):
      self.dec_batch[i, :] = ex.dec_input[:]
      self.target_batch[i, :] = ex.target[:]
      self.dec_lens[i] = ex.dec_len
      for j in xrange(ex.dec_len):
        self.dec_padding_mask[i][j] = 1

  def store_orig_strings(self, example_list):
    self.original_beforefixs = [ex.original_beforefix for ex in example_list] 
    self.original_afterfixs = [ex.original_afterfix for ex in example_list] 
    self.original_afterfixs_sents = [ex.original_afterfix_sents for ex in example_list] 


class Batcher(object):
  BATCH_QUEUE_MAX = 100 

  def __init__(self, data_path, vocab, mode, batch_size, single_pass):
    self._data_path = data_path  
    self._vocab = vocab  
    self._single_pass = single_pass  
    self.mode = mode  
    self.batch_size = batch_size 

    self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX) 
    self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)  

    if single_pass: 
      self._num_example_q_threads = 1 
      self._num_batch_q_threads = 1  
      self._bucketing_cache_size = 1 
      self._finished_reading = False
    else:
      self._num_example_q_threads = 1 
      self._num_batch_q_threads = 1
      self._bucketing_cache_size = 1 

    self._example_q_threads = []
    for _ in xrange(self._num_example_q_threads):  
      self._example_q_threads.append(Thread(target=self.fill_example_queue))  
      self._example_q_threads[-1].daemon = True
      self._example_q_threads[-1].start()
    self._batch_q_threads = []
    for _ in xrange(self._num_batch_q_threads):
      self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
      self._batch_q_threads[-1].daemon = True
      self._batch_q_threads[-1].start()

    if not single_pass: 
      self._watch_thread = Thread(target=self.watch_threads)
      self._watch_thread.daemon = True
      self._watch_thread.start()

  def next_batch(self):

    
    if self._batch_queue.qsize() == 0:
      tf.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
      if self._single_pass and self._finished_reading:
        tf.logging.info("Finished reading dataset in single_pass mode.")
        return None
    
    batch = self._batch_queue.get() 
    return batch

  def fill_example_queue(self):
    input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))   
                                         
    while True:

      try:
        (beforefix, afterfix,label) = input_gen.next() 
      except StopIteration: 
        tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
        if self._single_pass:
          tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
          self._finished_reading = True
          print("!!!!!!!")
          break
        else:
          raise Exception("single_pass mode is off but the example generator is out of data; error.")

      afterfix_sentences = [sent.strip() for sent in data.afterfix2sents(afterfix)] 
      example = Example(beforefix, afterfix_sentences, label, self._vocab)
      self._example_queue.put(example) 

  def fill_batch_queue(self):
    while True:
      if self.mode == 'decode':
        ex = self._example_queue.get()
        b = [ex for _ in xrange(self.batch_size)]
        self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
      else:
        inputs = []
        for _ in xrange(self.batch_size * self._bucketing_cache_size):
          inputs.append(self._example_queue.get())
        inputs = sorted(inputs, key=lambda inp: inp.enc_len, reverse=True) 

        batches = []
        for i in xrange(0, len(inputs), self.batch_size):
          batches.append(inputs[i:i + self.batch_size])
        if not self._single_pass:
          shuffle(batches)
        for b in batches:  
          self._batch_queue.put(Batch(b, self._vocab, self.batch_size))

  def watch_threads(self):
    while True:
      tf.logging.info(
        'Bucket queue size: %i, Input queue size: %i',
        self._batch_queue.qsize(), self._example_queue.qsize())

      time.sleep(60)
      for idx,t in enumerate(self._example_q_threads):
        if not t.is_alive(): 
          tf.logging.error('Found example queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_example_queue)
          self._example_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
      for idx,t in enumerate(self._batch_q_threads):
        if not t.is_alive(): 
          tf.logging.error('Found batch queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_batch_queue)
          self._batch_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()


  def text_generator(self, example_generator):
    while True:
      e = example_generator.next() 
      try:
        beforefix_text = e.features.feature['beforefix'].bytes_list.value[0] 
        afterfix_text = e.features.feature['afterfix'].bytes_list.value[0] 
        label_text = e.features.feature['label'].bytes_list.value[0]
      except ValueError:
        tf.logging.error('Failed to get beforefix or afterfix or label from example')
        continue
      if len(beforefix_text)==0: 
        continue
      else:
        yield (beforefix_text, afterfix_text,label_text)
