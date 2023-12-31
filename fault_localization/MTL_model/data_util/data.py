import glob
import random
import struct
import csv
from tensorflow.core.example import example_pb2

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' 
UNKNOWN_TOKEN = '[UNK]' 
START_DECODING = '[START]' 
STOP_DECODING = '[STOP]' 


class Vocab(object):

  def __init__(self, vocab_file, max_size):
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0 

    for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
      self._word_to_id[w] = self._count   
      self._id_to_word[self._count] = w   
      self._count += 1

    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:   
        pieces = line.split()   
        if len(pieces) != 2:
          print( 'Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
          continue
        w = pieces[0] 

        if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
          raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w) 

        if w in self._word_to_id: 
          raise Exception('Duplicated word in vocabulary file: %s' % w)

        self._word_to_id[w] = self._count  
        self._id_to_word[self._count] = w  
        self._count += 1

        if max_size != 0 and self._count >= max_size:  
          print( "max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count) )
          break

    print( "Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))

  def word2id(self, word):   
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def id2word(self, word_id): 
    if word_id not in self._id_to_word:
      raise ValueError('Id not found in vocab: %d' % word_id)
    return self._id_to_word[word_id]

  def size(self):  
    return self._count

  def write_metadata(self, fpath): 
    print ("Writing word embedding metadata file to %s..." % (fpath))
    with open(fpath, "w") as f:
      fieldnames = ['word']
      writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
      for i in xrange(self.size()):
        writer.writerow({"word": self._id_to_word[i]}) 


def example_generator(data_path, single_pass): 
  while True:
    filelist = glob.glob(data_path) 
    assert filelist, ('Error: Empty filelist at %s' % data_path) 
    if single_pass:   
      filelist = sorted(filelist)
    else:
      random.shuffle(filelist)
    data_len = 0
    for f in filelist:
      reader = open(f, 'rb')  
      while True:
        len_bytes = reader.read(8)
        if not len_bytes: break 
        str_len = struct.unpack('q', len_bytes)[0]  
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]  
        yield example_pb2.Example.FromString(example_str)   
        data_len = data_len + 1
    if single_pass:
      print ("example_generator completed reading all datafiles. No more data.")
      print( "the data in the path of " + data_path + "lens is...." +str( data_len )  )
      break


def beforefix2ids(beforefix_words, vocab):  
  ids = []
  oovs = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN) 
  for w in beforefix_words:
    i = vocab.word2id(w)
    if i == unk_id:
      if w not in oovs:
        oovs.append(w) 
      oov_num = oovs.index(w) 
      ids.append(vocab.size() + oov_num) 
    else:
      ids.append(i) 
  return ids, oovs  


def afterfix2ids(afterfix_words, vocab, beforefix_oovs): 
  ids = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in afterfix_words:
    i = vocab.word2id(w)
    if i == unk_id:
      if w in beforefix_oovs: 
        vocab_idx = vocab.size() + beforefix_oovs.index(w) 
        ids.append(vocab_idx)  
      else: 
        ids.append(unk_id) 
    else:
      ids.append(i)
  return ids


def outputids2words(id_list, vocab, beforefix_oovs):  
  words = []
  for i in id_list:
    try:
      w = vocab.id2word(i) 
    except ValueError as e: 
      assert beforefix_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
      beforefix_oov_idx = i - vocab.size() 
      try:
        w = beforefix_oovs[beforefix_oov_idx]
      except ValueError as e: 
        raise ValueError('Error: model produced word ID %i which corresponds to beforefix OOV %i but this example only has %i beforefix OOVs' % (i, beforefix_oov_idx, len(beforefix_oovs)))
    words.append(w)
  return words


def afterfix2sents(afterfix): 
  cur = 0
  sents = []
  while True:
    try:
      start_p = afterfix.index(SENTENCE_START, cur)
      end_p = afterfix.index(SENTENCE_END, start_p + 1)
      cur = end_p + len(SENTENCE_END)
      sents.append(afterfix[start_p+len(SENTENCE_START):end_p])
    except ValueError as e: 
      return sents


def show_art_oovs(beforefix, vocab):  
  unk_token = vocab.word2id(UNKNOWN_TOKEN)
  words = beforefix.split(' ')
  words = [("__%s__" % w) if vocab.word2id(w)==unk_token else w for w in words]
  out_str = ' '.join(words)
  return out_str


def show_abs_oovs(afterfix, vocab, beforefix_oovs): 
  unk_token = vocab.word2id(UNKNOWN_TOKEN)
  words = afterfix.split(' ')
  new_words = []
  for w in words:
    if vocab.word2id(w) == unk_token: 
      if beforefix_oovs is None: 
        new_words.append("__%s__" % w)
      else: 
        if w in beforefix_oovs:
          new_words.append("__%s__" % w)
        else:
          new_words.append("!!__%s__!!" % w)
    else: 
      new_words.append(w)
  out_str = ' '.join(new_words)
  return out_str
