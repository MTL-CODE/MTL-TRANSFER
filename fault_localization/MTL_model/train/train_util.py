
from torch.autograd import Variable
import numpy as np
import torch
from data_util import config

def get_input_from_batch(batch, use_cuda):
  batch_size = len(batch.enc_lens)
  



  enc_batch = Variable(torch.from_numpy(batch.enc_batch).long()) 
  enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
  enc_lens = batch.enc_lens
  extra_zeros = None
  enc_batch_extend_vocab = None

  enc_labels = batch.original_labels    # string
  enc_labels = map(int, enc_labels)      # string --> int
  enc_labels = np.array(enc_labels)       #  int list --> int numpy
  enc_labels = Variable(torch.from_numpy(enc_labels).long())     # int -->  tensorlong 




  if config.pointer_gen:
    enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
    if batch.max_art_oovs > 0:
      extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))

  c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))  # c_t_1


  coverage = None
  if config.is_coverage:
    coverage = Variable(torch.zeros(enc_batch.size()))

  if use_cuda:
    enc_batch = enc_batch.cuda()
    enc_padding_mask = enc_padding_mask.cuda()
    enc_labels = enc_labels.cuda()

    if enc_batch_extend_vocab is not None:
      enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
    if extra_zeros is not None:
      extra_zeros = extra_zeros.cuda()
    c_t_1 = c_t_1.cuda()

    if coverage is not None:
      coverage = coverage.cuda()

  return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, enc_labels

def get_output_from_batch(batch, use_cuda):

  dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
  dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()
  dec_lens = batch.dec_lens
  max_dec_len = np.max(dec_lens)
  dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()

  target_batch = Variable(torch.from_numpy(batch.target_batch)).long()

  if use_cuda:
    dec_batch = dec_batch.cuda()
    dec_padding_mask = dec_padding_mask.cuda()
    dec_lens_var = dec_lens_var.cuda()
    target_batch = target_batch.cuda()


  return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

