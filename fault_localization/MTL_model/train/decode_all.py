from __future__ import unicode_literals, print_function, division

import sys
import os
import time

import torch
from torch.autograd import Variable

from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util import data, config_all
from model import Model
from data_util.utils import write_for_rouge, rouge_eval, rouge_log
from decode_util import get_input_from_batch


use_cuda = config_all.use_gpu and torch.cuda.is_available()

class Beam(object):
  def __init__(self, tokens, log_probs, state, context, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage

  def extend(self, token, log_prob, state, context, coverage):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context,
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, decode_data_filename):
        # log_name = os.path.basename(decode_data_filename)
        self._decode_dir = os.path.join(decode_data_filename, 'decode_all' )
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')    
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')    
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        time.sleep(15)

        # self.model = Model(model_file_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def decode(self):
        faultTypes = ["InsertMissedStmt", "InsertNullPointerChecker", "MoveStmt", "MutateConditionalExpr",
                      "MutateDataType", "MutateLiteralExpr", "MutateMethodInvExpr", "MutateOperators",
                      "MutateReturnStmt", "MutateVariable", "RemoveBuggyStmt"]

        decode_alltype_dict = {}

        for type in faultTypes:

            vocab_path = "../data/data-new/code_data_demo_{}/finished_files/vocab".format(type)
            self.vocab = Vocab(vocab_path, config_all.vocab_size)
  
            d4j_data_path = "../data/data-new/code_data_demo_{}/finished_files/d4j_test.bin".format(type)
            self.batcher = Batcher(d4j_data_path, self.vocab, mode='decode',
                               batch_size=config_all.beam_size, single_pass=True)

            start = time.time()
            counter = 0
            batch = self.batcher.next_batch() 
            decode_onetype_all = []

            while batch is not None:
                    
                    model_path = "../data-new/code_data_demo_{}/Log/save_model/model/model_nmt_best".format(type)
                    self.model = Model(model_path, is_eval=True)



                    best_summary = self.beam_search(batch)  


                    output_ids = [int(t) for t in best_summary.tokens[1:]]
                    decoded_words = data.outputids2words(output_ids, self.vocab,
                                                         (batch.art_oovs[0] if config_all.pointer_gen else None))

                    original_abstract_sents = batch.original_abstracts_sents[0]
                    original_sents_sents = [w for w in original_abstract_sents]
                    decode_onetype_all.append(original_sents_sents)

                    decoded_sents = ' '.join(decoded_words)
                    decoded_sents = [type + ":  "+str(decoded_sents)]
                    decode_onetype_all.append(decoded_sents)

                    counter += 1
                    if counter % 10 == 0:
                        print('%d example in %d sec'%(counter, time.time() - start))
                        start = time.time()

                    batch = self.batcher.next_batch()
                
            decode_alltype_dict[type] = decode_onetype_all

        result11 = []

        len1 =  len(decode_alltype_dict['InsertMissedStmt'])
        print(len1)
        for i in range( 0,len1 ,2 ):
            print(i)
            every_result = []
            every_result.append(decode_alltype_dict['InsertMissedStmt'][i] )
            every_result.append(decode_alltype_dict['InsertMissedStmt'][i+1])
            every_result.append(decode_alltype_dict['InsertNullPointerChecker'][i+1])
            every_result.append(decode_alltype_dict['MoveStmt'][i+1])
            every_result.append(decode_alltype_dict['MutateConditionalExpr'][i+1])
            every_result.append(decode_alltype_dict['MutateDataType'][i+1])
            every_result.append(decode_alltype_dict['MutateLiteralExpr'][i+1])
            every_result.append(decode_alltype_dict['MutateMethodInvExpr'][i+1])
            every_result.append(decode_alltype_dict['MutateOperators'][i+1])
            every_result.append(decode_alltype_dict['MutateReturnStmt'][i+1])
            every_result.append(decode_alltype_dict['MutateVariable'][i+1])
            every_result.append(decode_alltype_dict['RemoveBuggyStmt'][i+1])

            print(every_result)

            result11.append(every_result)

        print(result11[0],result11[1],len(result11),len(result11[1]))

        result_txt_file = os.path.join(decode_data_filename,"219_result_top222.txt")
        with open(result_txt_file,'w') as f:
            for i in range(len(result11) ):   
                for j in range(len(result11[i])): 
                    f.write(str(result11[i][j]))
                    f.write('\n')
                f.write("\n\n\n\n")





    def beam_search(self, batch):

        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = get_input_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)

        s_t_0 = self.model.reduce_state(encoder_hidden)  

        dec_h, dec_c = s_t_0 
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = c_t_0[0],
                      coverage=(coverage_t_0[0] if config_all.is_coverage else None))
                 for _ in xrange(config_all.beam_size)]
        results = []
        steps = 0
        while steps < config_all.max_dec_steps and len(results) < config_all.beam_size:   
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t_1 = y_t_1.cuda()
            all_state_h =[]
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config_all.is_coverage:   
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)

            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config_all.beam_size * 2) 

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in xrange(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config_all.is_coverage else None)

                for j in xrange(config_all.beam_size * 2):  
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config_all.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config_all.beam_size or len(results) == config_all.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        if ( len(beams_sorted) == 1):
            beams_sorted.append( beams_sorted[0])
            beams_sorted.append( beams_sorted[0])
            beams_sorted.append( beams_sorted[0])
            beams_sorted.append( beams_sorted[0])
            beams_sorted.append( beams_sorted[0])
            beams_sorted.append( beams_sorted[0])
            beams_sorted.append( beams_sorted[0])
            beams_sorted.append( beams_sorted[0])
            beams_sorted.append( beams_sorted[0])
        if ( len(beams_sorted) == 2):
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
        if ( len(beams_sorted) == 3):
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
        if ( len(beams_sorted) == 4):
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
        if ( len(beams_sorted) == 5):
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
        if ( len(beams_sorted) == 6):
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
        if ( len(beams_sorted) == 7):
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
        if ( len(beams_sorted) == 8):
            beams_sorted.append(beams_sorted[0])
            beams_sorted.append(beams_sorted[0])
        if ( len(beams_sorted) == 9):
            beams_sorted.append(beams_sorted[0])



        return beams_sorted[1]

if __name__ == '__main__':
    # model_filename = sys.argv[1]
    decode_data_filename = "../data/data-new/code_data_demo_all/log"
    if not os.path.exists(decode_data_filename):
        os.makedirs(decode_data_filename)



    beam_Search_processor = BeamSearch(decode_data_filename)
    beam_Search_processor.decode()


