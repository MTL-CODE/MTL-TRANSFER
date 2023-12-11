
from __future__ import unicode_literals, print_function, division

import os
import time
import sys

import tensorflow as tf
import torch

from data_util import config_MutateMethodInvExpr

from data_util.batcher import Batcher
from data_util.data import Vocab

from data_util.utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch
from model import Model

use_cuda = config_MutateMethodInvExpr.use_gpu and torch.cuda.is_available()
print(" is use GPU ?",use_cuda)


class Evaluate(object):
    def __init__(self, trian_model):
        self.vocab = Vocab(config_MutateMethodInvExpr.vocab_path, config_MutateMethodInvExpr.vocab_size)
        self.batcher = Batcher(config_MutateMethodInvExpr.eval_data_path, self.vocab, mode='eval',
                               batch_size=config_MutateMethodInvExpr.batch_size, single_pass=False)
        time.sleep(15)
        # model_name = os.path.basename(model_file_path)

        eval_dir = os.path.join(config_MutateMethodInvExpr.log_root, 'eval_%s' % (trian_model))
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)
        self.summary_writer = tf.summary.FileWriter(eval_dir)

        # self.model = Model(model_file_path, is_eval=True)
        self.model = trian_model

    def eval_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, enc_labels   = get_input_from_batch(batch, use_cuda)
        
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = get_output_from_batch(batch, use_cuda)
        
        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        mlp_output = self.model.mlp(encoder_outputs)

        mlp_output = torch.tensor(mlp_output)
        enc_labels = torch.tensor(enc_labels)
        loss_function = torch.nn.CrossEntropyLoss(reduction='sum')
        mlp_loss = loss_function(mlp_output, enc_labels)


        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config_MutateMethodInvExpr.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1,attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config_MutateMethodInvExpr.eps)
            if config_MutateMethodInvExpr.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config_MutateMethodInvExpr.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_step_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_step_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)
        nmt_loss = loss

        mlp_weight = config_MutateMethodInvExpr.mlp_weight
        nmt_weight = config_MutateMethodInvExpr.nmt_weight
        new_loss = mlp_loss*mlp_weight + nmt_loss * nmt_weight
        loss = new_loss

        _,mlp_predicted = torch.max(mlp_output,1)
        mlp_total_acc = (mlp_predicted == enc_labels).sum().item()
        mlp_batch_len = len(enc_labels)
        mlp_batch_acc = mlp_total_acc/mlp_batch_len



        return nmt_loss.item(), mlp_batch_acc

    def run_eval(self):
        running_avg_loss, iter = 0, 0
        start = time.time()

        all_iter_acc = 0
        
        nmt_epoch_loss = 0

        iter = 0
        n_iters = config_MutateMethodInvExpr.eval_niters 

        while iter < n_iters:  #
            batch = self.batcher.next_batch()

            nmt_loss, mlp_batch_acc = self.eval_one_batch(batch)
            iter += 1
            all_iter_acc =  all_iter_acc + mlp_batch_acc
            
            nmt_epoch_loss = nmt_epoch_loss + nmt_loss
        

        eval_mlp_acc = all_iter_acc/n_iters
        nmt_epoch_loss = nmt_epoch_loss/n_iters
        

        

        return eval_mlp_acc,  nmt_epoch_loss

if __name__ == '__main__':
    
    trian_model = Model()
   
    eval_processor = Evaluate(trian_model)
    # eval_processor.run_eval()


