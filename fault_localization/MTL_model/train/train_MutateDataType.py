
from __future__ import unicode_literals, print_function, division

import os
import time
import argparse
import eval_MutateDataType

import tensorflow as tf
import torch
from model import Model
from torch.nn.utils import clip_grad_norm_

from torch.optim import Adagrad

from data_util import config_MutateDataType
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch

import sys


use_cuda = config_MutateDataType.use_gpu and torch.cuda.is_available()
print("is use GPU ?", use_cuda)


class Train(object):
    def __init__(self):
        self.vocab = Vocab(config_MutateDataType.vocab_path, config_MutateDataType.vocab_size)
        self.batcher = Batcher(config_MutateDataType.train_data_path, self.vocab, mode='train',
                               batch_size=config_MutateDataType.batch_size,
                               single_pass=False) 

        time.sleep(15)

        if not os.path.exists(config_MutateDataType.log_root):
            os.mkdir(config_MutateDataType.log_root)
        train_dir = os.path.join(config_MutateDataType.log_root, 'save_model')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.FileWriter(train_dir)  

    def save_model(self, running_avg_loss, iter, str1):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'mlp_state_dict': self.model.mlp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%s_best' % (str1))
        torch.save(state, model_save_path)
        print("save this model to：", model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)  

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters()) + list(self.model.mlp.parameters())
        initial_lr = config_MutateDataType.lr_coverage if config_MutateDataType.is_coverage else config_MutateDataType.lr  # 0.15
        self.optimizer = Adagrad(params, lr=initial_lr,
                                 initial_accumulator_value=config_MutateDataType.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config_MutateDataType.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):
        

        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, enc_labels = get_input_from_batch(
            batch, use_cuda)
        

        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = get_output_from_batch(batch, use_cuda)
       

        self.optimizer.zero_grad()

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        mlp_output = self.model.mlp(encoder_outputs)
        
        mlp_output = torch.tensor(mlp_output)
        enc_labels = torch.tensor(enc_labels)
        loss_function = torch.nn.CrossEntropyLoss(reduction='sum')
        mlp_loss = loss_function(mlp_output, enc_labels)
        

        s_t_1 = self.model.reduce_state(encoder_hidden)  

        step_losses = []
        for di in range(min(max_dec_len, config_MutateDataType.max_dec_steps)):  
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                                                           encoder_outputs,
                                                                                           encoder_feature,
                                                                                           enc_padding_mask, c_t_1,
                                                                                           extra_zeros,
                                                                                           enc_batch_extend_vocab,
                                                                                           coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config_MutateDataType.eps)
            if config_MutateDataType.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config_MutateDataType.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)
        nmt_loss = loss
       
        mlp_weight = config_MutateDataType.mlp_weight
        nmt_weight = config_MutateDataType.nmt_weight

        new_loss = mlp_loss * mlp_weight + nmt_loss * nmt_weight
        loss = new_loss
        # print("new_loss is .......", loss)
        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config_MutateDataType.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config_MutateDataType.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config_MutateDataType.max_grad_norm)
        clip_grad_norm_(self.model.mlp.parameters(), config_MutateDataType.max_grad_norm)

        self.optimizer.step()

        _, mlp_predicted = torch.max(mlp_output, 1)
        mlp_total_acc = (mlp_predicted == enc_labels).sum().item()
        mlp_batch_len = len(enc_labels)
        mlp_batch_acc = mlp_total_acc / mlp_batch_len

        return loss.item(), mlp_loss, nmt_loss, mlp_batch_acc

    def trainIters(self, n_iters, model_file_path=None):  
        iter = 0
        start = time.time()

        all_iter_acc = 0
        mlp_epoch_loss = 0
        nmt_epoch_loss = 0
        new_epoch_loss = 0

        while iter < n_iters:  
            batch = self.batcher.next_batch()
          

            loss_epoch, mlp_loss, nmt_loss, mlp_batch_acc = self.train_one_batch(
                batch)  
            iter += 1

            all_iter_acc = all_iter_acc + mlp_batch_acc
            mlp_epoch_loss = mlp_epoch_loss + mlp_loss
            nmt_epoch_loss = nmt_epoch_loss + nmt_loss
            new_epoch_loss = new_epoch_loss + loss_epoch

            if iter % 100 == 0:
                self.summary_writer.flush()  
            print_interval = 100
           

        mlp_acc_epoch = all_iter_acc / n_iters
        new_epoch_loss = new_epoch_loss / n_iters
        nmt_epoch_loss = nmt_epoch_loss / n_iters
        mlp_epoch_loss = mlp_epoch_loss / n_iters

        return mlp_acc_epoch, new_epoch_loss, nmt_epoch_loss, mlp_epoch_loss

    def Epochs_Train(self, epochs, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        best_mlp_acc = 0
        best_nmt_loss = 10000
        print("the proportion of mlp and nmt is.....{}/{}".format(config_MutateDataType.mlp_weight,
                                                                  config_MutateDataType.nmt_weight))
        print("\n")

        for i in range(epochs):
            start_time = time.time()


            self.model.encoder = self.model.encoder.train()
            self.model.decoder = self.model.decoder.train()
            self.model.reduce_state = self.model.reduce_state.train()
            self.model.mlp = self.model.mlp.train()

            print("[Epoch " + str(i + 1) + '/' + str(epochs) + ']')
            mlp_acc_epoch, new_epoch_loss, nmt_epoch_loss, mlp_epoch_loss = train_processor.trainIters(
                config_MutateDataType.max_iterations, args.model_file_path)
            print("train====train_mlp_acc is %.2f, nmt_loss is %f, mlp_loss is %f, all_loss is %f.  ...." % (
            mlp_acc_epoch, nmt_epoch_loss, mlp_epoch_loss, new_epoch_loss))

           
            eval_processor = eval_MutateDataType.Evaluate(train_processor.model)
            one_epoach_eval_acc, nmt_epoch_eval_loss = eval_processor.run_eval()
            print("eval====eval_mlp_acc is %.4f, eval_all_loss is %f ..........this epoch run_time is %d." %
                  (one_epoach_eval_acc, nmt_epoch_eval_loss, time.time() - start_time))

            if (one_epoach_eval_acc > best_mlp_acc):
                best_mlp_acc = one_epoach_eval_acc
                running_avg_loss = calc_running_avg_loss(nmt_epoch_eval_loss, running_avg_loss, self.summary_writer,
                                                         i)  
                self.save_model(running_avg_loss, i, "mlp")
                print("--MLP   Saving best model for MLP: epoch_{}".format(i + 1))

            if (nmt_epoch_eval_loss < best_nmt_loss):
                best_nmt_loss = nmt_epoch_eval_loss
                running_avg_loss = calc_running_avg_loss(nmt_epoch_eval_loss, running_avg_loss, self.summary_writer, i)
                self.save_model(running_avg_loss, i, "nmt")
                print("--NMT   Saving best model for NMT: epoch_{}".format(i + 1))
            print("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path",
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()

    train_processor = Train()

    train_processor.Epochs_Train(config_MutateDataType.epochs, config_MutateDataType.max_iterations,
                                 args.model_file_path)

    # train_processor.trainIters( config_MutateDataType.max_iterations, args.model_file_path) 
