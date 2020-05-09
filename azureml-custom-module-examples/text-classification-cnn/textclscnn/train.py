import logging
import os
import pickle
import sys

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from textclscnn.args_util import train_args
from textclscnn.data_util import load_data, process_data
from textclscnn.model import TextCNN


class Trainer():
    def __init__(self, args):
        self.args = args
        logging.info('processing data...')
        self.word2id, self.id2word, self.label2id, self.id2label, self.max_len \
            = process_data(self.args, self.args.train_file)

        # save the vocab dict the vocab path.
        if not os.path.isdir(self.args.vocab_path):
            os.makedirs(self.args.vocab_path)
        with open(self.args.vocab_path + '/' + 'word2id.pkl', 'wb') as f:
            pickle.dump(self.word2id, f)
        with open(self.args.vocab_path + '/' + 'label2id.pkl', 'wb') as f:
            pickle.dump(self.label2id, f)
        with open(self.args.vocab_path + '/' + 'id2label.pkl', 'wb') as f:
            pickle.dump(self.id2label, f)

        self.train_iter, self.test_iter = load_data(self.args.train_file, self.args.test_file, self.word2id,
                                                    self.label2id, self.args)
        self.args.embed_num = len(self.word2id)
        self.args.class_num = len(self.id2label)
        self.model = TextCNN(self.args)

        if self.args.snapshot is not None:
            logging.info('\nLoading model from %s...' % (self.args.snapshot))
            self.model.load_state_dict(torch.load(self.args.snapshot))

        if torch.cuda.is_available():
            print("GPU.............")
            torch.cuda.set_device(self.args.device)
            self.model = self.model.cuda()

    def train(self):
        if torch.cuda.is_available():
            self.model.cuda()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        step, best_acc, last_step = 0, 0, 0
        train_loss, train_acc = 0., 0.
        self.model.train()
        for epoch in range(1, self.args.epochs + 1):
            for feature, target in self.train_iter:
                feature = Variable(feature)
                target = Variable(target)
                if torch.cuda.is_available():
                    feature, target = feature.cuda(), target.cuda()
                optimizer.zero_grad()
                logit = self.model(feature)

                loss = F.cross_entropy(logit, target)
                l2_reg = 0.
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                loss += self.args.l2 * l2_reg
                loss.backward()
                optimizer.step()
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / self.args.batch_size
                step += 1
                train_loss += loss.data
                train_acc += accuracy
                sys.stdout.write('\rEpoch[%d] Step[%d] - loss: %f  acc: %f%% (%d/%d)'
                                 % (epoch, step, loss.data, accuracy, corrects, self.args.batch_size))
                if step % self.args.log_interval == 0:
                    train_loss /= self.args.log_interval
                    train_acc /= self.args.log_interval
                    train_loss, train_acc = 0., 0.

                if step % self.args.test_interval == 0:
                    logging.info('Epoch[%d] Step[%d] - loss: %f  acc: %f%% (%d/%d)'
                                 % (epoch, step, loss.data, accuracy, corrects, self.args.batch_size))
                    test_loss, test_acc = self.eval()

                    self.model.train()
                    if test_acc > best_acc:
                        best_acc, last_step = test_acc, step
                        if self.args.save_best:
                            self.save('best', last_step)
                            continue

    def eval(self):
        self.model.eval()
        corrects, avg_loss = 0, 0
        for feature, target in self.test_iter:
            feature = Variable(feature)
            target = Variable(target)
            if torch.cuda.is_available():
                feature, target = feature.cuda(), target.cuda()

            logit = F.softmax(self.model(feature), dim=1)
            loss = F.cross_entropy(logit, target, size_average=False)

            avg_loss += loss.data
            corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

        size = len(self.test_iter.dataset)
        avg_loss /= size
        accuracy = 100.0 * corrects / size
        logging.info('Evaluation - loss: %f  acc: %f (%d/%d)\n' % (avg_loss, accuracy, corrects, size))
        sys.stdout.write('\rEvaluation - loss: %f  acc: %f (%d/%d)\n' % (avg_loss, accuracy, corrects, size))
        return avg_loss, accuracy

    def save(self, save_prefix, steps):
        if not os.path.isdir(self.args.trained_model):
            os.makedirs(self.args.trained_model)
        save_prefix = os.path.join(self.args.trained_model, save_prefix)
        save_path = '%s_steps_%d.pt' % (save_prefix, 100)
        torch.save(self.model.state_dict(), save_path)


if __name__ == '__main__':
    args = train_args()
    if not os.path.isdir(args.trained_model):
        os.makedirs(args.trained_model)
    logname = os.path.join(args.trained_model, "train.log")
    logging.basicConfig(filename=logname, filemode='w', level=logging.DEBUG)
    trainer = Trainer(args)
    with open(os.path.join(args.trained_model, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        trainer.train()
    except KeyboardInterrupt:
        logging.warning('\nExiting from training early')
