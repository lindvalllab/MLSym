"""
Copyright 2022, Dana-Farber Cancer Institute
License: GNU GPL 2.0
"""
# import relavant libraries
import os
import collections
import pandas as pd
import numpy as np
import random
import itertools
import json
import argparse
import datetime
import time
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,
    BertTokenizer, BertForTokenClassification, 
    XLNetTokenizer, XLNetForTokenClassification, 
    RobertaTokenizer, RobertaForTokenClassification, RobertaTokenizer,
    XLMRobertaForTokenClassification, XLMRobertaTokenizer,
    CamembertForTokenClassification, CamembertTokenizer,
    DistilBertForTokenClassification, DistilBertTokenizer,
    ElectraTokenizer, ElectraForTokenClassification,
    #AutoTokenizer, AutoModelForTokenClassification,
    LongformerForTokenClassification, LongformerTokenizer
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorboardX import SummaryWriter
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--dset', type=str, default='data/all', help='name of dataset')
parser.add_argument('--seed', type=int, default=123, help='seed')
parser.add_argument('--lr', type=float, default=6e-5, help='learning rate')
parser.add_argument('--decay', type=float, default=0.01, help='weight decay ate')
parser.add_argument('--warmups', type=int, default=500, help='warmups')
parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adam')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batch', type=int, default=32, help='batch_size')
parser.add_argument('--model_class', type=str, default='electra', help='model class')
parser.add_argument('--pretrained_model', type=str, help='pretrained model name or model path')
parser.add_argument('--pretrained_tokenizer', type=str, help='pretrained model name or model path for tokenizer')
parser.add_argument('--ensemble', type=str, help='ensemble model class')
parser.add_argument('--gpu_id', type=int, nargs='+', default=[0,1])
parser.add_argument('--f1_loss', action='store_true')
parser.add_argument('--max_seq_length', type=int, default=512, help='maxium sequence length')
parser.add_argument('--early_stop', type=int, default=10, help='early stopping epochs')
args = parser.parse_args() 
dataset = args.dset
ensemble = args.ensemble
gpu_id = args.gpu_id
f1_loss = args.f1_loss
max_seq_length = args.max_seq_length
early_stop = args.early_stop

MODEL_CLASSES = {
    'bert': (BertForTokenClassification, BertTokenizer, 'bert-large-cased'),
    'clinicalbert': (BertForTokenClassification, BertTokenizer, 'emilyalsentzer/Bio_ClinicalBERT'),
    'xlnet': (XLNetForTokenClassification, XLNetTokenizer, 'xlnet-large-cased'),
    'clinicalxlnet': (XLNetForTokenClassification, XLNetTokenizer, 'xlnet-large-cased'),
    'roberta': (RobertaForTokenClassification, RobertaTokenizer, 'roberta-large'),
    'xlm-roberta': (XLMRobertaForTokenClassification, XLMRobertaTokenizer, 'xlm-roberta-large'),
    'camembert': (CamembertForTokenClassification, CamembertTokenizer, 'camembert-large'),
    'distilbert': (DistilBertForTokenClassification, DistilBertTokenizer, 'distilbert-base-cased'),
    'electra_small': (ElectraForTokenClassification, ElectraTokenizer, 'google/electra-small-discriminator'),
    'electra': (ElectraForTokenClassification, ElectraTokenizer, 'google/electra-large-discriminator'),
    'longformer': (LongformerForTokenClassification, LongformerTokenizer, 'allenai/longformer-base-4096'),
}

class TransformerNER():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_grad_norm = 1.0
        self.fine_tuning = True
        self.pad_token = 0
        self.pad_token_label_id = -100
        self.max_seq_length = max_seq_length
        self.early_stop = early_stop
        self.set_seed(args.seed)
        self.plot_cm = False
        self.plot_pr = False
        self.filepath = {'train': os.path.join(dataset, 'train.txt'),
                         'valid': os.path.join(dataset, 'valid.txt'),
                         'test':  os.path.join(dataset, 'test.txt')}  
        
    def set_seed(self, seed):
        """
        Set random seed for reproducibility
        """
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    def make_dir(self, path):
        """
        Create a directory in the path assigned
        """
        if not os.path.exists(path):
            os.makedirs(path)
            
    def prepare_data(self):
        """
        Read train/valid/test data from input_dir and 
        convert data to features (input_ids, label_ids, attention_masks)
        """

        self.data = {}
        self.tokenized_idx, self.tokenized_token = collections.defaultdict(list), collections.defaultdict(list)
        for mode in ['train', 'valid', 'test']:
            # read data and get sentences and labels
            with open(self.filepath[mode], 'r') as f:
                lines = f.readlines()
                sentences, labels, idx, sent, lab, id = [], [], [], [], [], []
                tags = set()
                for line in lines:
                    if '-DOCSTART-' in line or '</s>' in line  or '<s>' in line or line.rstrip()=='':
                        if sent and lab:
                            sentences.append(sent)
#                             if lab not in self.keep:
#                                 labels.append('O')
#                             else:
#                                 labels.append(lab)
                            labels.append(lab)
                            idx.append(id)
                        if '-DOCSTART-' in line:
                            sent, lab, id = ['D'], ['O'], ['D{}'.format(line.split()[-2])]
                        else:
                            sent, lab, id = [], [], []
                    else:
                        sent.append(line.split()[0])
                        id.append(line.split()[-2])
                        lab.append(line.split()[-1])  
                        tags.add(line.split()[-1])

            # label_map
            if mode == 'train':
                self.label2id = {t: i for i, t in enumerate(list(tags))}
                self.num_labels = len(self.label2id)
                self.label2id[self.pad_token] = self.pad_token_label_id
                self.id2label = {v: k for k, v in self.label2id.items()}
            # tokenize the sentences and save the start offset of each subwords
            tokenized_sentences, tokenized_labels, tokenized_idx, tokenized_toks= [], [], [], []
            
            for sent, label, id in zip(sentences, labels, idx):
                tokenized_sent, tokenized_lab, tokenized_id, tokenized_tok = [], [], [], []
                for word, lab, i in zip(sent, label, id):
                    tokenized_word = self.tokenizer.tokenize(word)
                    tokenized_sent.extend(tokenized_word)
                    tokenized_lab.extend([lab] * len(tokenized_word))
                    tokenized_id.extend([i] * len(tokenized_word))
                    tokenized_tok.extend([word] * len(tokenized_word))
                # truncate the subword tokens longer than maxium sequence length
                if len(tokenized_sent) > self.max_seq_length:
                    tokenized_sent = tokenized_sent[: self.max_seq_length]
                    tokenized_lab = tokenized_lab[: self.max_seq_length]
                    tokenized_id = tokenized_id[: self.max_seq_length]
                    tokenized_tok = tokenized_tok[: self.max_seq_length]
                tokenized_sentences.append(tokenized_sent)
                tokenized_labels.append(tokenized_lab)
                self.tokenized_idx[mode].append(tokenized_id)
                self.tokenized_token[mode].append(tokenized_tok)

            input_ids, label_ids, attention_masks = [], [], []
            i=0
            
            for sent, label in zip(tokenized_sentences, tokenized_labels):
                # get token's id and label's id
                input_id = self.tokenizer.convert_tokens_to_ids(sent)
                label_id = [self.label2id.get(lab) for lab in label]
                for k in label_id:
                    if isinstance(k, type(None)):
                        print(label)
                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to
                input_mask = [1] * len(input_id)
                # Zero-pad up to the sequence length (pad on right)
                padding_length = self.max_seq_length - len(input_id)
                input_id += [self.pad_token] * padding_length
                input_mask += [0] * padding_length
                label_id += [self.pad_token_label_id] * padding_length
                input_ids.append(input_id)
                label_ids.append(label_id)
                attention_masks.append(input_mask)
                i+=1
            self.data[mode] = TensorDataset(torch.tensor(input_ids), 
                                            torch.tensor(attention_masks), 
                                            torch.tensor(label_ids))

        # Save training parameters
        print('\ndset: %s, batch_size: %d, lr: %4f, weight_decay: %4f, warmups: %d'%(\
                self.dataset, self.batch_size, self.lr, self.weight_decay, self.warmups)) 
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(args.__dict__, f)

    def format_tags(self, predictions, true_labels, mode):
        """
        convert ids to original labels and create formatted output prediction
        """
        pred_tags, label_tags, out = [], [], []
        for prediction, true_label, token, id in zip(predictions, true_labels, self.tokenized_token[mode], self.tokenized_idx[mode]):
            for pred, gt, tok, i in zip(prediction, true_label, token, id):
                if self.id2label[gt] != self.pad_token:
                    pred_tags.append(self.id2label[pred])
                    label_tags.append(self.id2label[gt])
                    out.append('{} {} {} {}'.format(i, tok, self.id2label[gt], self.id2label[pred]))
            out.append('')
        
        return pred_tags, label_tags, out
    
    def trainer(self, parameterization, weight=None):
        # create output folder and tensorboard
        self.output_dir = 'processing/output/{}/{}/{}'.format(parameterization['model'], dataset.split('/')[-1],
                                                datetime.datetime.now().strftime('%m%d-%H%M%S'))
        self.model_dir = '{}/model'.format(self.output_dir)
        self.make_dir(self.output_dir)
        self.make_dir(self.model_dir)
        self.tsboard = {'train': SummaryWriter(os.path.join('tensorboard', parameterization['model'], 
                                                dataset.split('/')[-1]+'-train', 
                                                datetime.datetime.now().strftime('%m%d-%H%M%S'))),
                        'valid': SummaryWriter(os.path.join('tensorboard', parameterization['model'], 
                                                dataset.split('/')[-1]+'-valid', 
                                                datetime.datetime.now().strftime('%m%d-%H%M%S'))),
                        'test': SummaryWriter(logdir=os.path.join('tensorboard', parameterization['model'], 
                                                dataset.split('/')[-1]+'-test', 
                                                datetime.datetime.now().strftime('%m%d-%H%M%S')))}
        
        # load pretrained model and tokenizer
        self.model_class, self.tokenizer_class, pretrained_model = MODEL_CLASSES[parameterization['model']]
        self.pretrained_model = args.pretrained_model if args.pretrained_model else pretrained_model
        self.pretrained_tokenizer = args.pretrained_tokenizer if args.pretrained_tokenizer else self.pretrained_model
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_tokenizer)  
 
        # update parameters from optimization experiemts or arguments
        self.batch_size = parameterization['batch']
        self.n_epochs = parameterization['n_epochs']
        self.lr = parameterization['lr']
        self.weight_decay = parameterization['decay']
        self.warmups = parameterization['warmups']  
        self.eps = parameterization['eps']  
        self.dataset = parameterization['dset'] 
        
        # get datasets
        self.prepare_data()
        with open(f'{self.output_dir}/label2id.pkl', 'wb') as f:
            pickle.dump(self.label2id, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        train_data, valid_data, test_data = self.data['train'], self.data['valid'], self.data['test']
        train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=self.batch_size)
        valid_dataloader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), batch_size=self.batch_size)
        test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=self.batch_size)

        # load pretrained model and move to GPU
        model = self.model_class.from_pretrained(self.pretrained_model, num_labels=self.num_labels)
        model.to(self.device)
        model = nn.DataParallel(model, device_ids=gpu_id)

        # ensemble models
        if ensemble:
            ensemble_model_class, ensemble_tokenizer_class, ensemble_pretrained_model = MODEL_CLASSES[ensemble]  
            self.tokenizer = ensemble_tokenizer_class.from_pretrained(ensemble_pretrained_model)  
            train_data, valid_data, test_data = self.data['train'], self.data['valid'], self.data['test']
            train_dataloader_e = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=self.batch_size)
            valid_dataloader_e = DataLoader(valid_data, sampler=SequentialSampler(valid_data), batch_size=self.batch_size)
            test_dataloader_e = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=self.batch_size)

            model_e = ensemble_model_class.from_pretrained(ensemble_pretrained_model, num_labels=self.num_labels)
            model_e.to(self.device) 
        else:
            train_dataloader_e = train_dataloader
            valid_dataloader_e = valid_dataloader
            test_dataloader_e = test_dataloader
        
        # optimizer
        if self.fine_tuning:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': self.weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}]
        else:
            param_optimizer = list(model.classifier.named_parameters())
            optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.eps)

        # learning rate scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmups, num_training_steps=len(train_dataloader) * self.n_epochs)

        global_steps, valid_steps, test_steps = 0, 0, 0
        best_valid_f1, best_valid_epoch, best_test_f1, best_test_epoch = 0, 0, 0, 0
        best_val_loss, wait = float('inf'), 0
        
        for epoch in range(self.n_epochs):
            # --------------- Training --------------- 
            model.train()
            train_loss = 0
            predictions , true_labels, pr_pred, pr_label = [], [], [], []
            start = time.time()
            for batch, batch_e in tqdm(zip(train_dataloader, train_dataloader_e), total=len(train_dataloader), desc='train'):
                # move batch to gpu
                b_input_ids, b_input_mask, b_labels = tuple(b.to(self.device) for b in batch)

                model.zero_grad()

                # forward pass
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss, logits = outputs['loss'], outputs['logits']

                # average two model's logits and calculate the loss for ensemble model
                if ensemble:
                    e_input_ids, e_input_mask, e_labels = tuple(b.to(self.device) for b in batch_e)
                    model_e.zero_grad()
                    outputs_e = model_e(e_input_ids, attention_mask=e_input_mask, labels=e_labels)
                    logits = (outputs[1] + outputs_e[1])/2
                    loss = self.criterion(logits, b_input_mask, b_labels)
  
                # move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.detach().cpu().numpy()
                # store logits and labels of all batches
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.extend(label_ids)
                if self.plot_pr:
                    pr_pred.extend(torch.sigmoid(outputs[1]).detach().cpu().numpy())
                    pr_label.extend([label_binarize(label, classes=[v for v in self.label2id.values()]) 
                                     for label in label_ids])
                # backward pass
                loss = loss.mean()
                if f1_loss:
                    pred_tags, train_tags, _ = self.format_tags([list(p) for p in np.argmax(logits, axis=2)], label_ids, 'train')
                    f1 = f1_score(pred_tags, train_tags)
                    loss += 0.01*(1-f1)
                loss.backward()
                
                # train loss
                train_loss += loss.item()
                global_steps += 1
                self.tsboard['train'].add_scalar('loss/loss', loss.item(), global_steps)
                
                # avoid exploding gradients problem
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.max_grad_norm)
                
                # update parameters and learning rate
                optimizer.step()
                scheduler.step()
                self.tsboard['train'].add_scalar('loss/learning_rate', optimizer.param_groups[0]['lr'], global_steps)
                
            # train time
            train_time = time.time() - start
            # average train loss
            train_loss /= len(train_dataloader)
            
            # calculate metrics on each epoch
            pred_tags, train_tags, out = self.format_tags(predictions, true_labels, 'train') 
            train_metrics = self.metrics(pred_tags, train_tags, 'train', epoch)
            if self.plot_pr:
                self.pr_curve(pr_label, pr_pred, 'train', epoch)

            # --------------- Validation --------------- 
            model.eval()
            if ensemble: model_e.eval()
            valid_loss = 0
            predictions , true_labels, pr_pred, pr_label = [], [], [], []
            factor = len(train_dataloader)/len(valid_dataloader)
            start = time.time()
            for batch, batch_e in tqdm(zip(valid_dataloader, valid_dataloader_e), total=len(valid_dataloader), desc='valid'):
                # move batch to gpu
                b_input_ids, b_input_mask, b_labels = tuple(b.to(self.device) for b in batch)
                
                with torch.no_grad():
                    # Forward pass
                    outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                    loss = outputs[0].mean()
                    if ensemble:
                        e_input_ids, e_input_mask, e_labels = tuple(b.to(self.device) for b in batch_e)
                        outputs_e = model_e(e_input_ids, attention_mask=e_input_mask, labels=e_labels)
                        logits = (outputs[1] + outputs_e[1])/2
                        loss = self.criterion(logits, b_input_mask, b_labels)
                    else:
                        logits = outputs[1]
                loss = loss.mean()
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.detach().cpu().numpy()
                valid_loss += loss.item()
                if f1_loss:
                    pred_tags, train_tags, _ = self.format_tags([list(p) for p in np.argmax(logits, axis=2)], label_ids, 'valid')
                    f1 = f1_score(pred_tags, train_tags)
                    valid_loss += 0.01*(1-f1)
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.extend(label_ids)
                if self.plot_pr:
                    pr_pred.extend(torch.sigmoid(outputs[1]).detach().cpu().numpy())
                    pr_label.extend([label_binarize(label, classes=[v for v in self.label2id.values()]) 
                                     for label in label_ids])
                self.tsboard['valid'].add_scalar('loss/loss', outputs[0].mean().item(), valid_steps*factor)
                valid_steps += 1                
            valid_loss /= valid_steps
            valid_time = time.time() - start
            # format output and calculate metrics
            pred_tags, valid_tags, out = self.format_tags(predictions, true_labels, 'valid')
            valid_metrics = self.metrics(pred_tags, valid_tags, 'valid', epoch)
            if self.plot_pr:
                self.pr_curve(pr_label, pr_pred, 'valid', epoch)
            
            # save best result
            if valid_metrics['all']['f1'] > best_valid_f1:
                best_valid_epoch = epoch + 1
                best_valid_f1 = valid_metrics['all']['f1']
                # confusion matrix
                if self.plot_cm:
                    class_names = sorted([k for k in self.label2id.keys() if k!= self.pad_token])
                    cm = confusion_matrix(valid_tags, pred_tags, labels=class_names)
                    cm_fig = self.plot_confusion_matrix(cm, class_names)
                    self.tsboard['valid'].add_figure(f'Confusion_Matrix', cm_fig, epoch)
                # save best prediction output
                with open(os.path.join(self.output_dir, 'prediction_valid.txt'), 'w') as f:
                    f.write('\n'.join(out))
                # save best model
                model.module.save_pretrained(self.model_dir)
                self.tokenizer.save_pretrained(self.model_dir)
                # save best result
                with open(os.path.join(self.output_dir, 'result_valid.json'), 'w') as f:
                    valid_metrics['time'] = valid_time
                    valid_metrics['best_epoch'] = best_valid_epoch  
                    json.dump(valid_metrics, f)
                
            print('[Epoch %d] train_loss: %.4f, val_loss: %.4f' % (
                      epoch+1, train_loss, valid_loss))
            print('Train - time: %.2f, acc: %.2f%%, Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%%' % (
                      train_time, train_metrics['all']['accuracy'], train_metrics['all']['precision'],
                      train_metrics['all']['recall'], train_metrics['all']['f1']))            
            print('Valid - time: %.2f, acc: %.2f%%, Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%% (best epoch: %d)' % (
                      valid_time, valid_metrics['all']['accuracy'], valid_metrics['all']['precision'],
                      valid_metrics['all']['recall'], valid_metrics['all']['f1'], best_valid_epoch))

            # --------------- Test --------------- 
            model.eval()
            if ensemble: model_e.eval()
            test_loss = 0
            predictions , true_labels, pr_pred, pr_label = [], [], [], []
            start = time.time()
            for batch, batch_e in zip(test_dataloader, test_dataloader_e):
                # move batch to gpu
                b_input_ids, b_input_mask, b_labels = tuple(t.to(self.device) for t in batch)

                with torch.no_grad():
                    # Forward pass
                    outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                    loss = outputs[0].mean()
                    if ensemble:
                        e_input_ids, e_input_mask, e_labels = tuple(b.to(self.device) for b in batch_e)
                        outputs_e = model_e(e_input_ids, attention_mask=e_input_mask, labels=e_labels)
                        logits = (outputs[1] + outputs_e[1])/2
                        loss = self.criterion(logits, b_input_mask, b_labels)
                    else:
                        logits = outputs[1]
                loss = loss.mean()
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.detach().cpu().numpy()
                test_loss += loss.item()
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.extend(label_ids)
                if self.plot_pr:
                    pr_pred.extend(torch.sigmoid(outputs[1]).detach().cpu().numpy())
                    pr_label.extend([label_binarize(label, classes=[v for v in self.label2id.values()]) 
                                     for label in label_ids])
                test_steps += 1
            test_loss /= test_steps
            test_time = time.time() - start
            
            # format output and calculate metrics
            pred_tags, test_tags, out = self.format_tags(predictions, true_labels, 'test')
            test_metrics = self.metrics(pred_tags, test_tags, 'test', epoch)
            if self.plot_pr:
                self.pr_curve(pr_label, pr_pred, 'test', epoch)
   
            if test_metrics['all']['f1'] > best_test_f1:
                best_test_epoch = epoch + 1
                best_test_f1 = test_metrics['all']['f1']
                # save best prediction output
                with open(os.path.join(self.output_dir, 'prediction_test.txt'), 'w') as f:
                    f.write('\n'.join(out))
                # save best test result
                with open(os.path.join(self.output_dir, 'result_test.json'), 'w') as f:
                    test_metrics['time'] = test_time
                    test_metrics['best_epoch'] = best_test_epoch  
                    json.dump(test_metrics, f)
                    
            print('Test  - time: %.2f, acc: %.2f%%, Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%% (best epoch: %d)' % (
                      test_time, test_metrics['all']['accuracy'], test_metrics['all']['precision'], 
                      test_metrics['all']['recall'], test_metrics['all']['f1'], best_test_epoch))

            # Early stopping
            if valid_loss < best_val_loss:
                wait = 0
                best_val_loss = valid_loss
            else:
                wait += 1
                if wait >= self.early_stop or train_loss < valid_loss:
                    print('\nTerminated Training for Early Stopping at Epoch %d' % epoch)
                    break
                    
        for mode in ['train', 'valid', 'test']:
            self.tsboard[mode].close()
        
        self.model = model
        # return metrics for optimization experiments
        return {
            'f1': (valid_metrics['all']['f1'], 0.0), 
            'precision': (valid_metrics['all']['precision'], 0.0), 
            'recall': (valid_metrics['all']['recall'], 0.0),   
            'accuracy': (valid_metrics['all']['accuracy'], 0.0),
        }
    
    def criterion(self, logits, b_input_mask, b_labels):
        loss_fct = CrossEntropyLoss()
        # Only keep active parts of the loss
        attention_mask = b_input_mask
        labels = b_labels

        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return loss
    
    def metrics(self, pred_tags, gt_tags, mode, epoch):
        """
        calculate metrics and save to tensorboard
        """
        def calculate_metrics(pred_tags, gt_tags):
            f1 = f1_score(pred_tags, gt_tags)*100
            ppv = precision_score(pred_tags, gt_tags)*100
            sen = recall_score(pred_tags, gt_tags)*100
            acc = accuracy_score(pred_tags, gt_tags)*100
            return {'f1':f1, 'precision':ppv, 'recall':sen, 'accuracy':acc}
        
        # get metrics on all labels
        metric = {}
        metric['all'] = calculate_metrics(pred_tags, gt_tags)
        for m in ['accuracy', 'f1', 'precision', 'recall']:
            self.tsboard[mode].add_scalar('metrics/{}'.format(m), metric['all'][m], epoch)
            
        # get metrics on single label
        metric['individual'] = {}
        for tag in self.label2id.keys():
            if tag != 'O' and tag != self.pad_token and tag in gt_tags:
                pred = [p for p, g in zip(pred_tags, gt_tags) if p==tag or g==tag]
                gt = [g for p, g in zip(pred_tags, gt_tags) if p==tag or g==tag]
                metric['individual'][tag] = calculate_metrics(pred, gt)
            
        return metric
    
    def pr_curve(self, pr_label, pr_pred, mode, epoch):
        """
        Save PR curve of each labels on tensorbaord
        """
        for tag, idx in self.label2id.items():
            if tag != self.pad_token: 
                self.tsboard[mode].add_pr_curve(f'PR_Curve/{tag}', 
                                                   np.array(pr_label)[:,:,idx].flatten(), 
                                                   np.array(pr_pred)[:,:,idx].flatten(), epoch) 
                
    def plot_confusion_matrix(self, cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.
        Args:
          cm (array, shape = [n, n]): a confusion matrix of classes
          class_names (array, shape = [n]): String names of the classes
        """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix', fontsize=16)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
        plt.yticks(tick_marks, class_names, fontsize=12)
        # normalize the confusion matrix
        cm_norm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # use white text if squares are dark; otherwise black
        threshold = cm_norm.max() / 2.
        for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
            color = 'white' if cm_norm[i, j] > threshold else 'black'
            plt.text(j, i, cm[i, j], horizontalalignment='center', color=color, fontsize=14)

        plt.tight_layout()
        plt.ylabel('True label', fontsize=14)
        plt.xlabel('Predicted label', fontsize=14)

        return figure

if __name__ == "__main__":
    # if not running optimization experiments, get the parameters from arguments
    parameterization = {'lr': args.lr, 'decay': args.decay, 'warmups': args.warmups, 'eps': args.eps,
                        'batch': args.batch, 'n_epochs': args.n_epochs, 
                        'dset':args.dset, 'model':args.model_class}
    ner = TransformerNER()
    ner.trainer(parameterization)
    
