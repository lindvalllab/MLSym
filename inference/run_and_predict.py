import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
from transformers import ElectraForTokenClassification, ElectraConfig, WEIGHTS_NAME, CONFIG_NAME
from transformers import ElectraTokenizer
import time
import pandas as pd
import spacy
import re
import _pickle as cPickle
import numpy as np
from extract_hpi import get_hpi_output
import argparse

#load pickle file which is a label to id mapping required for predictions
with open(r"./label2id.pkl", "rb") as input_file:
    label2id = cPickle.load(input_file)

_nlp = spacy.load('en_core_sci_lg', disable=['ner', 'parser', 'tagger', 'lemmatizer'])
_nlp.create_pipe('sentencizer')
_nlp.add_pipe('sentencizer')

tokenizer = ElectraTokenizer.from_pretrained('./electra/')
model = ElectraForTokenClassification.from_pretrained('./electra/')

def processing_hpi_output(data):
    """
    Clean the HPI section - text analysis

    :return: text
    :rtype: text

    """
    final = []
    for fileid, record in enumerate(data):

        text = re.sub('[\r\n]{2}', '. ', record['text'])
        text = re.sub('\r|\n|\W', ' ', text)
        text = re.sub('\\n', '', text)
        text = re.sub('\r|\n', ' ', text)
        esc = re.escape('.\\n\\n')
        text = re.sub(esc, '', text)
        esc2 = re.escape('\\n')
        text = re.sub(esc2, '', text)
        tokens = {tok.idx: tok.text for tok in _nlp.tokenizer(text)}

        out = ['-DOCSTART- -X- O O', '<s> _ _ O']
        sentence_sep = '</s> _ _ O'

        # get tokens and tags for documents
        for idx, txt in tokens.items():
            if re.search('\n+|\s+', txt):
                out.append('<blank> _ _ ')
            else:
                out.append(txt + ' _ _ ')

        # split documents into sentences and apply HPI sectioning if required
        text = ' '.join(o.split(' ')[0] for o in out)
        doc = _nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        start = 2
        for sent in sentences:
            start += len(sent.split())
            out.insert(start, sentence_sep)
            start += 1

        out.append(sentence_sep)
        final += out
    return final


def get_essentials(filepath):
    """
    Read train/valid/test data from input_dir and
    convert data to features (input_ids, label_ids, attention_masks)

    :return: label2id, num_labels, label2id[pad_token], id2label
    :rtype: dict, number, list, dict

    """

    pad_token = 0
    pad_token_label_id = -100
    with open(filepath, 'r') as f:
        lines = f.readlines()
        sentences, labels, idx, sent, lab, id = [], [], [], [], [], []
        tags = set()
        for line in lines:
            if '-DOCSTART-' in line or '</s>' in line or '<s>' in line or line.rstrip() == '':
                if sent and lab:
                    sentences.append(sent)
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
    label2id = {t: i for i, t in enumerate(list(tags))}
    num_labels = len(label2id)
    label2id[pad_token] = pad_token_label_id
    id2label = {v: k for k, v in label2id.items()}
    return label2id, num_labels, label2id[pad_token], id2label


def prepare_data(filepath):
    """
    Read train/valid/test data from input_dir and
    convert data to tokens

    :return: processed data, tokenized_token, tokenized_sentences
    :rtype: dataframe, list, list

    """

    pad_token = 0
    with open(filepath, 'r') as f:
        lines = f.readlines()
        sentences, sent = [], []
        for line in lines:
            if '-DOCSTART-' in line or '</s>' in line or '<s>' in line or line.rstrip() == '':
                if sent:
                    sentences.append(sent)
                if '-DOCSTART-' in line:
                    sent = ['D']
                else:
                    sent = []
            else:
                sent.append(line.split()[0])

    # tokenize the sentences and save the start offset of each subwords
    tokenized_sentences, tokenized_token = [], []
    for sent in sentences:
        tokenized_sent, tokenized_tok = [], []
        for word in sent:
            tokenized_word = tokenizer.tokenize(word)
            tokenized_sent.extend(tokenized_word)
            tokenized_tok.extend([word] * len(tokenized_word))
        # truncate the subword tokens longer than maxium sequence length
        if len(tokenized_sent) > 512:
            tokenized_sent = tokenized_sent[: 512]
            tokenized_tok = tokenized_tok[: 512]
        tokenized_sentences.append(tokenized_sent)
        tokenized_token.append(tokenized_tok)

    input_ids, attention_masks = [], []
    for sent in tokenized_sentences:
        # get token's id and label's id
        input_id = tokenizer.convert_tokens_to_ids(sent)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to
        input_mask = [1] * len(input_id)
        # Zero-pad up to the sequence length (pad on right)
        padding_length = 512 - len(input_id)
        input_id += [pad_token] * padding_length
        input_mask += [0] * padding_length
        input_ids.append(input_id)
        attention_masks.append(input_mask)

    data = TensorDataset(torch.tensor(input_ids), torch.tensor(attention_masks))
    return data, tokenized_token, tokenized_sentences


def format_tags(predictions,tokenized_token,id2label):
    """
    convert ids to original labels and create formatted output prediction

    :return: predicted tags, labels, indices
    :rtype: list, list, list

    """

    pred_tags, out = [], []
    docid =0
    start=0
    indices = []
    for prediction, token in zip(predictions,tokenized_token):
        i,j = False,False
        docid+=1
        if token == ['D']:
            i=True
        for index, (pred, tok) in enumerate(zip(prediction, token)):
            pred_tags.append(id2label[pred])
            text = '{} {}'.format(tok, id2label[pred])
            if text == 'D O':
                j=True
            out.append(text)
            if i==False and j==True:
                indices.append(start)
            j=False
            start+=1
        out.append('')
        start+=1
    return pred_tags, out,indices


def get_text_file(hpi_ip,col_name):
    """
    perform HPI processing

    """
    hpi_output = get_hpi_output(hpi_ip,col_name)
    processed_hpi_output = processing_hpi_output(hpi_output)
    with open('processed_file.txt', 'w') as f:
        f.writelines("%s\n" % o for o in processed_hpi_output)


def predict(file):
    """
    Load the saved model and perform prediction

    :return: labels dataframe
    :rtype: dataframe

    """
    hpi_ip = pd.read_csv(file)
    get_text_file(hpi_ip)
    te_data, tt, ts = prepare_data('./processed_file.txt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataloader = DataLoader(te_data, sampler=SequentialSampler(te_data), batch_size=32)
    model.to(device)
    model.eval()
    test_loss = 0
    test_steps = 0
    predictions, pr_label = [], []
    start = time.time()
    for batch in tqdm(test_dataloader, desc='Prediction'):
        # move batch to gpu
        b_input_ids, b_input_mask = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            # Forward pass
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

        test_steps += 1
    test_loss /= test_steps
    test_time = time.time() - start
    id2label = {v: k for k, v in label2id.items()}
    pred_tags, out, indices = format_tags(predictions, tt, id2label)
    index_pos_list = [i for i in range(len(out)) if out[i] == 'D O']
    for i in tqdm(indices, desc='indices'):
        index_pos_list.pop(index_pos_list.index(i))
    final = []
    for i in range(len(index_pos_list)):
        start = index_pos_list[i]
        if i < len(index_pos_list) - 1:
            end = index_pos_list[i + 1]
            final.append(out[start:end])
        else:
            final.append(out[start:])
    tags = []
    for i in tqdm(final):
        s = []
        for j in i:
            s.append(j.split(' ')[-1])
        tags.append(list(set(s)))
    labels = []
    for i in tqdm(tags, desc="Tags"):
        k = []
        for j in i:
            if j != '' or j != 'O':
                k.append(j.split('-')[-1])

        labels.append(k)
    hpi_ip['labels'] = pd.Series(labels)
    return hpi_ip


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict labels for clinical text')
    parser.add_argument('-ipf', '--input_file', dest='ipf', help='Path to input file', required=True)
    parser.add_argument('-cn', '--column_name', dest='cn', help='Name of the column containing the note', required=True)
    parser.add_argument('-opf', '--output_file', dest='opf', help='Path to output file', required=True)
    args = parser.parse_args()
    final_op = predict(args.ipf)
    final_op.to_csv(args.opf, index=False)
