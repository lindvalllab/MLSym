"""
Copyright 2022, Dana-Farber Cancer Institute
License: GNU GPL 2.0
"""
# import relavant libraries
import os
import glob
import spacy
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
import json
import re

warnings.filterwarnings('ignore')
start_interval = ['[ ?y.]{0,6}(hpi\W{0,3}|)interval (history|hx)',
                  '[ ?y.]{0,6}interim history',
                  '[ ?y.]{0,6}change(s|) since last visit',
                  '[ ?y.]{0,6}at today\'s visit', 
                  '^[ ?y.]{0,6}current status']
start_hpi = ['hpi', 'history of present illness']
start_cc = ['^[ ?y.]{0,2}diagnosis/identification',
            'identif(ication|ier|ying information)',
            '(cc/|patient |)id(/cc| and cc):',
            'reason(s|) for (the |)(visit|consult(ation|))',
            'chief complaint','c(/|)c:',
            'today'] 
stop_sec = [#'^[ ?y.]{0,6}(history of present illness|hpi)',
            '^[ ?y.]{0,6}oncolog(y|ic(al|)) (and other medical |)history', 'oncolog(y|ic(al|)) (and other medical |)history:', 
            '^[ ?y.]{0,6}\(copied from .{1,15} for clinical reference', '^[ ?y.]{0,6}medical oncology visit',
            '^[ ?y.]{0,6}(onc|gyn(ecologic|)|bone health fall|menstrual( and obstetric|)|ob(stetric|)|psychiatric) history',
            '^[ ?y.]{0,6}(past|prior)( medical|) .{0,20}history', '^[ ?y.]{0,6}pm(d|h)', 'pmh(/psh|):', 
            '^[ ?y.]{0,6}(personal and |)(social|family|desensitization) (history|hx:)',  
            '^[ ?y.]{0,6}problem(s|)( list|).{1,3}',
            '^[ ?y.]{0,6}(patient|) active problem list', 
            '^[ ?y.]{0,6}relevant .{0,15}history', '^[ ?y.]{0,6}relevant past .{0,20}history',
            #'^[ ?y.]{0,6}review of (systems|symptoms)', '^[ ?y.]{0,6}ros was negative for', '[ ?y.]{0,6}(ros|constitutional):',
            '^[ ?y.]{0,6}physical exam(ination|)', 
            '^[ ?y.]{0,6}lab(s:| result)',  '^[ ?y.]{0,6}latest known visit with result', '^[ ?y.]{0,6}blood Draw', 
            'component\s+date\s+value\s+ref', 'component\s+value\s+date', 'result\s+value\s+ref\s+range',
            '^[ ?y.]{0,6}imaging( studies| result(s|)|):', '^[ ?y.]{0,6}.{0,15}\sstudies:', '^[ ?y.]{0,6}pathology(/biopsy|):',
            '^[ ?y.]{0,6}(pathologic |)diagnosis', '^[ ?y.]{0,6}diagnosis:',
            '^[ ?y.]{0,6}allerg(ies:|en reactions)', '^[ ?y.]{0,6}medication((s|):| sig)', '^[ ?y.]{0,6}medications reviewed', '^[ ?y.]{0,6}drug/med allergies',
            '^[ ?y.]{0,6}current (med(ication|)s:|outpatient (prescription|medication))', '^[ ?y.]{0,6}current .{0,10}(treatments/ |)medications',
            '^[ ?y.]{0,6}controlled substance information', '^[ ?y.]{0,6}immunizations:',
            '^[ ?y.]{0,6}interventions(:| tried for .{1,15}:)',
            '^[ ?y.]{0,6}problems, past medical history',
            '^[ ?y.]{0,6}information obtained from\s','^[ ?y.]{0,6}information from collateral',
            '^[ ?y.]{0,6}stage [1-4]$', '[ ?y.]{0,6}active therapy at visit .{0,15}',
            '^[ ?y.]{0,6}symptom assessment:', 'assessment (&|and) plan', 
            '^[ ?y.]{0,6}review of patient\'s current problem list',
            '^[ ?y.]{0,6}oncopanel', '[ ?y.]{0,6}treatment (summary|plan):','[ ?y.]{0,6}active treatment (& therapy |)days', 
            '^[ ?y.]{0,6}requesting physician', 'staging form:', '--hospital course--', '\W.{1,15} symptom scale:', 
            '^[ ?y.]{0,6}answers for hpi/ros submitted by the patient',
            '^[ ?y.]{0,6}i have reviewed and confirmed', '24 hour event'] 


class Processor():
    def __init__(self, label_config):
        # check file exist and get label groups and names
        self.label_config = label_config
        self.check_dir(self.label_config) 
        self.parse_label()
        
    def _init_label_output(self, input, label, keep, tag, hpi, data_split, stratified_split, test, model):
        self.input = input
        self.label = label
        self.keep = keep
        self.tag = tag
        self.hpi = hpi
        self.data_split = data_split  
        self.stratified_split = stratified_split
        self.test = test
        self.model = model
    
        self.check_dir(self.input)
        print('loading en_core_sci_lg....')
        self._nlp = spacy.load('en_core_sci_lg', disable=['ner', 'parser', 'tagger'])
        self._nlp.add_pipe(self._nlp.create_pipe('sentencizer'))
        
    def _init_model_output(self, model_output_path, idx_map_path):
        self.model_output_path = model_output_path
        self.idx_map_path = idx_map_path
        
        self.check_dir(self.model_output_path)
        self.check_dir(self.idx_map_path)

    def check_dir(self, dir_path):
        """
        check if the files exist in the directory path
        """
        if not os.path.exists(dir_path):
            raise FileNotFoundError('{} doesn\'t exist'.format(dir_path))

    def make_dir(self, dir_path):
        """
        create a directory if directory path doesn't exist
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def parse_label(self):
        """
        get labels group names and labels within the group
        """
        self.label_map = defaultdict(list)
        self.label_group = {}
        tree = ET.parse(self.label_config)
        root = tree.getroot()
        for labels in root.findall('.//Labels'):
            name = labels.get('name')
            for lab in labels.getiterator():
                if lab.tag=='Label': 
                    self.label_map[name].append(lab.get('value'))
                    self.label_map['all'].append(lab.get('value'))
                    self.label_group[lab.get('alias')] = name
        for text in root.findall('.//Text'):
            if text.get('name') == 'text': 
                self.label_map['input'] = text.get('value')[1:]

    def processing_dir(self):
        """
        processing the all the files in dir
        :return - output [list]: model's input data for all the files in dir
                - file   [list]: lists of filenames in the dir with same order as output
        """        
        # split the data into train, valid, test set (70/15/15) if required
        output, idx_maps, file = defaultdict(list), defaultdict(list), {}
        if self.data_split:
            file_list = glob.glob(os.path.join(self.input, '*.json'))
            file['train'], file['valid'] = train_test_split(file_list, test_size=self.data_split, random_state=123)
            if self.test:
                file['valid'], file['test'] = train_test_split(file['valid'], test_size=0.5, random_state=123)
        elif self.stratified_split:
            labels, Y, X = [], [], glob.glob(os.path.join(self.input, '*.json'))
            # get label count on document level
            for filename in glob.glob(os.path.join(self.input, '*.json')):
                with open(filename, 'r') as fp:
                    data = json.load(fp)   
                    # B-tag
                    labels.extend(list(set(d['value']['labels'][0] for d in data['completions'][0]['result'] if 'value' in d and d['value']['end']-d['value']['start']>=2)))
                    # I-tag
                    labels.extend(list(set('|'+d['value']['labels'][0] for d in data['completions'][0]['result'] 
                                           if 'value' in d and len(data['data'][self.label_map['input']][d['value']['start']:d['value']['end']].split())>1)))
            label_count, label_ids = Counter(labels), {}
            
            # convert label to ids and set less common labels (count <3) to ids 0
            for idx, lab in enumerate(sorted(label_count, key=label_count.get)):
                label_ids[lab.split('|')[-1]] = 0 if label_count[lab]<3 else idx+1 

            # get each document least common label ids
#             pdb.set_trace()
            for filename in glob.glob(os.path.join(self.input, '*.json')):
                with open(filename, 'r') as fp:   
                    data = json.load(fp)  
                    if not data['completions'][0]['result']:
                        Y.append(0)
                    else:
                        try:
                            Y.append(min([label_ids[d['value']['labels'][0]] \
                                      for d in data['completions'][0]['result'] if 'value' in d and d['value']['end']-d['value']['start']>=2]))
                        except:
                            print(data['completions'][0]['result'])
                        
            # remove documents with less common lebals (count <3) from stratified split
            x = [i for i, j in zip(X, Y) if j > 0 and Counter(Y)[j] >1]
            y = [j for i, j in zip(X, Y) if j > 0 and Counter(Y)[j] >1]
            
             # stratified split according to least common label
            file['train'], file['valid'] = train_test_split(x, test_size=self.stratified_split, 
                                                            stratify=y, random_state=123)
            
            # add documents with less common lebals (count <3) to train dataset
            file['train'].extend([i for i, j in zip(X, Y) if j == 0 or Counter(Y)[j] == 1])

            # valid and test split
            if self.test:
                file['valid'], file['test'] = train_test_split(file['valid'], test_size=0.5, random_state=123)
        else:
            file['train'] = glob.glob(os.path.join(self.input, '*.json'))
            
        # dictonary of filename belongs to which dataset (train/valid/test)
        modes = {f: mode for mode, files in file.items() for f in files}
        for mode in file.keys():
            print('{}: {}'.format(mode, len(file[mode])))
        # call processing_json function to process all the files in dir
        for filename in tqdm(glob.glob(os.path.join(self.input, '*.json'))):
            out, idx_map  =  self.processing_json(filename)
            output[modes[filename]].extend(out)
            idx_maps[modes[filename]].extend(idx_map)

        return output, idx_maps

        
    def processing_json(self, filename):
        """
        processing the annotation json files to model's input format
        :return - out      [list]: model's input data for single json file
                - idx_map  [list]: filename of the json file and start offset of each token
        """
        
        keep = self.keep if self.keep else list(set(self.label_map[self.label]))
       
        with open(filename, 'r') as fp:
            # load annotation output
            data = json.load(fp)
            text = re.sub('[\r\n]{2}', '. ', data['data'][self.label_map['input']])
            text = re.sub('\r|\n', ' ', text)
            tokens = {tok.idx:tok.text for tok in self._nlp.tokenizer(text)}
            labels = [d['value'] for d in data['completions'][0]['result'] if 'value' in d and d['value']['end']-d['value']['start']>=2]
            labels = sorted(labels, key=lambda x:x['start'])
            l, flag = 0, False
            out = ['-DOCSTART- -X- O O','<s> _ _ O'] if self.model == 'graphie' else []
            idx_map = [int(filename.split('/')[-1].split('.')[0]), filename] if self.model == 'graphie' else []
            sentence_sep = '</s> _ _ O' if self.model == 'graphie' else '\n'
            
            # get tokens and tags for documents
            for idx, txt in tokens.items():
                ner_tag = 'O'
                if len(labels) > 0: 
                    while l < len(labels)-1 and labels[l]['end'] <= idx:
                        l += 1; flag = False
                    if labels[l]['labels'][0] in keep:
                        label = self.tag if self.tag else labels[l]['labels'][0].replace('-', '_')
                    else:
                        label = ''
                    check, adjust = labels[l]['start'], 0
                    while text[check:check+1] in [' ', '\s']:
                        check += 1; adjust += 1
                    if idx == labels[l]['start']+adjust: 
                        ner_tag = 'B-' + label if label else 'O'
                        flag = True
                    elif flag and idx < labels[l]['end']: 
                        ner_tag = 'I-' + label if label else 'O'
                    elif flag:
                        flag = False
                if re.search('\n+|\s+', txt): 
                    out.append('<blank> _ _ ' + ner_tag)
                else:
                    out.append(txt + ' _ _ ' + ner_tag) 
                idx_map.append(idx)
                        
            # split documents into sentences and apply HPI sectioning if required
            text = ' '.join(o.split(' ')[0] for o in out)
            doc = self._nlp(text)
            sentences = [sent.string.strip() for sent in doc.sents]
            start = 2 if self.model == 'graphie' and self.hpi else 0
            if self.hpi == True:
                hpi, cc = False, False
                inter = any(re.search(s.replace('[','[\n\r'), doc.text.lower()) for s in start_interval)
                for idx, sent in enumerate(sentences): 
                    include = False
                    if hpi != 'done':
                        if cc and any(re.search(s.replace('^','^(<blank>|)'), sent.lower()) for s in stop_sec + start_interval + start_hpi): cc = False
                        if hpi or cc:
                            if any(re.search(s.replace('^','^(<blank>|)'), sent.lower()) for s in stop_sec): hpi = 'done'
                            include = True
                        else:
                            if any(re.search(s.replace('^','^(<blank>|)'), sent.lower()) for s in start_cc): include = True; cc = True
                            if (inter and any(re.search(s.replace('^','^(<blank>|)'), sent.lower()) for s in start_interval)) or \
                                (not inter and any(re.search(s.replace('^','^(<blank>|)'), sent.lower()) for s in start_hpi)): 
                                include = True; hpi = True
                    else: 
                        del out[start:]; del idx_map[start:]; break

                    if idx == 0 and self.model == 'graphie': length = len(sent.split()) - 2
                    else: length = len(sent.split()) 
                    if include:
                        start += length 
                        out.insert(start, sentence_sep)
                        idx_map.insert(start, -1)
                        start += 1
                    else:
                        del out[start:start+length]
                        del idx_map[start:start+length]
            else:       
                for sent in sentences: 
                    start += len(sent.split())
                    out.insert(start, sentence_sep)
                    idx_map.insert(start, -1)
                    start += 1  

            out.append(sentence_sep)
            idx_map.append(-1)

        return out, idx_map
    
    def load_model_output(self):
         # load the model output file
        with open(self.model_output_path, 'r') as file:
            self.model_output = [line.strip() for line in file.readlines()]
        with open(self.idx_map_path, 'r') as file:
            self.idx_map = [line.strip() for line in file.readlines()]
        
    def processing_model_output_graphie(self):
        """
        add model's prediction to each annotation file
        :return - completion [dict]: key: filename; value: ground truth annotation and model's prediction
        """
        
        completion, result = {}, []
        for idx, labels in zip(self.idx_map, self.model_output):
            if not result and labels.split()[1] == '-DOCSTART-': 
                note_id = int(idx)
            elif not result and labels.split()[1] == '<s>':
                filename = idx
            elif result and labels.split()[1] == '-DOCSTART-': 
                completion[filename] = {'id': note_id*1000 + 2, 'lead_time': 100.001, 'result': result}
                result = []
            elif labels.split()[1] != '<s>' and labels.split()[-1] != 'O':
                prefix = labels.split()[-1].split('-')[0]
                label = labels.split()[-1].split('-')[-1]
                token = labels.split()[1]
                # if all labels are replaced by a single tag during the label_output.py processing
                # assign an existing label in order to open the files in annotation server
                if label not in self.label_map.keys():
                    from_name = [k for k in self.label_map][0]
                    label = self.label_map[from_name][0]
                else:
                    from_name = self.label_map[label]
                if prefix == 'B':
                    value = {'end': int(idx)+len(token), 'labels': [label], 'start': int(idx), 'text': ''}
                    result.append({'from_name': from_name, 'id':'graphie', 'source':'${}'.format(self.label_map['input']),
                                    'to_name':'text', 'type':'labels', 'value':value})
                elif prefix == 'I':
                    result[-1]['value']['end'] = int(idx)+len(token)
                    
        return completion
    
    def processing_model_output_transformer(self):
        """
        add model's prediction to each annotation file
        :return - completion [dict]: key: filename; value: ground truth annotation and model's prediction
        """

        completion, result = {}, [] 
        prev_start = -1
        for labels in self.model_output:
            if labels.split() and 'D' in labels.split()[0]:
                if result: 
                    completion[note_id] = {'id': note_id*1000 + 2, 'lead_time': 100.001, 'result': result}
                    result = []
                note_id = int(labels.split()[0][1:])
            elif labels.split() and labels.split()[-1] != 'O':
                prefix = labels.split()[-1].split('-')[0]
                label = labels.split()[-1].split('-')[-1]
                token = labels.split()[1]
                idx = int(labels.split()[0])
                if idx != prev_start:
                    prev_start = idx
                    # if all labels are replaced by a single tag during the label_output.py processing
                    # assign an existing label in order to open the files in label-stuio annotation server
                    if label not in self.label_group.keys():
                        from_name = [k for k in self.label_map][0]
                        label = self.label_map[from_name][0]
                    else:
                        from_name = self.label_group[label]
                    if prefix == 'B':
                        value = {'end': idx+len(token), 'labels': [label], 'start': idx, 'text': ''}
                        result.append({'from_name': from_name, 'id':'model', 'source':'${}'.format(self.label_map['input']),
                                        'to_name':'text', 'type':'labels', 'value':value})
                    elif prefix == 'I' and result:
                        result[-1]['value']['end'] = idx+len(token)
        if result: 
            completion[note_id] = {'id': note_id*1000 + 2, 'lead_time': 100.001, 'result': result}
        
        print('Total output files: ', len(completion))
                    
        return completion
