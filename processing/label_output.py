"""
Copyright 2022, Dana-Farber Cancer Institute
License: GNU GPL 2.0
"""
# import relavant libraries
import os
import argparse
from util import Processor

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='input dir or input json file')
parser.add_argument('--label_config', type=str)
parser.add_argument('--label', type=str, default='all', help='include which group of labels or include all labels')
parser.add_argument('--keep', type=str, nargs='+', help='list of labels to keep')
parser.add_argument('--tag', type=str, default=None, help='replace all labels to single tag') 
parser.add_argument('--hpi', action='store_true', help='hpi sectioning')
parser.add_argument('--data_split', type=float, help='train, valid set split (test proportion)')
parser.add_argument('--stratified_split', type=float, help='train, valid set stratified split (test proportion)') 
parser.add_argument('--test', action='store_true', help='test set split')
parser.add_argument('--model', type=str, default='graphie')
args = parser.parse_args()

input = args.input
label_config = args.label_config
label = args.label
keep = args.keep
tag = args.tag
hpi = args.hpi
data_split = args.data_split
stratified_split = args.stratified_split
test = args.test
model = args.model

if __name__== "__main__":
    # check dir/file exist and get label config
    process = Processor(label_config)
    process._init_label_output(input, label, keep, tag, hpi, data_split, stratified_split, test, model)
    
    # make output dir
    output_dir = 'processing/outputs/v1_{}_{}_{}'.format(input.split('/')[0], 'hpi' if hpi else 'entire', label)
    process.make_dir(output_dir)
    # process annotation output by dir/file
    output, idx_maps = process.processing_dir() if os.path.isdir(input) else process.processing_json(input)
    
    for dset in output.keys():
        # save output files for train, valid and test dataset
        with open('{}/{}.txt'.format(output_dir, dset), 'w') as file:
            file.writelines("%s\n" % o for o in output[dset])
        file.close()
        
        # save idx map and filenames for furture use (process model output)
        with open('{}/idx_map_{}'.format(output_dir, dset), 'w') as file:
            file.writelines("%s\n" % f for f in idx_maps[dset])
