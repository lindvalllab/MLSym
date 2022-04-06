"""
Copyright 2022, Dana-Farber Cancer Institute
License: GNU GPL 2.0
"""
# import relavant libraries
import os
import argparse
import json
from util import Processor

parser = argparse.ArgumentParser()
parser.add_argument('--model_output', type=str, help='path for model output file')
parser.add_argument('--label_output_dir', type=str)
parser.add_argument('--label_config', type=str)
parser.add_argument('--idx_map', type=str, help='path for list of idex maps (for GraphIE)')
parser.add_argument('--graphie', action='store_true', help='GraphIE model')
args = parser.parse_args()

model_output_path = args.model_output
idx_map_path = args.idx_map
label_config = args.label_config
label_output_dir = args.label_output_dir
graphie = args.graphie
        
if __name__== "__main__":
     
    process = Processor(label_config)
    process._init_model_output(model_output_path, idx_map_path)
    output_dir = os.path.join('/'.join(model_output_path.split('/')[:-1]), 'model_output')
    process.make_dir(output_dir)
    process.load_model_output()
    if graphie:
        completion = process.processing_model_output_graphie()
    else:
        completion = process.processing_model_output_transformer()
    
    # read the annotation output file, add model's prediction and save it
    for file, complete in completion.items():
        filename = os.path.join(label_output_dir, '{}.json'.format(file))
        with open(filename, 'r') as fin:
            data = json.load(fin)
            data['completions'].append(complete)
            with open(os.path.join(output_dir, '{}.json'.format(file)), 'w') as fout:
                json.dump(data, fout)

            
