import os
import json
import numpy as np
import argparse
import datetime
from ax import ParameterType, RangeParameter, FixedParameter, SearchSpace, SimpleExperiment, modelbridge, models
from ax.plot.contour import interact_contour, plot_contour
from ax.plot.diagnostic import interact_cross_validation
from ax.plot.scatter import interact_fitted, plot_objective_vs_constraints
from ax.plot.slice import plot_slice
from ax.modelbridge.cross_validation import cross_validate
from ax.plot.trace import optimization_trace_single_method
from plotly.offline import init_notebook_mode, plot
from examples.NERCRF_conll_ax import graphie_evaluation_function

parser = argparse.ArgumentParser()
parser.add_argument('--dset', type=str, default='aaaaa', help='name of dataset')
parser.add_argument('--lr', type=float, nargs=2, default=[1e-4,1e-2], help='search space of learning rate')
parser.add_argument('--decay_rate', type=float, nargs=2, default=[0.01,0.2], help='search space of weight decay')
parser.add_argument('--lr_gcns', type=float, nargs=2, default=[1e-4,1e-2], help='search space of lr gcn')
parser.add_argument('--gcn_warmups', type=int, nargs=2, default=[500,2000], help='search space of gcn_warmups')
parser.add_argument('--pretrain_lstm', type=int, nargs=2, default=[3,7], help='search space of pretrain_lstm')
parser.add_argument('--init_trials', type=int, default=5, help='initialization trials')
parser.add_argument('--opt_trials', type=int, default=25, help='optimization trials')
parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs')

args = parser.parse_args()
dset = args.dset
lr = args.lr
decay_rate = args.decay_rate
lr_gcns = args.lr_gcns
gcn_warmups = args.gcn_warmups
pretrain_lstm = args.pretrain_lstm
init_trials = args.init_trials
opt_trials = args.opt_trials
n_epochs = args.n_epochs

# Search space
graphie_search_space = SearchSpace(parameters=[
    RangeParameter(
        name='lr', parameter_type=ParameterType.FLOAT, 
        lower=min(lr), upper=max(lr), log_scale=True),
    RangeParameter(
        name='decay_rate', parameter_type=ParameterType.FLOAT, 
        lower=min(decay_rate), upper=max(decay_rate), log_scale=False),
    RangeParameter(
        name='lr_gcns', parameter_type=ParameterType.FLOAT, 
        lower=min(lr_gcns), upper=max(lr_gcns), log_scale=False),
    RangeParameter(
        name='gcn_warmups', parameter_type=ParameterType.FLOAT, 
        lower=min(gcn_warmups), upper=max(gcn_warmups), log_scale=False), 
    RangeParameter(
        name='pretrain_lstm', parameter_type=ParameterType.FLOAT, 
        lower=min(pretrain_lstm), upper=max(pretrain_lstm), log_scale=False), 
    FixedParameter(name="dset", parameter_type=ParameterType.STRING, value=dset),
    FixedParameter(name="n_epochs", parameter_type=ParameterType.INT, value=n_epochs),
])

# Create Experiment
exp = SimpleExperiment(
    name = 'graphie',
    search_space = graphie_search_space,
    evaluation_function = graphie_evaluation_function,
    objective_name = 'f1',
)

# Run the optimization and fit a GP on all data
sobol = modelbridge.get_sobol(search_space=exp.search_space)
print(f"\nRunning Sobol initialization trials...\n{'='*40}\n")
for _ in range(init_trials):
    exp.new_trial(generator_run=sobol.gen(1))

for i in range(opt_trials):
    print(f"\nRunning GP+EI optimization trial {i+1}/{opt_trials}...\n{'='*40}\n")
    #print('exp status: ',exp.trials[0].status)
    gpei = modelbridge.get_GPEI(experiment=exp, data=exp.eval())
    exp.new_trial(generator_run=gpei.gen(1))
    
    if (i+1)%5==0:
        output_dir = os.path.join('data', 'Ax_output', 'new', datetime.datetime.now().strftime('%m%d-%H%M%S'))
        os.makedirs(output_dir)
        # Save all experiment parameters and best parameter
        df = exp.eval().df
        df.to_csv(os.path.join(output_dir, 'exp_eval.csv'), index=False)
        best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
        exp_arm = {k:v.parameters for k, v in exp.arms_by_name.items()}
        exp_arm['best'] = best_arm_name
        print('Best arm:\n', str(exp.arms_by_name[best_arm_name]))

        with open(os.path.join(output_dir, 'exp_arm.json'), 'w') as f: 
            json.dump(exp_arm, f)

        # Contour Plot
        os.makedirs(os.path.join(output_dir, 'contour_plot'))
        for metric in ['f1', 'precision', 'recall', 'accuracy']:
            contour_plot = interact_contour(model=gpei, metric_name=metric)
            plot(contour_plot.data, filename=os.path.join(output_dir, 'contour_plot', '{}.html'.format(metric)))

        # Tradeoff Plot
        tradeoff_plot = plot_objective_vs_constraints(gpei, 'f1', rel=False)
        plot(tradeoff_plot.data, filename=os.path.join(output_dir, 'tradeoff_plot.html'))


        # Slice Plot
        # show the metric outcome as a function of one parameter while fixing the others
        os.makedirs(os.path.join(output_dir, 'slice_plot'))
        for param in ["lr", "lr_gcns", "decay_rate", "gcn_warmups", "pretrain_lstm"]:
            slice_plot = plot_slice(gpei, param, "f1")
            plot(slice_plot.data, filename=os.path.join(output_dir, 'slice_plot', '{}.html'.format(param)))

        # Tile Plot
        # the effect of each arm
        tile_plot = interact_fitted(gpei, rel=False)
        plot(tile_plot.data, filename=os.path.join(output_dir, 'tile_plot.html'))

        # Cross Validation plot
        # splits the model's train data into train/test folds and makes out-of-sample predictions on the test folds.
        cv_results = cross_validate(gpei)
        cv_plot = interact_cross_validation(cv_results)
        plot(cv_plot.data, filename=os.path.join(output_dir, 'cv_plot.html'))
