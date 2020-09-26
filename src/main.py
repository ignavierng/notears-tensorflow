import os
import logging
from pytz import timezone
from datetime import datetime
import numpy as np

from data_loader import SyntheticDataset
from models import NoTears
from trainers import ALTrainer
from helpers.config_utils import save_yaml_config, get_args
from helpers.log_helper import LogHelper
from helpers.tf_utils import set_seed
from helpers.dir_utils import create_dir
from helpers.analyze_utils import count_accuracy, plot_estimated_graph


# For logging of tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    # Get arguments parsed
    args = get_args()

    # Setup for logging
    output_dir = 'output/{}'.format(datetime.now(timezone('Canada/Central')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
    create_dir(output_dir)
    LogHelper.setup(log_path='{}/training.log'.format(output_dir), level_str='INFO')
    _logger = logging.getLogger(__name__)

    # Save the configuration for logging purpose
    save_yaml_config(args, path='{}/config.yaml'.format(output_dir))

    # Reproducibility
    set_seed(args.seed)

    # Get dataset
    dataset = SyntheticDataset(args.n, args.d, args.graph_type, args.degree, args.sem_type,
                               args.noise_scale, args.dataset_type)
    _logger.info('Finished generating dataset')

    model = NoTears(args.n, args.d, args.seed, args.l1_lambda, args.use_float64)
    model.print_summary(print_func=model.logger.info)

    trainer = ALTrainer(args.init_rho, args.rho_max, args.h_factor, args.rho_multiply,
                        args.init_iter, args.learning_rate, args.h_tol)
    W_est = trainer.train(model, dataset.X, dataset.W, args.graph_thres,
                          args.max_iter, args.iter_step, output_dir)
    _logger.info('Finished training model')

    # Save raw estimated graph, ground truth and observational data after training
    np.save('{}/true_graph.npy'.format(output_dir), dataset.W)
    np.save('{}/X.npy'.format(output_dir), dataset.X)
    np.save('{}/final_raw_estimated_graph.npy'.format(output_dir), W_est)

    # Plot raw estimated graph
    plot_estimated_graph(W_est, dataset.W,
                         save_name='{}/raw_estimated_graph.png'.format(output_dir))

    _logger.info('Thresholding.')
    # Plot thresholded estimated graph
    W_est[np.abs(W_est) < args.graph_thres] = 0    # Thresholding
    plot_estimated_graph(W_est, dataset.W,
                         save_name='{}/thresholded_estimated_graph.png'.format(output_dir))
    results_thresholded = count_accuracy(dataset.W, W_est)
    _logger.info('Results after thresholding by {}: {}'.format(args.graph_thres, results_thresholded))


if __name__ == '__main__':
    main()
