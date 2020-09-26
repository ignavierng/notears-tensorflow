import logging
import numpy as np
import tensorflow as tf

from helpers.dir_utils import create_dir
from helpers.analyze_utils import count_accuracy


class ALTrainer(object):
    """
    Augmented Lagrangian method with gradient-based optimization
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, init_rho, rho_max, h_factor, rho_multiply, init_iter, learning_rate, h_tol):
        self.init_rho = init_rho
        self.rho_max = rho_max
        self.h_factor = h_factor
        self.rho_multiply = rho_multiply
        self.init_iter = init_iter
        self.learning_rate = learning_rate
        self.h_tol = h_tol

    def train(self, model, X, W_true, graph_thres, max_iter, iter_step, output_dir):
        """
        model object should contain the several class member:
        - sess
        - train_op
        - loss
        - mse_loss
        - h
        - W_prime
        - X
        - rho
        - alpha
        - lr
        """
        model.sess.run(tf.compat.v1.global_variables_initializer())
        rho, alpha, h, h_new = self.init_rho, 0.0, np.inf, np.inf

        self._logger.info('Started training for {} iterations'.format(max_iter))
        for epoch in range(1, max_iter + 1):
            while rho < self.rho_max:
                self._logger.info('rho {:.3E}, alpha {:.3E}'.format(rho, alpha))
                loss_new, mse_new, h_new, W_new = self.train_step(model, iter_step, X, rho, alpha)
                if h_new > self.h_factor * h:
                    rho *= self.rho_multiply
                else:
                    break

            self.train_callback(epoch, loss_new, mse_new, h_new, W_true, W_new, graph_thres, output_dir)
            W_est, h = W_new, h_new
            alpha += rho * h

            if h <= self.h_tol and epoch > self.init_iter:
                self._logger.info('Early stopping at {}-th iteration'.format(epoch))
                break

        # Save model
        model_dir = '{}/model/'.format(output_dir)
        model.save(model_dir)
        self._logger.info('Model saved to {}'.format(model_dir))

        return W_est

    def train_step(self, model, iter_step, X, rho, alpha):
        for _ in range(iter_step):
            _, curr_loss, curr_mse, curr_h, curr_W \
                = model.sess.run([model.train_op, model.loss, model.mse_loss, model.h, model.W_prime],
                                 feed_dict={model.X: X,
                                            model.rho: rho,
                                            model.alpha: alpha,
                                            model.lr: self.learning_rate})

        return curr_loss, curr_mse, curr_h, curr_W

    def train_callback(self, epoch, loss, mse, h, W_true, W_est, graph_thres, output_dir):
        # Evaluate the learned W in each iteration after thresholding
        W_thresholded = np.copy(W_est)
        W_thresholded[np.abs(W_thresholded) < graph_thres] = 0
        results_thresholded = count_accuracy(W_true, W_thresholded)

        self._logger.info(
            '[Iter {}] loss {:.3E}, mse {:.3E}, acyclic {:.3E}, shd {}, tpr {:.3f}, fdr {:.3f}, pred_size {}'.format(
                epoch, loss, mse, h, results_thresholded['shd'], results_thresholded['tpr'],
                results_thresholded['fdr'], results_thresholded['pred_size']
            )
        )

        # Save the raw estimated graph in each iteration
        create_dir('{}/raw_estimated_graph'.format(output_dir))
        np.save('{}/raw_estimated_graph/graph_iteration_{}.npy'.format(output_dir, epoch), W_est)