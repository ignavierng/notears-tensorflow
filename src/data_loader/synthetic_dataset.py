import logging
import numpy as np
import networkx as nx


class SyntheticDataset(object):
    _logger = logging.getLogger(__name__)

    def __init__(self, n, d, graph_type, degree, sem_type, noise_scale=1.0, dataset_type='linear'):
        self.n = n
        self.d = d
        self.graph_type = graph_type
        self.degree = degree
        self.sem_type = sem_type
        self.noise_scale = noise_scale
        self.dataset_type = dataset_type
        self.w_range = (0.5, 2.0)

        self._setup()
        self._logger.debug('Finished setting up dataset class')

    def _setup(self):
        self.W = SyntheticDataset.simulate_random_dag(self.d, self.degree,
                                                      self.graph_type, self.w_range)

        self.X = SyntheticDataset.simulate_sem(self.W, self.n, self.sem_type, self.w_range,
                                               self.noise_scale, self.dataset_type)

    @staticmethod
    def simulate_random_dag(d, degree, graph_type, w_range):
        """Simulate random DAG with some expected degree.

        Args:
            d: number of nodes
            degree: expected node degree, in + out
            graph_type: {erdos-renyi, barabasi-albert, full}
            w_range: weight range +/- (low, high)

        Returns:
            W: weighted DAG
        """
        if graph_type == 'erdos-renyi':
            prob = float(degree) / (d - 1)
            B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
        elif graph_type == 'barabasi-albert':
            m = int(round(degree / 2))
            B = np.zeros([d, d])
            bag = [0]
            for ii in range(1, d):
                dest = np.random.choice(bag, size=m)
                for jj in dest:
                    B[ii, jj] = 1
                bag.append(ii)
                bag.extend(dest)
        elif graph_type == 'full':  # ignore degree, only for experimental use
            B = np.tril(np.ones([d, d]), k=-1)
        else:
            raise ValueError('Unknown graph type')
        # random permutation
        P = np.random.permutation(np.eye(d, d))  # permutes first axis only
        B_perm = P.T.dot(B).dot(P)
        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
        U[np.random.rand(d, d) < 0.5] *= -1
        W = (B_perm != 0).astype(float) * U

        return W

    @staticmethod
    def simulate_sem(W, n, sem_type, w_range, noise_scale=1.0, dataset_type='nonlinear_1'):
        """Simulate samples from SEM with specified type of noise.

        Args:
            W: weigthed DAG
            n: number of samples
            sem_type: {linear-gauss,linear-exp,linear-gumbel}
            noise_scale: scale parameter of noise distribution in linear SEM

        Returns:
            X: [n,d] sample matrix
        """
        G = nx.DiGraph(W)
        d = W.shape[0]
        X = np.zeros([n, d])
        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        for j in ordered_vertices:
            parents = list(G.predecessors(j))
            if dataset_type == 'linear':
                eta = X[:, parents].dot(W[parents, j])
            else:
                raise ValueError('Unknown dataset type')

            if sem_type == 'linear-gauss':
                X[:, j] = eta + np.random.normal(scale=noise_scale, size=n)
            elif sem_type == 'linear-exp':
                X[:, j] = eta + np.random.exponential(scale=noise_scale, size=n)
            elif sem_type == 'linear-gumbel':
                X[:, j] = eta + np.random.gumbel(scale=noise_scale, size=n)
            else:
                raise ValueError('Unknown sem type')    

        return X


if __name__ == '__main__':
    n, d = 3000, 20
    graph_type, degree, sem_type = 'erdos-renyi', 3, 'linear-gauss'
    noise_scale = 1.0

    dataset = SyntheticDataset(n, d, graph_type, degree, sem_type,
                               noise_scale, dataset_type='linear')
    print('dataset.X.shape: {}'.format(dataset.X.shape))
    print('dataset.W.shape: {}'.format(dataset.W.shape))
