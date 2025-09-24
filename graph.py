import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import pairwise_distances
from torch import Tensor
from scipy.linalg import inv
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
import time

class AdaptiveMutualKNN():
    def __init__(
        self,
        scale: str = 'StandardScaler',
        metric: str = 'euclidean',
    ) -> None:
        self.scaler = None
        if scale == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        elif scale == 'StandardScaler':
            self.scaler = StandardScaler()
        elif scale == 'RobustScaler':
            self.scaler = RobustScaler()
        
        self.metric = metric
        self.neighbor_counts = []
        self.k_values = []
        self.mutual_neighbor_counts = []
        self.noise_counter = 0
        self.edge_index = None
    
    def fit(
        self,
        X: Tensor,
        base_k: int = 3,
        percentile: float = 1.0,
    ) -> Tensor:
        X = X.numpy()
        num_samples, num_rows = X.shape
        print(f'# of samples: {num_samples}')
        print(f'# of features: {num_rows}')
        
        dist_matrix = None
        if self.metric == 'mahalanobis':
            VI = inv(np.cov(X.T))
            dist_matrix = pairwise_distances(X, metric=self.metric, VI=VI)
        else:
            if self.scaler != None:
                X = self.scaler.fit_transform(X)
            dist_matrix = pairwise_distances(X, metric=self.metric)
        
        np.fill_diagonal(dist_matrix, np.inf)
        
        adjacency_list = {i: set() for i in range(num_samples)}
        sample_rows = int(num_samples * 0.1)
        sample_indices = np.random.choice(num_samples, size=sample_rows, replace=False)
        threshold = np.percentile(dist_matrix[sample_indices], percentile).astype(np.float32)
        for i in range(num_samples):
            
            neighbors = np.where(dist_matrix[i] <= threshold)[0]
            self.neighbor_counts.append(len(neighbors))
            
            k = int((base_k * len(neighbors))**0.35)
            
            self.k_values.append(k)
            
            if k < len(neighbors):
                neighbors = neighbors[np.argpartition(dist_matrix[i][neighbors], k)[:k]]
            adjacency_list[i].update(neighbors)
        
        edge_index = []
        for i in range(num_samples):
            count = 0
            for j in adjacency_list[i]:
                if i in adjacency_list[j]:
                    count += 1
                    edge_index.append([i, j])
                    
            self.mutual_neighbor_counts.append(count)
            if count == 0:
                self.noise_counter += 1

        print(f'After MutualKNN, total # of noises: {self.noise_counter}.')
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        self.edge_index = edge_index
        return edge_index
    
    def show(
        self,
    ) -> None:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.hist(self.neighbor_counts, bins=50, alpha=0.9, color='y', edgecolor='black')
        plt.xlabel('Number of Potential Neighbors')
        plt.ylabel('Counting')
        plt.title('Result of Potential Neighbors')

        plt.subplot(1, 3, 2)
        plt.hist(self.k_values, bins=50, alpha=0.9, color='b', edgecolor='black')
        plt.xlabel('Adaptive k-values')
        plt.ylabel('Counting')
        plt.title('Result of Adaptive k-values')

        plt.subplot(1, 3, 3)
        plt.hist(self.mutual_neighbor_counts, bins=100, alpha=0.9, color='r', edgecolor='black')
        plt.xlabel('Number of Actual Neighbors')
        plt.ylabel('Counting')
        plt.title('Result of Actual Neighbors')

        plt.tight_layout()
        plt.show()

from mtam_knn import compute_mtam_knn

class MTAM_KNN:
    def __init__(
        self,
        scale: str = 'StandardScaler',
        metric: str = 'euclidean',
        num_threads: int = 1,
    ) -> None:
        if scale == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        elif scale == 'StandardScaler':
            self.scaler = StandardScaler()
        elif scale == 'RobustScaler':
            self.scaler = RobustScaler()
        else:
            self.scaler = None

        self.metric = metric
        self.edge_index = None
        self.num_threads = num_threads

    def fit(
        self,
        X: torch.Tensor,
        base_k: int = 3,
        sample_ratio: float = 0.1,
        percentile: float = 1.0,
        random_sample: bool = True,
    ) -> torch.Tensor:
        t_start = time.time()
        X_np = X.numpy()
        num_samples, num_features = X_np.shape
        print(f'# of samples: {num_samples}')
        print(f'# of features: {num_features}')

        if self.metric == 'mahalanobis':
            VI = inv(np.cov(X_np.T))
            dist_matrix = pairwise_distances(X_np, metric=self.metric, VI=VI)
        else:
            if self.scaler is not None:
                X_np = self.scaler.fit_transform(X_np)
            with threadpool_limits(limits=self.num_threads):
                dist_matrix = pairwise_distances(X, metric=self.metric)

        t1 = time.time()
        print(f"Distance time:   {t1 - t_start:.3f} s")
        np.fill_diagonal(dist_matrix, np.inf)
        sample_rows = int(num_samples * sample_ratio)
        sample_indices = np.random.choice(num_samples, size=sample_rows, replace=False)
        if random_sample:
            threshold = np.percentile(dist_matrix[sample_indices], percentile).astype(np.float32)
        else:
            threshold = np.percentile(dist_matrix, percentile).astype(np.float32)
        t2 = time.time()
        print(f'Percentile time: {t2 - t1:.3f} s')
        edge_list = compute_mtam_knn(dist_matrix, base_k, threshold, self.num_threads)
        t3 = time.time()
        print(f"Cython time:     {t3 - t2:.3f} s")

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        t_end = time.time()
        print(f"Tensor time:     {t_end - t3:.3f} s")
        print(f"Total:           {t_end - t_start:.3f} s")

        return edge_index
