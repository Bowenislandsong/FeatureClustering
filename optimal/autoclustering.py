from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

def assert_all_hasattr(obj, attrs):
    _hasAttr = [hasattr(obj, attr) for attr in attrs]
    assert all(_hasAttr), f"Object does not have attributes: {', '.join([attr for attr, hasAttr in zip(attrs, _hasAttr) if not hasAttr])}"

class OptimalClusters:
    def __init__(self, data):
        self.data = data
        self.elbow = self.Elbow(data)
        self.silhouette = self.Silhouette(data)

    class Elbow:
        def __init__(self, data):
            self.data = data
            self.wcss = []

        def run(self, clustering_method, max_clusters:int=10):
            self.wcss = []
            for i in range(1, max_clusters + 1):
                # assert_all_hasattr(clustering_method, ['n_clusters', 'random_state', 'inertia_'])
                method = clustering_method(n_clusters=i, random_state=42)
                method.fit(self.data)
                self.wcss.append(method.inertia_)

        def plot(self, save_path=None, ax=None):
            if not self.wcss:
                raise ValueError("WCSS values are empty. Run run method first.")
            if ax is None:
                fig, ax = plt.subplots()
            ax.plot(range(1, len(self.wcss) + 1), self.wcss)
            ax.set_title('Elbow Method')
            ax.set_xlabel('Number of clusters')
            ax.set_ylabel('WCSS')
            ax.grid(True)
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            
            # # Plot the first and second derivatives of WCSS
            # first_derivative = np.diff(self.wcss)
            # second_derivative = np.diff(first_derivative)

            # fig, ax = plt.subplots()
            # ax.plot(range(2, len(self.wcss) + 1), first_derivative, label='First Derivative')
            # ax.plot(range(3, len(self.wcss) + 1), second_derivative, label='Second Derivative')
            # ax.set_title('First and Second Derivatives of WCSS')
            # ax.set_xlabel('Number of clusters')
            # ax.set_ylabel('Derivative')
            # ax.legend()
            # if save_path:
            #     plt.savefig(save_path.replace('.png', '_derivatives.png'))
            # else:
            #     plt.show()

        def get_optimal_clusters(self):
            if not self.wcss:
                raise ValueError("WCSS values are empty. Run run method first.")
            if len(self.wcss) < 3:
                raise ValueError("Not enough data points to calculate the second derivative.")

            # Calculate the second derivative of WCSS
            second_derivative = np.diff(self.wcss, n=2)

            # Find the index of the maximum second derivative
            optimal_clusters = np.argmax(second_derivative) + 3
            return optimal_clusters

    class Silhouette:
        def __init__(self, data):
            self.data = data
            self.silhouette_scores = []

        def run(self, clustering_method, max_clusters=10):
            # assert_all_hasattr(clustering_method, ['n_clusters', 'random_state', 'labels_'])
            self.silhouette_scores = []
            for i in range(2, max_clusters + 1):
                method = clustering_method(n_clusters=i, random_state=42)
                method.fit(self.data)
                score = silhouette_score(self.data, method.labels_)
                self.silhouette_scores.append(score)

        def plot(self, save_path=None, ax=None):
            if not self.silhouette_scores:
                raise ValueError("Silhouette scores are empty. Run run method first.")
            if ax is None:
                fig, ax = plt.subplots()
            ax.plot(range(2, len(self.silhouette_scores) + 2), self.silhouette_scores)
            ax.set_title('Silhouette Method')
            ax.set_xlabel('Number of clusters')
            ax.set_ylabel('Silhouette Score')
            ax.grid(True)
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()

        def get_optimal_clusters(self):
            if not self.silhouette_scores:
                raise ValueError("Silhouette scores are empty. Run run method first.")
            optimal_clusters = self.silhouette_scores.index(max(self.silhouette_scores)) + 2
            return optimal_clusters
        
if __name__ == '__main__':
    # Generate random data with 5 clusters
    data, _ = make_blobs(n_samples=500, centers=5, cluster_std=0.60, random_state=42)

    # Create an instance of OptimalClusters
    optimal_clusters = OptimalClusters(data)

    # Run Elbow method
    optimal_clusters.elbow.run(KMeans)
    optimal_clusters.elbow.plot(save_path='elbow.png')
    print("Optimal clusters (Elbow method):", optimal_clusters.elbow.get_optimal_clusters())

    # Run Silhouette method
    optimal_clusters.silhouette.run(KMeans)
    optimal_clusters.silhouette.plot(save_path='silhouette.png')
    print("Optimal clusters (Silhouette method):", optimal_clusters.silhouette.get_optimal_clusters())