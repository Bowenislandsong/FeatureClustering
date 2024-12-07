import unittest
import numpy as np
from sklearn.datasets import make_blobs
from optimal.autoclustering import OptimalClusters

class TestOptimalClusters(unittest.TestCase):
    def setUp(self):
        # Generate some sample data
        self.data = np.random.rand(100, 2)
        self.optimal_clusters = OptimalClusters(self.data)

    def test_get_optimal_clusters(self):
        self.optimal_clusters.elbow.run(max_clusters=5)
        optimal_clusters = self.optimal_clusters.elbow.get_optimal_clusters()
        self.assertTrue(isinstance(optimal_clusters, int))
        self.assertGreater(optimal_clusters, 0)

if __name__ == '__main__':
    unittest.main()
