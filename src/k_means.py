import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

style.use('ggplot')


class KMeans:
    def __init__(self, k=2, iterations=200, tolerance=0.001):
        self.k = k
        self.iterations = iterations
        self.classifications = {}
        self.centroids = {}
        self.tolerance = tolerance

    def fit(self, data_frame):
        # Optimization: Initializing centroid based on data distribution
        self.centroids = self.initialize_centroids(data_frame)

        data = data_frame.to_numpy()
        centroid = 0
        while centroid < self.k:
            # to assign random numbers. first shuffle and pick the first 2
            self.centroids[centroid] = data[centroid]
            centroid += 1

        for iteration in range(1, self.iterations):
            print(iteration)

            # Initialize clusters to empty lists
            self.initialize_clusters()

            for sample in data:
                distance_to_clusters = [np.linalg.norm(sample - self.centroids[centroid]) for centroid in
                                        self.centroids]
                classification = distance_to_clusters.index(min(distance_to_clusters))
                self.classifications[classification].append(sample)
            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            # Stopping Criteria
            optimized = True
            for index in self.centroids:
                previous_centroid = prev_centroids[index]
                current_centroid = self.centroids[index]
                if np.sum((current_centroid - previous_centroid) / previous_centroid * 100.0) > self.tolerance:
                    print(np.sum((current_centroid - previous_centroid) / previous_centroid * 100.0))
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distance_to_clusters = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        cluster_group = distance_to_clusters.index(min(distance_to_clusters))
        return cluster_group

    def initialize_centroids(self, data_frame):
        max_column = (data_frame.max() - data_frame.min()).idxmax()
        df_sorted = data_frame.sort_values(by=[max_column])
        df_partition = np.array_split(df_sorted, self.k)
        df_mean = [np.mean(arr) for arr in df_partition]
        centroid_dict = {}
        for index in range(self.k):
            centroid_dict[index] = df_mean[index]
        return centroid_dict

    def initialize_clusters(self):
        for cluster in range(self.k):
            self.classifications[cluster] = []

    def display_result(self):
        colors = 10 * ["g", "r", "c", "b", "k"]
        for centroid in self.centroids:
            plt.scatter(self.centroids[centroid][0], self.centroids[centroid][1],
                        marker="o", color="k", s=150, linewidths=5)

        for classification in self.classifications:
            color = colors[classification]
            for featureset in self.classifications[classification]:
                plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

        plt.show()


if __name__ == '__main__':
    clf = KMeans(2, 10)
    df = pd.read_csv('data/test.csv')
    clf.fit(df)
    clf.display_result()
