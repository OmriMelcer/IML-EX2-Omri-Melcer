import numpy as np
import faiss

class KNNClassifier:
    def __init__(self, k, distance_metric='l2'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.Y_train = None
        self.index = None

    def fit(self, X_train, Y_train):
        """
        Update the kNN classifier with the provided training data.

        Parameters:
        - X_train (numpy array) of size (N, d): Training feature vectors.
        - Y_train (numpy array) of size (N,): Corresponding class labels.
        """
        self.X_train = X_train.astype(np.float32)
        self.Y_train = Y_train
        d = self.X_train.shape[1]
        if self.distance_metric == 'l2':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L2)
        elif self.distance_metric == 'l1':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L1)
        else:
            raise NotImplementedError
        pass
        self.index.add(self.X_train)

    def predict(self, X):
        """
        Predict the class labels for the given data using majority voting.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.

        Returns:
        - (numpy array) of size (M,): Predicted class labels.
        """
        distances, indices = self.knn_distance(X)
        M = X.shape[0]
        y_pred = np.zeros(M, dtype=self.Y_train.dtype)
        for i in range(M):
            neighbor_labels = self.Y_train[indices[i]]
            values, counts = np.unique(neighbor_labels, return_counts=True)
            y_pred[i] = values[np.argmax(counts)]

        return y_pred
    
    def predict_weighted(self, X, epsilon=1e-10):
        """
        Predict the class labels for the given data using inverse distance weighting.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.
        - epsilon (float): Small constant to prevent division by zero.

        Returns:
        - (numpy array) of size (M,): Predicted class labels.
        """
        distances, indices = self.knn_distance(X)
        M = X.shape[0]
        y_pred = np.zeros(M, dtype=self.Y_train.dtype)
        
        for i in range(M):
            neighbor_labels = self.Y_train[indices[i]]
            neighbor_distances = distances[i]
            
            # Calculate weights as inverse distance
            weights = 1.0 / (neighbor_distances + epsilon)
            
            # Get unique labels and sum weights for each
            unique_labels = np.unique(neighbor_labels)
            label_weights = np.zeros(len(unique_labels))
            
            for j, label in enumerate(unique_labels):
                # Sum weights for all neighbors with this label
                label_mask = (neighbor_labels == label)
                label_weights[j] = np.sum(weights[label_mask])
            
            # Predict label with highest weighted sum
            y_pred[i] = unique_labels[np.argmax(label_weights)]

        return y_pred

    def knn_distance(self, X):
        """
        Calculate kNN distances for the given data. You must use the faiss library to compute the distances.
        See lecture slides and https://github.com/facebookresearch/faiss/wiki/Getting-started#in-python-2 for more information.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.

        Returns:
        - (numpy array) of size (M, k): kNN distances.
        - (numpy array) of size (M, k): Indices of kNNs.
        """
        X = X.astype(np.float32)
        return self.index.search(X, self.k)
    
    
        
        
