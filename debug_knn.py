import pandas as pd
import numpy as np
import knn

def debug():
    # Load data
    print("Loading data...")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    X_train = train[["lat", "long"]].values
    Y_train = train["state"].values
    
    # Pick one test point
    idx = 0
    X_test_sample = test[["lat", "long"]].values[idx:idx+1] # Shape (1, 2)
    Y_test_sample = test["state"].values[idx]

    print(f"Test Point Coords: {X_test_sample[0]}")
    print(f"True Label: {Y_test_sample}")

    # Init KNN
    k = 5
    print(f"Training KNN with k={k}...")
    classifier = knn.KNNClassifier(k=k, distance_metric='l2')
    classifier.fit(X_train, Y_train)

    # Get neighbors manually
    print("Searching for neighbors...")
    distances, indices = classifier.knn_distance(X_test_sample)
    
    print("\n--- Neighbors Found ---")
    print(f"Indices: {indices[0]}")
    print(f"Distances: {distances[0]}")
    
    neighbor_indices = indices[0]
    neighbor_coords = X_train[neighbor_indices]
    neighbor_labels = Y_train[neighbor_indices]

    for i in range(k):
        print(f"Neighbor {i}: Coords={neighbor_coords[i]}, Label={neighbor_labels[i]}, Dist={distances[0][i]:.4f}")

    # Check prediction
    pred = classifier.predict(X_test_sample)
    print(f"\nPrediction: {pred[0]}")
    print(f"Correct? {pred[0] == Y_test_sample}")

if __name__ == "__main__":
    debug()
