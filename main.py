import pandas as pd
import knn
import numpy as np
import helpers as helper
import matplotlib.pyplot as plt


def load_data():
    """
    Load train and test data from CSV files.
    Returns:
        X_train, Y_train, X_test, Y_test (as numpy arrays)
    """
    train = pd.read_csv("train.csv")
    X_train = train[["long", "lat"]].values
    Y_train = train["state"].values
    test = pd.read_csv("test.csv")
    X_test = test[["long", "lat"]].values
    Y_test = test["state"].values
    return X_train, Y_train, X_test, Y_test

def run_knn_experiment(X_train, Y_train, X_test, Y_test, k, metric):
    """
    Train and evaluate a single KNN model.
    Returns:
        y_pred, accuracy
    """
    knn_classifier = knn.KNNClassifier(k=k, distance_metric=metric)
    knn_classifier.fit(X_train, Y_train)
    y_pred = knn_classifier.predict(X_test)
    accuracy = np.mean(y_pred == Y_test)
    return y_pred, accuracy

def print_results(k, metric, accuracy):
    """
    Print the results in a readable format.
    """
    print(f"K={k:<5} | Metric={metric:<3} | Accuracy={accuracy:.4f}")

def run_all_experiments(X_train, Y_train, X_test, Y_test, K_values, metrics):
    """
    Run KNN experiments for all combinations of K and metrics.
    Returns:
        predictions_accuracies: List of lists containing (y_pred, accuracy) tuples.
    """
    predictions_accuracies = []
    
    print(f"{'='*40}")
    print(f"{'KNN Experiment Results':^40}")
    print(f"{'='*40}")

    for k in K_values:
        for_each_l = []
        for l in metrics:
            # 2. Run Experiment
            y_pred, accuracy = run_knn_experiment(X_train, Y_train, X_test, Y_test, k, l)
            
            # 3. Store and Print
            for_each_l.append((y_pred, accuracy))
            print_results(k, l, accuracy)
            
        predictions_accuracies.append(for_each_l)
        print(f"{'-'*40}")
        
    return predictions_accuracies

def plot_anomalies(X_train, X_test_AD, anomaly_indices):
    """
    Plot normal points in blue, anomalies in red, and training data in faint black.
    """
    
    # 1. Plot Training Data (Background)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c='black', alpha=0.01, label='Training Data', s=10)
    
    # 2. Separate Normal vs Anomaly points from AD_test
    # Create a boolean mask for all indices
    is_anomaly = np.zeros(len(X_test_AD), dtype=bool)
    is_anomaly[anomaly_indices] = True
    
    X_normal = X_test_AD[~is_anomaly]
    X_anomaly = X_test_AD[is_anomaly]
    
    # 3. Plot Normal Points (Blue)
    plt.scatter(X_normal[:, 0], X_normal[:, 1], c='blue', s=20, label='Normal Points', alpha=0.6)
    
    # 4. Plot Anomalies (Red)
    plt.scatter(X_anomaly[:, 0], X_anomaly[:, 1], c='red', s=40, label='Anomalies', edgecolor='k')
    
    plt.title("Anomaly Detection (Top 50 Outliers)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.show()

def main():
    np.random.seed(0)
    
    # 1. Load Data
    X_train, Y_train, X_test, Y_test = load_data()
    
    K = [1, 10, 100, 1000, 3000]
    L = ["l1", "l2"]
    
    # 2. Run All Experiments
    predictions_accuracies = run_all_experiments(X_train, Y_train, X_test, Y_test, K, L)
    # prediciction accuraceis row are different K values and columns are different L values
    kmax = knn.KNNClassifier(k=1, distance_metric='l2')
    kmax.fit(X_train, Y_train)
    kmin = knn.KNNClassifier(k=3000, distance_metric='l2')
    kmin.fit(X_train, Y_train)
    kmaxl1 = knn.KNNClassifier(k=1, distance_metric='l1')
    kmaxl1.fit(X_train, Y_train)
    # helper.plot_decision_boundaries(kmax,X_test,Y_test, "KNN Decision Boundaries (k=1, L2)")
    # helper.plot_decision_boundaries(kmin,X_test,Y_test, "KNN Decision Boundaries (k=3000, L2)")
    # helper.plot_decision_boundaries(kmaxl1,X_test,Y_test, "KNN Decision Boundaries (k=1, L1)")
    ad = pd.read_csv("AD_test.csv")
    X_Test_AD = ad[["long", "lat"]].values
    ad_model = knn.KNNClassifier(k=5, distance_metric='l2')
    ad_model.fit(X_train, Y_train)
    distances, _ = ad_model.knn_distance(X_Test_AD)
    total_distance_per_prediction = np.sum(distances, axis=1)
    dirty_indices = np.argsort(total_distance_per_prediction)[-50:]
    plot_anomalies(X_train, X_Test_AD, dirty_indices)
    
    



if __name__ == "__main__":
    main()
