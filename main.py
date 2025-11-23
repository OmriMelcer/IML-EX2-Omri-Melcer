import pandas as pd
import knn
import numpy as np
import helpers as helper
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb

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
    #plot_anomalies(X_train, X_Test_AD, dirty_indices)
    #helper.decision_tree_demo()
    
    # Load validation data
    validation = pd.read_csv("validation.csv")
    X_val = validation[["long", "lat"]].values
    Y_val = validation["state"].values
    
    # # Define hyperparameter grid
    # depths = [1, 2, 4, 6, 10, 20, 50, 100]
    # max_leaf_nodes = [50, 100, 1000]
    
    # # Create results DataFrame
    # results = []
    # for depth in depths:
    #     for max_leaf in max_leaf_nodes:
    #         model = DecisionTreeClassifier(max_depth=depth, max_leaf_nodes=max_leaf)
    #         model.fit(X_train, Y_train)
    #         val_acc = np.mean(model.predict(X_val) == Y_val)
    #         test_acc = np.mean(model.predict(X_test) == Y_test)
    #         results.append({
    #             'depth': depth,
    #             'max_leaf_nodes': max_leaf,
    #             'val_accuracy': val_acc,
    #             'test_accuracy': test_acc,
    #             'model': model
    #         })
    
    # df_results = pd.DataFrame(results)
    
    # # Sort by validation accuracy and display
    # df_sorted_val = df_results.sort_values('val_accuracy', ascending=False)
    # print("\n" + "="*60)
    # print("VALIDATION SET RESULTS (sorted by validation accuracy)")
    # print("="*60)
    # for _, row in df_sorted_val.iterrows():
    #     print(f"Depth={row['depth']:<3} | Max Leaf Nodes={row['max_leaf_nodes']:<4} | Val Acc={row['val_accuracy']:.4f}")
    
    # # Find best model based on validation
    # best_idx = df_results['val_accuracy'].idxmax()
    # best_model_info = df_results.iloc[best_idx]
    
    # print("\n" + "="*60)
    # print("BEST MODEL (based on validation accuracy)")
    # print("="*60)
    # print(f"Depth={best_model_info['depth']}, Max Leaf Nodes={best_model_info['max_leaf_nodes']}")
    # print(f"Validation Accuracy: {best_model_info['val_accuracy']:.4f}")
    # print(f"Test Accuracy: {best_model_info['test_accuracy']:.4f}")
    
    # # Optional: Show test set results sorted
    # print("\n" + "="*60)
    # print("TEST SET RESULTS (sorted by test accuracy)")
    # print("="*60)
    # df_sorted_test = df_results.sort_values('test_accuracy', ascending=False)
    # for _, row in df_sorted_test.iterrows():
    #     print(f"Depth={row['depth']:<3} | Max Leaf Nodes={row['max_leaf_nodes']:<4} | Test Acc={row['test_accuracy']:.4f}")
    # delta_accuarcy = df_results['test_accuracy']-df_results['val_accuracy']
    # print("\n" + "="*60)
    # print("DELTA ACCURACY (Test - Validation)")
    # print("="*60)
    # for i, row in df_results.iterrows():
    #     delta = row['test_accuracy'] - row['val_accuracy']
    #     print(f"Depth={row['depth']:<3} | Max Leaf Nodes={row['max_leaf_nodes']:<4} | Delta Acc={delta:.4f}")
    # print(df_results[df_results['max_leaf_nodes'] == 50])
    # helper.plot_decision_boundaries(best_model_info['model'], X_test, Y_test, 
    #                               title=f"Decision Boundaries (Depth={best_model_info['depth']}, Max Leaf Nodes={best_model_info['max_leaf_nodes']})")
    # best_idx_50_leaves = df_results[df_results['max_leaf_nodes'] == 50]['test_accuracy'].idxmax()
    # best_model_50_leaves_info = df_results.iloc[best_idx_50_leaves]
    # helper.plot_decision_boundaries(best_model_50_leaves_info['model'], X_test, Y_test, 
    #                               title=f"Decision Boundaries (Depth={best_model_50_leaves_info['depth']}, Max Leaf Nodes={best_model_50_leaves_info['max_leaf_nodes']})")
    # best_idx_max_six_depth = df_results[df_results['depth'] <= 6]['test_accuracy'].idxmax()
    # best_model_max_six_depth_info = df_results.iloc[best_idx_max_six_depth]
    # helper.plot_decision_boundaries(best_model_max_six_depth_info['model'], X_test, Y_test, 
    #                               title=f"Decision Boundaries (Depth={best_model_max_six_depth_info['depth']}, Max Leaf Nodes={best_model_max_six_depth_info['max_leaf_nodes']})")
    # # Random Forest Experiments
    # print("\n" + "="*60)
    # print("RANDOM FOREST EXPERIMENTS")
    # print("="*60)
    
    # rf_configs = [
    #     {'n_estimators': 300, 'max_depth': 6, 'name': 'RF (300 trees, depth=6)'},
    #     {'n_estimators': 100, 'max_depth': 10, 'name': 'RF (100 trees, depth=10)'},
    #     {'n_estimators': 100, 'max_depth': 20, 'name': 'RF (100 trees, depth=20)'},
    #     {'n_estimators': 50, 'max_depth': None, 'name': 'RF (50 trees, no depth limit)'},
    # ]
    
    # rf_results = []
    # for config in rf_configs:
    #     rf_model = RandomForestClassifier(
    #         n_estimators=config['n_estimators'], 
    #         max_depth=config['max_depth'], 
    #         n_jobs=4,
    #         random_state=0
    #     )
    #     rf_model.fit(X_train, Y_train)
        
    #     val_acc = np.mean(rf_model.predict(X_val) == Y_val)
    #     test_acc = np.mean(rf_model.predict(X_test) == Y_test)
        
    #     rf_results.append({
    #         'name': config['name'],
    #         'n_estimators': config['n_estimators'],
    #         'max_depth': config['max_depth'],
    #         'val_accuracy': val_acc,
    #         'test_accuracy': test_acc,
    #         'model': rf_model
    #     })
        
    #     print(f"{config['name']:<35} | Val Acc={val_acc:.4f} | Test Acc={test_acc:.4f}")
    
    # # Plot the original RF (300 trees, depth=6)
    # original_rf = rf_results[0]
    # helper.plot_decision_boundaries(original_rf['model'], X_test, Y_test, 
    #                               title=f"Random Forest Decision Boundaries (n_estimators=300, max_depth=6)")
    
    print("\nTraining XGBoost model...")
    xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6, n_jobs=4, random_state=0, learning_rate=0.1)  # Changed n_jobs=4 to n_jobs=1
    xgb_model.fit(X_train, Y_train)
    print("XGBoost training complete. Computing accuracies...")
    val_acc = np.mean(xgb_model.predict(X_val) == Y_val)
    test_acc = np.mean(xgb_model.predict(X_test) == Y_test)
    print(f"XGBoost (n_estimators=300, max_depth=6) | Val Acc={val_acc:.4f} | Test Acc={test_acc:.4f}")
    helper.plot_decision_boundaries(xgb_model, X_test, Y_test, 
                                  title=f"XGBoost Decision Boundaries (n_estimators=300, max_depth=6)")
    

if __name__ == "__main__":
    main()
