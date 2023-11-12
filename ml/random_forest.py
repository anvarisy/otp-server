import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import plot_tree
import numpy as np
import shutil

def empty_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

class RandomForestImplementation:
    def __init__(self) -> None:
        pass
    
    def train(self, data, tree_count, test_count):
        X = data.drop('status', axis=1)
        y = data['status']

        # Pra-pemrosesan
        categorical_features = ['pic_id', 'purpose','latitude', 'longitude', 'device_name', 'os_version', 
                                'manufacturer', 'cpu_info', 'platform', 'ip']
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features)
            ])

        X_processed = preprocessor.fit_transform(X)
        feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)

        # Membagi data menjadi set pelatihan dan set pengujian
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=test_count, random_state=42)

        # Inisialisasi dan pelatihan model Random Forest
        clf = RandomForestClassifier(n_estimators=tree_count,  class_weight='balanced', random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        conf_matrix_html = conf_matrix.tolist()
        # report = classification_report(y_test, y_pred, zero_division=1, output_dict=True)
        precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred)
        # precision = report['macro avg']['precision']
        # recall = report['macro avg']['recall']
        # f1score = report['macro avg']['f1-score']
        # support = report['macro avg']['support']
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1score': f1_score,
            'support': support,
            'conf_matrix': conf_matrix_html,
        }
        joblib.dump(clf, 'random_forest_model.pkl')
        joblib.dump(preprocessor, 'preprocessor.pkl')

        svd = TruncatedSVD(n_components=2)
        X_pca = svd.fit_transform(X_processed)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[y == 'normal', 0], X_pca[y == 'normal', 1], color='blue', label='Normal')
        plt.scatter(X_pca[y == 'anomaly', 0], X_pca[y == 'anomaly', 1], color='red', label='Anomaly')
        plt.title('Hasil')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plot_filename = 'scatterplot.png'
        plt.savefig(f'static/{plot_filename}')  # Simpan plot ke dalam direktori static
        plt.close()
        
        tree_predictions = [tree.predict(X_test) for tree in clf.estimators_]
        tree_predictions_matrix = np.array(tree_predictions).T
        voting_results = [dict(zip(unique, counts)) for row in tree_predictions_matrix for unique, counts in [np.unique(row, return_counts=True)]]
        voting_results_html = [f"Sampel ke-{i+1}: {vote}" for i, vote in enumerate(voting_results)]
        
        tree_filenames = []
        empty_folder('static/trees')
        for idx, single_tree in enumerate(clf.estimators_):
            plt.figure(figsize=(20, 10))
            plot_tree(single_tree, filled=True, feature_names=feature_names, class_names=True, rounded=True)
            used_features_indices = np.unique(single_tree.tree_.feature[single_tree.tree_.feature >= 0])
            used_features_names = feature_names[used_features_indices]
            # print(f"Fitur yang digunakan oleh pohon ke-{idx + 1}:")
            # print(used_features_names)
            tree_filename = f'tree_{idx}.png'
            plt.savefig(f'static/trees/{tree_filename}')
            tree_filenames.append(tree_filename)
        return {
        'metrics': metrics,
        'plot_filename': plot_filename,
        'voting_results': voting_results_html,
        'tree_filenames': tree_filenames
        }
    
    def predict(self, data):
        model = joblib.load('random_forest_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')  # Memuat preprocessor
        data_df = pd.DataFrame([data.dict()])
        # categorical_features = ['pic_id', 'purpose','latitude', 'longitude', 'device_name', 'os_version', 
        #                         'manufacturer', 'cpu_info', 'platform', 'ip']
        # feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        # Melakukan pra-pemrosesan pada data
        data_processed = preprocessor.transform(data_df)
        result = model.predict(data_processed)
        votes = []
        tree_filenames = []
        for idx, single_tree in enumerate(model.estimators_):
            # Mendapatkan prediksi dari pohon individual
            tree_vote = single_tree.predict(data_processed)
            votes.append(tree_vote[0])
            
            # plt.figure(figsize=(20, 10))
            # plot_tree(single_tree, filled=True, feature_names=feature_names, class_names=True, rounded=True)
            # used_features_indices = np.unique(single_tree.tree_.feature[single_tree.tree_.feature >= 0])
            # used_features_names = feature_names[used_features_indices]
            # print(f"Fitur yang digunakan oleh pohon ke-{idx + 1}:")
            # print(used_features_names)
            # tree_filename = f'tree_{idx}.png'
            # plt.savefig(f'static/model_trees/{tree_filename}')
            # tree_filenames.append(tree_filename)

            # Kembalikan hasil prediksi dan vote
        return {'prediction': result[0], 'votes': votes, 'tree_filesname':tree_filenames}