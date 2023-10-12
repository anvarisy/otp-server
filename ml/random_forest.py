import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import plot_tree
import numpy as np


class RandomForestImplementation:
    def __init__(self) -> None:
        pass
    
    def train(self, data):
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
        # feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)

        # Membagi data menjadi set pelatihan dan set pengujian
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.25, random_state=50)

        # Inisialisasi dan pelatihan model Random Forest
        clf = RandomForestClassifier(n_estimators=20, class_weight='balanced', random_state=50)
        clf.fit(X_train, y_train)
        # for idx, single_tree in enumerate(clf.estimators_[:10]):
        # # for idx, single_tree in enumerate(clf.estimators_[98:100], start=99):

        #     # Mendapatkan fitur yang digunakan oleh pohon tersebut
        #     plt.figure(figsize=(20, 10))
            
        #     # Menggunakan nama fitur yang telah di-extract
        #     plot_tree(single_tree, filled=True, feature_names=feature_names, class_names=True, rounded=True)
            
        #     used_features_indices = np.unique(single_tree.tree_.feature[single_tree.tree_.feature >= 0])
        #     used_features_names = feature_names[used_features_indices]

        #     print(f"Fitur yang digunakan oleh pohon ke-{idx + 1}:")
        #     print(used_features_names)
            
        #     plt.show()
        joblib.dump(clf, 'random_forest_model.pkl')
        joblib.dump(preprocessor, 'preprocessor.pkl')
        # Prediksi dan evaluasi performa
        y_pred = clf.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confussion Matrix:", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))
        tree_predictions = [tree.predict(X_test) for tree in clf.estimators_]
        tree_predictions_matrix = np.array(tree_predictions).T
        
        #How vote works
        # votes = []
        # for row in tree_predictions_matrix:
        #     unique, counts = np.unique(row, return_counts=True)
        #     votes.append(dict(zip(unique, counts)))
        # for i, vote in enumerate(votes):
        #     print(f"Sampel ke-{i + 1} mendapatkan suara: {vote}")


        svd = TruncatedSVD(n_components=2)
        X_pca = svd.fit_transform(X_processed)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[y == 'normal', 0], X_pca[y == 'normal', 1], color='blue', label='Normal')
        plt.scatter(X_pca[y == 'anomaly', 0], X_pca[y == 'anomaly', 1], color='red', label='Anomaly')
        plt.title('PCA of Dataset')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()

    
    def predict(self, data):
        model = joblib.load('random_forest_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')  # Memuat preprocessor
        data_df = pd.DataFrame([data])
        # Melakukan pra-pemrosesan pada data
        data_processed = preprocessor.transform(data_df)
        result = model.predict(data_processed)
        print(result)
        return result[0]