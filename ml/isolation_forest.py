import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sender.helper import generate_augmented_data, generate_seed_data
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class IsolationImplementaion:
    
    def __init__(self):
        self.clf = None
        self.model_filename = "isf_model.pkl"

    def train(self, data):
        # Transform data
        features = ['pic_id', 'purpose', 'latitude', 'longitude', 'device_name', 
                    'os_version', 'manufacturer', 'cpu_info', 'platform', 'ip']
        X = data[features]

        # Preprocessing
        categorical_features = ['pic_id', 'purpose', 'device_name', 'os_version', 
                                'manufacturer', 'cpu_info', 'platform', 'ip']
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Create pipeline
        self.clf = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', IsolationForest(contamination=0.1))])

        # Train the model
        model = self.clf.fit(X)
        joblib.dump(model, self.model_filename)
        # Get the anomaly predictions
        predictions = self.clf.predict(X)
        data['status'] = ['anomaly' if x == -1 else 'normal' for x in predictions]
        data.to_csv("updated_data.csv", index=False)
        # Extract the indices of the anomalies
        anomaly_indices = np.where(predictions == -1)[0]
        print(len(anomaly_indices))
        anomalous_records = [data.iloc[i] for i in anomaly_indices]
        print(anomalous_records)

        # Visualization using scatter plot
        X_transformed = preprocessor.transform(X)  # Apply the preprocessing
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_transformed)

        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[predictions == 1, 0], X_pca[predictions == 1, 1], c='blue', label='Normal')
        plt.scatter(X_pca[predictions == -1, 0], X_pca[predictions == -1, 1], c='red', label='Anomaly')
        plt.legend()
        plt.title("Visualization of Anomaly Detection Results")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()


    def save_model(self, filename):
        joblib.dump(self.clf, filename)

    def load_model(self, filename):
        self.clf = joblib.load(filename)

    def predict(self, data):
        loaded_clf = joblib.load(self.model_filename)
        
        prediction = loaded_clf.predict(pd.DataFrame([data]))

        # Evaluasi hasil
        if prediction == -1:
            return "Anomali"
        else:
            return "Normal"


    # def train(self, data):
    #     features = ['purpose', 'latitude', 'longitude', 'device_name', 
    #             'os_version', 'manufacturer', 'cpu_info', 'platform', 'ip']
    #     X = data[features]

    #     # Create the Isolation Forest model directly without preprocessing
    #     self.clf = IsolationForest(contamination=0.05)

    #     # Train the model
    #     self.clf.fit(X)

    #     # Get the anomaly predictions
    #     predictions = self.clf.predict(X)

    #     # Extract the indices of the anomalies
    #     anomaly_indices = np.where(predictions == -1)[0]
    #     anomalous_records = [data.iloc[i] for i in anomaly_indices]
    #     print(anomalous_records)

    #     # Visualization using scatter plot
    #     # Given that the data is not transformed, PCA might not work as expected if there are categorical variables
    #     pca = PCA(n_components=2)
    #     X_pca = pca.fit_transform(X)  # Directly use X without any transformation

    #     plt.figure(figsize=(10, 6))
    #     plt.scatter(X_pca[predictions == 1, 0], X_pca[predictions == 1, 1], c='blue', label='Normal')
    #     plt.scatter(X_pca[predictions == -1, 0], X_pca[predictions == -1, 1], c='red', label='Anomaly')
    #     plt.legend()
    #     plt.title("Visualization of Anomaly Detection Results")
    #     plt.xlabel("Principal Component 1")
    #     plt.ylabel("Principal Component 2")
    #     plt.show()

    #     print(f"Total anomalies detected: {len(anomaly_indices)}")

    #     print(f"Total anomalies detected: {len(anomaly_indices)}")
    # def __init__(self):
    #     self.clf = None

    # def train(self, data):
    #     # Transform data
    #     features = ['pic_id', 'purpose', 'latitude', 'longitude', 'device_name', 
    #                 'os_version', 'manufacturer', 'cpu_info', 'platform', 'ip']
    #     X = pd.DataFrame([ [record[feature] for feature in features] for record in data ], columns=features)

    #     # Menggunakan pipeline untuk menggabungkan preprocessing
    #     numeric_features = ['latitude', 'longitude']
    #     numeric_transformer = StandardScaler()

    #     categorical_features = ['pic_id', 'purpose', 'device_name', 'os_version', 
    #                             'manufacturer', 'cpu_info', 'platform', 'ip']
    #     categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    #     preprocessor = ColumnTransformer(
    #         transformers=[
    #             ('num', numeric_transformer, numeric_features),
    #             ('cat', categorical_transformer, categorical_features)])

    #     self.clf = Pipeline(steps=[('preprocessor', preprocessor),
    #                         ('classifier', IsolationForest(contamination=0.05))])

    #     # Pelatihan model
    #     self.clf.fit(X)

    #     # Prediksi anomali
    #     predictions = self.clf.predict(X)

    #     # Mendapatkan indeks dari data yang diprediksi sebagai anomali
    #     anomaly_indices = np.where(predictions == -1)[0]
    #     anomalous_records = [data[i] for i in anomaly_indices]
    #     print(anomalous_records)

    #     print(f"Total anomali terdeteksi: {len(anomaly_indices)}")
    
    # class IsolationImplementaion:
#     def __init__(self):
#         self.clf = None
#         self.scaler_filename = 'scaler.pkl'
#         self.model_filename = 'isf_model.pkl'

#     def train(self, df):
#         # Asumsi data sudah berada dalam format DataFrame
#         X = df

#         # Menskalakan fitur numerik
#         # self.scaler = StandardScaler()
#         # X_scaled = self.scaler.fit_transform(X)

#         self.clf = IsolationForest(contamination=0.1)
#         # self.clf.fit(X_scaled)
#         self.clf.fit(X)

#         # Menyimpan scaler ke disk
#         # joblib.dump(self.scaler, self.scaler_filename)

#         # Prediksi anomali
#         predictions = self.clf.predict(X)

#         # Mendapatkan indeks dari data yang diprediksi sebagai anomali
#         anomaly_indices = np.where(predictions == -1)[0]
#         anomalous_records = [X.iloc[i].to_dict() for i in anomaly_indices]
#         print(anomalous_records)

#         print(f"Total anomali terdeteksi: {len(anomaly_indices)}")

#     def save_model(self):
#         joblib.dump(self.clf, self.model_filename)

#     def load_model(self):
#         self.clf = joblib.load(self.model_filename)
#         # self.scaler = joblib.load(self.scaler_filename)

#     def predict(self, new_record):
#         X = pd.DataFrame([new_record])
#     # Memuat scaler dari disk
#         # scaler = joblib.load(self.scaler_filename)
        
#         # Menskalakan new_record dengan scaler yang telah dilatih sebelumnya
#         # X_new = scaler.transform(X)

#         # Prediksi dengan model
#         # prediction = self.clf.predict(X_new)
#         prediction = self.clf.predict(X)

#         # Evaluasi hasil
#         if prediction == -1:
#             return "Anomali"
#         else:
#             return "Normal"