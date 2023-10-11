from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import matplotlib.pyplot as plt

class KMeansImplementation:
    def __init__(self, n_clusters=2):
        self.clf = None
        self.n_clusters = n_clusters
        self.scaler_filename = 'kmeans_scaler.pkl'
        self.model_filename = 'kmeans_model.pkl'

    def train(self, df):
        # Asumsi data sudah berada dalam format DataFrame
        X = df

        # Menskalakan fitur numerik
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.clf = KMeans(n_clusters=self.n_clusters)
        self.clf.fit(X_scaled)

        # Menyimpan scaler ke disk
        joblib.dump(self.scaler, self.scaler_filename)

        # Prediksi cluster untuk data pelatihan
        cluster_labels = self.clf.predict(X_scaled)
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='rainbow')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('KMeans Clustering Result')
        plt.show()
        print(f"Data telah dikelompokkan ke {self.n_clusters} cluster.")

    def save_model(self):
        joblib.dump(self.clf, self.model_filename)

    def load_model(self):
        self.clf = joblib.load(self.model_filename)
        self.scaler = joblib.load(self.scaler_filename)

    def predict(self, new_record):
        X = pd.DataFrame([new_record])
        # Memuat scaler dari disk
        scaler = joblib.load(self.scaler_filename)
        
        # Menskalakan new_record dengan scaler yang telah dilatih sebelumnya
        X_new = scaler.transform(X)

        # Prediksi cluster untuk data baru
        cluster_label = self.clf.predict(X_new)

        return f"Data termasuk ke dalam cluster {cluster_label[0]}"
