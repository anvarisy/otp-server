import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib

class LinkageImplementation:
    def __init__(self):
        self.scaler_filename = 'scaler_linkage.pkl'
        self.linkage_matrix = None

    def preprocess(self, df):
        # Asumsi data sudah berada dalam format DataFrame
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(df)

        # Menyimpan scaler ke disk
        joblib.dump(self.scaler, self.scaler_filename)

        return X_scaled

    def compute_linkage_matrix(self, X_scaled):
        # Menghitung linkage matrix untuk hierarchical clustering
        self.linkage_matrix = linkage(X_scaled, method='ward')  # 'ward' adalah metode yang umum digunakan, tetapi Anda bisa menggantinya
        return self.linkage_matrix

    def plot_dendrogram(self):
        # Plot dendrogram
        plt.figure(figsize=(10, 7))
        dendrogram(self.linkage_matrix)
        plt.title("Dendrogram")
        plt.xlabel("Data Points")
        plt.ylabel("Euclidean Distances")
        plt.show()

    def load_scaler(self):
        self.scaler = joblib.load(self.scaler_filename)