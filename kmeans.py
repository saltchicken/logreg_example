from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import json
import joblib

def train(output_file):

    with open("data/data.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    # Convert text to vector representations
    model = SentenceTransformer("all-MiniLM-L6-v2") # all-mpnet-base-v2
    X = [model.encode(text) for text, _ in data]

    # Apply K-means clustering
    num_clusters = 2  # Adjust as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    joblib.dump(kmeans, f"models/{output_file}.pkl", compress=3)

# def print_model():
#     # Print clusters
#     clusters = {i: [] for i in range(num_clusters)}
#     for i, label in enumerate(kmeans.labels_):
#         clusters[label].append(data[i][0])
#
#     for cluster_id, group in clusters.items():
#         print(f"Cluster {cluster_id}:")
#         for text in group:
#             print(f"  - {text}")
#         print()
#
def predict(text):
    model = SentenceTransformer("all-MiniLM-L6-v2")  # all-mpnet-base-v2
    text_vector = model.encode(text)
    kmeans = joblib.load("models/kmeans_model.pkl")
    cluster_id = kmeans.predict([text_vector])[0]
    return cluster_id


if __name__ == "__main__":
    # train("model")
    prediction = predict("What was the weather today?")
    print(prediction)
