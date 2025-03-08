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
    labels = determine_labels(kmeans, data, num_clusters)

    kmeans_data = {
        "model": kmeans,
        "labels": labels
    }

    joblib.dump(kmeans_data, f"models/{output_file}.pkl", compress=3)

def determine_labels(kmeans, data, num_clusters):
    labels = {}
    clusters = {i: [] for i in range(num_clusters)}
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(data[i][0])

    for cluster_id, group in clusters.items():
        print(f"Cluster {cluster_id}:")
        for i, text in enumerate(group):
            print(f"  - {text}")
            if i >= 10:
                break
        label = input(f"Label for cluster {cluster_id}: ")
        labels[cluster_id] = label
    return labels


def predict(text):
    model = SentenceTransformer("all-MiniLM-L6-v2")  # all-mpnet-base-v2
    text_vector = model.encode(text)
    kmeans_data = joblib.load("models/kmeans_model.pkl")
    kmeans = kmeans_data["model"]
    labels = kmeans_data["labels"]
    cluster_id = kmeans.predict([text_vector])[0]
    return labels[cluster_id]


if __name__ == "__main__":
    train("kmeans_model")
    # prediction = predict("traffic?")
    # print(prediction)
