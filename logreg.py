from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import json
import joblib

def train(output_file):
    with open("data.json", "r") as f:
        data = json.load(f)

    labels = {
        "command": 0,
        "question": 1,
    }

    model = SentenceTransformer("all-MiniLM-L6-v2") # all-mpnet-base-v2

    X = [model.encode(text) for text, label in data]
    y = [labels[label] for text, label in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=38)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    joblib.dump(clf, f"{output_file}.pkl")

def test():

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")


def predict(text):
    clf = joblib.load("logreg_model.pkl")
    model = SentenceTransformer("all-MiniLM-L6-v2") # all-mpnet-base-v2
    text_vector = model.encode(text)
    prediction = clf.predict([text_vector])
    print("Command" if prediction[0] == 0 else "Question")

if __name__ == "__main__":
    # train("logreg_model")
    predict("What is the weather today?")

