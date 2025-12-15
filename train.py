import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# carregar dados
df = pd.read_csv("data/dataset.csv")

X = df["texto"]
y = df["label"]

# dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# pipeline de IA
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("modelo", MultinomialNB())
])

# treinar modelo
pipeline.fit(X_train, y_train)

# previsões no conjunto de teste
y_pred = pipeline.predict(X_test)

# métricas
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("📊 Avaliação do Modelo")
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

# salvar modelo treinado
with open("model/modelo.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Modelo treinado, avaliado e salvo!")
