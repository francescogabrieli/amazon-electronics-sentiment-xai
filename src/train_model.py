import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
import numpy as np

# Percorso del dataset preprocessato
DATA_PATH = 'data/processed/electronics_preprocessed.csv'

# Carica il dataset preprocessato
print("Caricamento del dataset...")
df = pd.read_csv(DATA_PATH)

print("Rimozione di valori NaN...")
df = df.dropna(subset=['cleaned_text'])

# Verifica che non ci siano valori NaN residui
print(f"Valori NaN residui: {df['cleaned_text'].isna().sum()}")

# Suddividi i dati in train e test
print("Suddivisione in train e test...")
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42
)

# Trasforma il testo in vettori TF-IDF
print("Creazione dei vettori TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Calcola i pesi delle classi
print("Calcolo dei pesi delle classi...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
weights_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Pesi delle classi: {weights_dict}")

# Allena il modello con Class Weights
print("Training del modello Logistic Regression...")
model = LogisticRegression(class_weight=weights_dict, max_iter=1000)
model.fit(X_train_vec, y_train)

# Valutazione del modello
print("Valutazione del modello...")
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Salva il modello
print("Salvataggio del modello...")
os.makedirs('models', exist_ok=True)
with open('models/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Salva il vettorizzatore TF-IDF
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Training completato e modello salvato in 'models/sentiment_model.pkl'")
