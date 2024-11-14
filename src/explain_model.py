import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
import os

# Paths to model and vectorizer
MODEL_PATH = 'models/sentiment_model.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
DATA_PATH = 'data/processed/electronics_preprocessed.csv'

# Load the model and vectorizer
print("Loading the model and vectorizer...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

# Load a sample of the dataset for analysis
print("Loading the dataset...")
df = pd.read_csv(DATA_PATH).sample(1000, random_state=42)  # Use a sample for speed

# Remove rows with NaN or empty values in 'cleaned_text'
df = df.dropna(subset=['cleaned_text'])
df = df[df['cleaned_text'].str.strip() != '']

# Transform the text data
X_sample = vectorizer.transform(df['cleaned_text'])

# Initialize SHAP
print("Calculating SHAP values...")
explainer = shap.LinearExplainer(model, X_sample, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_sample)

# Ensure the plots directory exists
if not os.path.exists('plots'):
    os.makedirs('plots')

# Global visualization: Feature Importance with improved clarity
print("Creating improved feature importance plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values,
    features=X_sample,
    feature_names=vectorizer.get_feature_names_out(),
    show=False,
    plot_size=(10, 6),
    color_bar_label="Feature value"
)
plt.title("SHAP Feature Importance (Improved)")
plt.savefig('plots/feature_importance_improved.png', bbox_inches='tight', dpi=300)
plt.close()
print("Improved feature importance plot saved as 'feature_importance_improved.png'")

# Local visualization: Explanation for specific reviews
print("Creating local explanation plots for multiple reviews...")
for idx in range(5):  # Analyze the first 5 reviews
    print(f"Creating local explanation plot for review index {idx}...")
    shap.force_plot(
        explainer.expected_value,
        shap_values[idx, :],
        feature_names=vectorizer.get_feature_names_out(),
        matplotlib=True,
        show=False
    )
    plt.savefig(f'plots/local_explanation_{idx}.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved explanation for review {idx} as 'local_explanation_{idx}.png'")

# Analysis of negative reviews
print("Creating explanations for negative reviews...")
negative_reviews = df[df['sentiment'] == 0]
negative_sample = vectorizer.transform(negative_reviews['cleaned_text'].sample(5, random_state=42))
shap_values_negative = explainer.shap_values(negative_sample)

for idx, review in enumerate(negative_reviews.sample(5, random_state=42)['cleaned_text']):
    print(f"Explaining negative review {idx}: {review}")
    shap.force_plot(
        explainer.expected_value,
        shap_values_negative[idx, :],
        feature_names=vectorizer.get_feature_names_out(),
        matplotlib=True,
        show=False
    )
    plt.savefig(f'plots/negative_local_explanation_{idx}.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved negative review explanation {idx} as 'negative_local_explanation_{idx}.png'")

print("All plots saved in 'plots/'")