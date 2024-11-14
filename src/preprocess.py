import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')  

DATA_PATH = 'data/Electronics_5.json'

# Function for cleaning text
def clean_text(text):
    text = re.sub(r'\d+', '', text)  
    text = text.translate(str.maketrans('', '', string.punctuation))  
    text = text.lower()  
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Function for processing dataset in chunks
def preprocess_data_in_chunks(input_path, output_path, chunk_size=5000):
    chunks = pd.read_json(input_path, lines=True, chunksize=chunk_size)
    processed_chunks = []
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}...")
        # Cleaning and preprocessing of chunk
        chunk = chunk[['reviewText', 'overall']].dropna()
        chunk['sentiment'] = chunk['overall'].apply(lambda x: 1 if x > 3 else 0)
        chunk['cleaned_text'] = chunk['reviewText'].apply(clean_text)
        processed_chunks.append(chunk[['cleaned_text', 'sentiment']])
    
    # Combines all chunks and save
    processed_df = pd.concat(processed_chunks, ignore_index=True)
    processed_df.to_csv(output_path, index=False)
    print("Preprocessing completato!")

if __name__ == "__main__":
    preprocess_data_in_chunks(DATA_PATH, 'data/processed/electronics_preprocessed.csv')
