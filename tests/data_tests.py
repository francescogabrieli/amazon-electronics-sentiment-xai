import pandas as pd


# Carica il dataset preprocessato
df = pd.read_csv('data/processed/electronics_preprocessed.csv')

# Mostra le prime righe
print(df.head())

# Verifica la dimensione del dataset
print(f"Total number of preprocessed reviews: {len(df)}")


# Conta le recensioni positive e negative
print(df['sentiment'].value_counts())



