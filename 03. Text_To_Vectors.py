import requests
import os
import json
import pandas as pd
import numpy as np 
import joblib
from sklearn.metrics.pairwise import cosine_similarity


# Function to convert text to embedding in batches(not possible to convert whole text at a time)
def create_embedding(text_list, batch_size=32):
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": batch
        })
        data = r.json()
        if 'embeddings' not in data:
            print(f"Error on batch {i}-{i+batch_size}:", data)
            raise RuntimeError(f"Embedding failed: {data}")
        all_embeddings.extend(data['embeddings'])
        print(f"Embedded {i+len(batch)}/{len(text_list)}")
    return all_embeddings


# Code to use the function on json file
jsons = os.listdir("jsons")
my_dicts = []

for json_file in jsons:
    print(f"\nProcessing {json_file}...")
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)

    texts = [c['text'] for c in content['chunks']]
    embeddings = create_embedding(texts, batch_size=32)

    for i, chunk in enumerate(content['chunks']):
        chunk['embedding'] = embeddings[i]
        my_dicts.append(chunk)


# Store the embeddings along with other info in a DataFrame
df = pd.DataFrame.from_records(my_dicts)


# Dump the DataFrame along with embeddings using joblib
joblib.dump(df, 'joblib/embeddings.joblib')