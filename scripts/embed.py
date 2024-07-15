import os
import sys
import argparse

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.append('.')
from nn4sa.models.embedder import MistralEmbedder

parser = argparse.ArgumentParser(description='Embed documents')

parser.add_argument("--cache_dir", type=str, default=None)
parser.add_argument("--documents_path", type=str, default="./data/senti_data.csv")
parser.add_argument("--corpus_column_name", type=str, default="text")
parser.add_argument("--batch_size", type=int, default=1)

def main():
    args = parser.parse_args()
    documents_path = args.documents_path
    corpus_column_name = args.corpus_column_name
    cache_dir = args.cache_dir
    batch_size = args.batch_size

    save_dir = os.path.dirname(documents_path)

    input_texts = pd.read_csv(documents_path)[corpus_column_name].tolist()

    model = MistralEmbedder(cache_dir=cache_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    result = []
    for i in tqdm(range(0, len(input_texts), batch_size)):
        batch = input_texts[i:i+batch_size]
        batched_embeddings = model.embed(batch)
        result.append(batched_embeddings)
    
    batch_embeddings_concat = np.vstack(result)
    np.save(os.path.join(save_dir, "embeddings.npy"), batch_embeddings_concat)

if __name__ == "__main__":
    main()