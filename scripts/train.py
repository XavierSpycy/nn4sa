import argparse
import logging
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

sys.path.append(".")

from nn4sa.datasets import SADataset
from nn4sa.models import BertClassifier, CLSHead, RNNClassifier
from nn4sa.metrics import accuracy #, precision, recall
from nn4sa.utils import seed_everything, train
from nn4sa.callbacks import EarlyStopping, History
from nn4sa.preprocessing import RNNTokenizer

parser = argparse.ArgumentParser(description='Fine-tune a Bert model for sentiment analysis')
parser.add_argument("--method", type=str, default=None, choices=["rnn", "lstm", "gru", "embed", "bert"])
parser.add_argument("--data_path", type=str, default="./data/senti_data.csv")
parser.add_argument("--embed_path", type=str, default=None) # "./data/embeddings.npy"
parser.add_argument("--bert_model_path", type=str, default=None) # "./models/bert-base-uncased"
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=256)

logging.basicConfig(level=logging.INFO)

def main(args):
    method = args.method
    if method not in {"rnn", "lstm", "gru", "embed", "bert"}:
        raise ValueError("Invalid method. Choose from 'rnn', 'embed', 'bert'")
    data_path = args.data_path
    embed_path = args.embed_path
    bert_model_path = args.bert_model_path
    seed = args.seed
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    
    seed_everything(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = pd.read_csv(data_path)
    logging.info(("Data loaded successfully"))

    labels = data['sentiment']
    labels = labels.replace({4: 1})
    labels = labels.tolist()

    if method in {"rnn", "lstm", "gru"}:
        inputs = data['text'].tolist()
        tokenizer = RNNTokenizer()
        logging.info("Building vocabulary...")
        tokenizer.build_vocabulary(inputs)
        logging.info("Vocabulary built successfully")
        vocab_size = tokenizer.vocab_size
        model = RNNClassifier(output_dim=2, rnn_type=method.upper(), vocab_size=vocab_size)
    elif method == "bert":
        inputs = data['text'].tolist()
        tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        model = BertClassifier(output_dim=2, model_id_or_path=bert_model_path)
    else:
        inputs = np.load(embed_path)
        model = CLSHead(output_dim=2, input_dim=inputs.shape[1])

    input_train, input_test, label_train, label_test = train_test_split(
        inputs, labels, 
        test_size=0.1, stratify=labels,
        random_state=seed)
    
    def rnn_fn(batch):
        texts, labels = zip(*batch)
        encoded_texts = [torch.tensor(tokenizer.tokenize(text), dtype=torch.long) for text in texts]
        padded_texts = pad_sequence(encoded_texts, batch_first=True, padding_value=tokenizer.vocab["<pad>"])
        labels = torch.tensor(labels)
        return padded_texts, labels
    
    def bert_fn(batch):
        texts, labels = zip(*batch)
        encoded_inputs = tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True)
        labels = torch.tensor(labels)
        return encoded_inputs, labels
    
    def embed_fn(batch):
        embeds, labels  = zip(*batch)
        embeds = torch.from_numpy(np.array(embeds)).to(torch.float32)
        labels = torch.tensor(labels)
        return embeds, labels

    func_map = {
        "bert": bert_fn,
        "embed": embed_fn}

    train_dataset = SADataset(input_train, label_train)
    eval_dataset = SADataset(input_test, label_test)
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=4, 
        pin_memory=True, collate_fn=func_map.get(method, rnn_fn))
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=4, 
        pin_memory=True, collate_fn=func_map.get(method, rnn_fn))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    logging.info("Starting training...")

    train(model, 
          device=device,
          train_dataloader=train_dataloader, 
          eval_dataloader=eval_dataloader, 
          optimizer=optimizer,
          criterion=criterion,
          num_epochs=num_epochs,
          early_stopping=EarlyStopping(),
          history=History({'accuracy': accuracy}))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)