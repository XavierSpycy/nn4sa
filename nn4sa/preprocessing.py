from collections import Counter
from multiprocessing import Pool

import spacy

class RNNTokenizer:
    nlp = spacy.load("en_core_web_sm")
    
    def __init__(self) -> None:
        self.vocab = {"<pad>": 0, "<unk>": 1}
    
    def preprocess(self, text):
        doc = self.nlp(text)
        return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    
    def process_texts(self, texts):
        preprocessed_texts = []
        for doc in self.nlp.pipe(texts, batch_size=20):
            preprocessed_texts.extend(self.preprocess(doc))
        return preprocessed_texts
    
    def build_vocabulary(self, texts, num_processes=12):
        chunk_size = int(len(texts) / num_processes)
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        with Pool(processes=num_processes) as pool:
            results = pool.map(self.process_texts, chunks)
        word_counts = Counter()
        for result in results:
            word_counts.update(result)

        start_index = max(self.vocab.values()) + 1
        for word, idx in zip(word_counts, range(start_index, start_index + len(word_counts))):
            self.vocab[word] = idx

    def tokenize(self, text, max_len=None):
        tokens = self.preprocess(text)
        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        if max_len is not None:
            token_ids = token_ids[:max_len]
            token_ids += [self.vocab["<pad>"]] * (max_len - len(token_ids))
        return token_ids
    
    @property
    def vocab_size(self):
        return len(self.vocab)