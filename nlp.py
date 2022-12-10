!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master

import gluonnlp as nlp
import pandas as pd
from torch.utils.data import Dataset

from google.colab import drive
drive.mount('/content/drive')

class BERTDataset(Dataset):#tokenizing
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

a=pd.read_csv('/content/drive/MyDrive/train_data.csv')