import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
import torch

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

class loadContent():
    def __init__(self, fn):
        super().__init__()
        self.init_tab = pd.read_csv(fn)
        self.init_tab = self.init_tab.dropna()
        self.init_tab = self.init_tab.reset_index()
        self.__extract_content()
        self.content = {'date': self.__date, 'content': self.__content}

    def __sentens_tokenize(self, doc):
        sent = doc.split('. ')
        return [s for s in sent if s != '']

    def __extract_content(self):
        self.__content = []
        self.__date = []
        for i, a in enumerate(self.init_tab['release date']):
            try:
                if a.find('hour') > -1:
                    d = self.init_tab['load date'].loc[i]
                else:
                    d = pd.to_datetime(a[3:]).date()
                self.__date.append(str(d))
                self.__content.append(self.__sentens_tokenize(self.init_tab['content'].loc[i]))
            except Exception:
                print(f'Error SENTENS TOKENIZER: index {i}')
                print(self.init_tab['content'].loc[i])

class finBertSentiment():
    def __init__(self, path, content):
        super().__init__()
        self.path = path
        self.content = content

    def __get_last_index(self, path):
        if os.path.isfile(path):
            df = pd.read_csv(path)
            return df['date'].size
        else:
            return 0

    def __save2csv(self, path, df):
        if not os.path.isfile(path):
            df.to_csv(path)
        else:
            df.to_csv(path, mode='a', header=False)

    def computeSentiment(self, save_count=10):
        last_index = self.__get_last_index(self.path)
        count = 0
        sent_block = []
        for i, sentenses in enumerate(tqdm(self.content['content'][last_index:])):
            try:
                d = {}
                inputs = tokenizer(sentenses, return_tensors="pt", padding=True)
                outputs = finbert(**inputs)[0]
                out = outputs.detach().numpy().T
                sentiments = [np.mean(o) for o in out]
                d['date'] = self.content['date'][last_index + i]
                d['neu'], d['pos'], d['neg'] = sentiments[0], sentiments[1], sentiments[2]
                sent_block.append(d)
                count += 1
                if count == save_count:
                    df = pd.DataFrame(sent_block)
                    self.__save2csv(self.path, df)
                    sent_block = []
                    count = 0
            except Exception:
                print(f'ERROR Bert sentiment: index {i}')
                print(self.content['date'][i])
                df = pd.DataFrame(sent_block)
                self.__save2csv(self.path, df)
                sent_block = []
                count = 0

def run_eng_spx():
    content = loadContent('eng_spx_news_content.csv')
    sentiment = finBertSentiment('eng_spx_vectors.csv', content.content)
    sentiment.computeSentiment()


def main():
    run_eng_spx()

if __name__=='__main__':
    main()
