#libraries
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import seaborn

import os

from torch.utils.data import Dataset, DataLoader


class IDAIC_Data_Generator(Dataset):
    def __init__(self,
                 data_dir,
               output_format='text',
               batch_size=2,
               shuffle=True,
               split="train",
               seed=None,
               prompt_maker = None,
              ):

        self.diac_woz_convo_path = os.path.join(data_dir, "transcript_DAIC_WOZ")
        self.diac_woz_ext_convo_path = os.path.join(data_dir, "transcript_ext_DAIC_WOZ")
        self.diac_woz_dev_path = os.path.join(data_dir, "dev_split_Depression_AVEC2017.csv")
        self.diac_woz_ext_dev_path = os.path.join(data_dir, "dev_split.csv")
        self.diac_woz_train_path = os.path.join(data_dir, "train_split_Depression_AVEC2017.csv")
        self.diac_woz_ext_train_path = os.path.join(data_dir, "train_split.csv")
        self.eatd_label_file = os.path.join(data_dir, "label_files_EATD")
        self.eatd_data = os.path.join(data_dir, "mentalHealth_bt.json")
        self.diac_woz_test_path = os.path.join(data_dir, "full_test_split.csv")
        self.diac_woz_ext_test_path = os.path.join(data_dir, "test_split.csv")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.split = split
        self.total_batches_seen = 0
        self.index_array = None
        self.total_batches_seen = 0
        self.format = output_format
        self.samples = self.read_samples(split, self.diac_woz_convo_path, self.diac_woz_ext_convo_path,
               self.diac_woz_dev_path, self.diac_woz_ext_dev_path,
               self.diac_woz_train_path, self.diac_woz_ext_train_path,
               self.diac_woz_test_path, self.diac_woz_ext_test_path,
               self.eatd_label_file, self.eatd_data)
        self.sample_ids = np.linspace(0, 1, len(self.samples))

        self.prompt_maker = prompt_maker

    def concatenate_dataframe(self, df):
        prev_speaker = None
        result_df = pd.DataFrame(columns=['speaker', 'value'])

        for index, row in df.iterrows():
            if prev_speaker == row['speaker']:
                result_df.at[result_df.index[-1], 'value'] += ' ' + row['value']
            else:
                result_df = pd.concat([result_df, pd.DataFrame({'speaker': [row['speaker']], 'value': [row['value']]})], ignore_index=True)
                prev_speaker = row['speaker']

        if result_df.iloc[0]['speaker'] == 'Ellie':
            result_df = result_df[2:]
        else:
            result_df = result_df[3:]

        return result_df

    def get_eatd_records(self):
      data = json.load(self.eatd_data)
      df = pd.DataFrame.from_dict(data = data, orient = 'columns')
      df = df.reset_index()
      df['combined'] = df['negative_418'] + df['neutral_418'] + df['positive_418']

      x = []
      y = []
      for f in df["index"].values:
          file = os.path.join(self.eatd_label_file, str(f) + ".txt")
          with open(file, 'r') as files:
              label = files.read()
              if float(label) >= 53:
                  y.append(1)
              else:
                  y.append(0)
          value = df[df['index'] == f].combined.values[0]
          x.append(value)

      x_train_eatd, X, y_train_eatd, Y = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)
      x_test_eatd, x_val_eatd, y_test_eatd, y_val_eatd = train_test_split(X, Y, test_size=0.50, random_state=42, stratify=Y)

      return x_train_eatd, y_train_eatd, x_test_eatd, y_test_eatd, x_val_eatd, y_val_eatd



    def get_records_and_labels(self, diac_woz_convo_path, diac_woz_ext_convo_path, daic_woz_split_path, daic_woz_ext_split_path, x_eatd, y_eatd):
        x = []
        y = []

        dc = pd.read_csv(daic_woz_split_path)
        dc_ex = pd.read_csv(daic_woz_ext_split_path)

        for f in dc["Participant_ID"].values:
            file = os.path.join(diac_woz_convo_path, str(f) + "_TRANSCRIPT.csv")
            df_ = pd.read_csv(file, delimiter = "\t")
            df_['value'] = df_['value'].astype(str)
            df = self.concatenate_dataframe(df_)
            df['combined'] = df['value']
            x.append(". ".join(map(str, df['combined'].values.tolist())))
            label = dc[dc['Participant_ID'] == int(f)].PHQ8_Binary.values[0]
            y.append(label)

        for f in dc_ex["Participant_ID"].values:
            file = os.path.join(diac_woz_ext_convo_path, str(f) + "_Transcript.csv")
            df_ex = pd.read_csv(file, delimiter = ",")
            df_ex['combined'] = df_ex['Text']
            x.append(". ".join(map(str, df_ex['combined'].values.tolist())))
            label = dc_ex[dc_ex['Participant_ID'] == int(f)].PHQ_Binary.values[0]
            y.append(label)

        for i in range(len(x_eatd)):
            x.append(x_eatd[i])
            y.append(y_eatd[i])

        return x, y


    def read_samples(self, split, diac_woz_convo_path, diac_woz_ext_convo_path,
               diac_woz_dev_path, diac_woz_ext_dev_path,
               diac_woz_train_path, diac_woz_ext_train_path,
               diac_woz_test_path, diac_woz_ext_test_path,
               eatd_label_file, eatd_data):
        samples = []
        x_train_eatd, y_train_eatd, x_test_eatd, y_test_eatd, x_val_eatd, y_val_eatd = self.get_eatd_records()

        if split =="train":
            Xs, ys = self.get_records_and_labels(diac_woz_convo_path, diac_woz_ext_convo_path, diac_woz_train_path, diac_woz_ext_train_path, x_train_eatd, y_train_eatd)
        elif split =="dev":
            Xs, ys = self.get_records_and_labels(diac_woz_convo_path, diac_woz_ext_convo_path, diac_woz_dev_path, diac_woz_ext_dev_path, x_val_eatd, y_val_eatd)
        elif split =="test":
            Xs, ys = self.get_records_and_labels(diac_woz_convo_path, diac_woz_ext_convo_path, diac_woz_test_path, diac_woz_ext_test_path, x_test_eatd, y_test_eatd)

        for x, y in zip(Xs, ys):
            samples.append({
                'conversation': x,
                'label': int(y),
            })

        return samples

    def set_index_array(self):
        self.index_array = np.arange(0, len(self), 1)
        if self.shuffle:
            np.random.shuffle(self.index_array)

    def __len__(self):
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError

        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self.set_index_array()

        index_array = self.index_array[self.batch_size * int(idx): self.batch_size * int(idx + 1)]

        return self._get_batch(index_array)

    def _get_batch(self, index_array):
        batch_x, batch_y = self._get_sample_batch(index_array)
        return batch_x, batch_y

    def _get_sample_batch(self, index_array):
        batch_x, batch_y = [], []
        for x, y in self._get_sample_pair(index_array):
            batch_x.append(x)
            batch_y.append(y)
        return batch_x, batch_y

    def _get_sample_pair(self, index):
        for i in index:
            sample = self.samples[i]
            prompt = self._get_record(sample)
            label = 'depressed' if sample['label'] == 1 else 'not depressed'
            yield prompt, label

    def _get_record(self, sample):
        instruct = {
            # 'list': 'Given the context, provide the answer in the form of a list.',
            'factoid': 'Given the context, answer the factoid based question.',
            # 'summary': 'Provide the answer to the question in the form of a context based summary.',
            # 'yesno': 'Based on the context, answer in either yes or no for the following question, and also provide a brief one line explanation for the answer.'
        }

        prompt = f"""
        {sample['conversation']}
        """

        return prompt
