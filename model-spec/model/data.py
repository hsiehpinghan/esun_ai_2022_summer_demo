import re

from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

class EsunDataset(Dataset):
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.data[index]['original_text'],
                self.data[index]['correct_text'])

class EsunDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        (original_texts, correct_texts) = zip(*data)
        encoded_texts = [self.tokenizer.tokenize(original_text) for original_text in original_texts]
        max_len = max([len(encoded_text) for encoded_text in encoded_texts]) + 2
        return (original_texts, correct_texts)

class EsunDataModule(LightningDataModule):
    def __init__(self, data_path, batch_size, tokenizer, split):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.split = split

    def prepare_data(self):
        (data_train, data_val) = self._get_data(data_path=self.data_path)
        self.data_train = data_train
        self.data_val = data_val
        print(f'data_train length({len(self.data_train)}) / data_val length({len(self.data_val)})')

    def setup(self, stage):
        if stage == 'fit':
            self.ds_train = EsunDataset(name='train',
                                        data=self.data_train)
            self.ds_val = EsunDataset(name='val',
                                      data=self.data_val)
        elif stage == 'validate':
            self.ds_val = EsunDataset(name='val',
                                      data=self.data_val)
        elif stage == 'test':
            self.ds_test = EsunDataset(name='test',
                                       data=self.data_val)
        elif stage == 'predict':
            self.ds_predict = EsunDataset(name='predict',
                                          data=self.data_val)
        else:
            raise Exception(f'stage({stage}) not implement !!!')

    def train_dataloader(self):
        return DataLoader(dataset=self.ds_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=EsunDataCollator(tokenizer=self.tokenizer),
                          num_workers=16,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.ds_val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=EsunDataCollator(tokenizer=self.tokenizer),
                          num_workers=16,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.ds_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=EsunDataCollator(tokenizer=self.tokenizer),
                          num_workers=16,
                          pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(dataset=self.ds_predict,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=EsunDataCollator(tokenizer=self.tokenizer),
                          num_workers=16,
                          pin_memory=True)

    def _get_data(self, data_path):
        data = []
        with open(file=data_path,
                  mode='r') as f:
            for line in f:
                s = self._remove_not_chinese_char(text=line)
                if line.startswith('\t') == False:
                    d = {'correct_text': s,
                         'original_texts': set([s])}
                    data.append(d)
                else:
                    if len(s) != len(data[-1]['correct_text']):
                        continue
                    data[-1]['original_texts'].add(s)
        data_train = []
        data_val = []
        for (i, d) in enumerate(data):
            if (i % 5) == self.split:
                for original_text in d['original_texts']:
                    data_val.append({'correct_text': d['correct_text'],
                                     'original_text': original_text})
            else:
                for original_text in d['original_texts']:
                    data_train.append({'correct_text': d['correct_text'],
                                       'original_text': original_text})
        return (data_train, data_val)

    def _remove_not_chinese_char(self, text):
        text = re.sub(pattern='([^\u4e00-\u9fa5])+',
                    repl='',
                    string=text)
        return text

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='shibing624/macbert4csc-base-chinese')
    esun_data_module = EsunDataModule(data_path='/home/hsiehpinghan/git/esun_ai_2022_summer_demo/model-spec/data/esun_ai_2022_summer_20220415.txt',
                                      batch_size=2,
                                      tokenizer=tokenizer,
                                      split=0)
    esun_data_module.prepare_data()
    esun_data_module.setup(stage='validate')
    val_dataloader = esun_data_module.val_dataloader()
    for (i, batch) in enumerate(val_dataloader):
        break
    print(batch)