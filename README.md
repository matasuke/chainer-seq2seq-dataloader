# chainer-seq2seq-dataloader
Chainer data loader for text dataset using chainer.dataset.DatasetMixin.
This data loader can be used for Japanese, Chinese and English.

## Prerequesites
```
pip install -r requirements.txt
```

## Usage

### Clone the repository
```
git clone https://github.com/matasukef/chainer-seq2seq-dataloader
cd chainer-seq2seq-dataloader
```

### Download nltk tokenizer
```
python
import nltk
nltk.download('punkt')
```

### Clone Test Dataset
In this repository, [small_parallel_enja](https://github.com/odashi/small_parallel_enja) is used in [example.ipynb](https://github.com/matasukef/chainer-seq2seq-dataloader/example.ipynb)
```
git clone https://github.com/odashi/small_parallel_enja
```

### Build word2token dictionary and tokenize sentences.
```
python preprocess_tokens.py small_parallel_enja/train.ja data/ja_dataset.pickle --lang jp --tokenize --val_in_path small_parallel_enja/dev.ja --val_out_path data/ja_dataset_dev.pickle
python preprocess_tokens.py small_parallel_enja/train.en data/en_dataset.pickle --lang en --tokenize --val_in_path small_parallel_enja/dev.en --val_out_path data/en_dataset_dev.pickle
```

### Check DataLoader
For usage, please see [example.ipynb]()
In this repository, [small_parallel_enja](https://github.com/odashi/small_parallel_enja) is used in [example.ipynb](https://github.com/matasukef/chainer-seq2seq-dataloader/example.ipynb)
```
