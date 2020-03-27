# CliCR-NLP
Medical Reading Comprehension QA 

Launch the BERT_NER_agave.ipynb after you gitclone the repository.

1. jupyter notebook / google colab
2. get into the direcotry 
3. open BERT_NER_agave.ipynb and change the path of the directory to DATA_PATH
( i.e. data_reader = MyDataReader(data_path= DATA_PATHH , bs=10000) in BERT_NER_agave.ipynb )
4. launch BERT_NER_agave.ipynb 

Requirements:

$ git clone https://github.com/EswarSaiKrish/CliCR-NLP.git
$ cd DATA_PATH
## if there is no data in your directory, put the dataset in there. 
## install Python >= 3.6
$ pip install python3
## install Tensorflow >= 2.0.0
$ pip install "tensorflow>=2.0.0"
## install pytorch-transformers / Transformer
$ pip install pytorch-transformers / transformers 
## install pytorch_pretrained_bert
$ pip install pytorch_pretrained_bert
## install torch
$ pip install torch

In the case that you want to download the dataset, use the following link:
https://drive.google.com/file/d/1K6QScXlR01RbWqycT9ixafkQQIxH8MhW/view


