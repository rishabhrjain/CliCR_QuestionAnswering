# CliCR-NLP
Medical Reading Comprehension QA 

Goal: 
Build a model that ouputs entities given a passage and a query. In this case, the passage a paragraph about a disease, treatment, etc and a query is a question related to the passage. 
The model needs to output an answer (entities) based on the passage and the query. The entities can be the name of the disease, treatment, etc. 

Steps:

#### 1. Setup

    install Python >= 3.6

    $ pip install python3

    $ pip install -r requirements.txt

    In the case that you want to download the dataset, use the following link:
    https://drive.google.com/file/d/1K6QScXlR01RbWqycT9ixafkQQIxH8MhW/view

    Save the dataset in a directory. 

#### 2. Run the BERT_NER notebook. The notebook: 
        i. Processes the data in a format suitable for training the model. 
        ii. Train the model and save it
        iii. Evaluation


