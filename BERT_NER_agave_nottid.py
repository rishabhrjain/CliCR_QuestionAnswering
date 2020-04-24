import os
import json
import pandas as pd
import numpy as np
from pytorch_transformers import BertModel,BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from tqdm import tqdm, trange
from create_processed_data import *
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

DATASET_SIZE = 53914
MAX_LEN = 424
SEED = 520
bs = 16

print('Loading training data')
train_data_processor = DataProcessor('train',size=None,tokenizer=None)
tags_vals = train_data_processor.get_tags_vals()
tr_inputs,tr_masks,tr_tags,_,train_tokenizer = train_data_processor.get_processed_data()
print('Loaded training data')

print('Loading dev data')
valid_data_processor = DataProcessor('dev',size=None,tokenizer=train_tokenizer)
val_inputs,val_masks,val_tags,_,train_val_tokenizer = valid_data_processor.get_processed_data()
print('Loaded dev data')

tr_inputs = torch.tensor(tr_inputs)
tr_tags = torch.tensor(tr_tags)
tr_masks = torch.tensor(tr_masks)
#tr_ttids = torch.tensor(tr_ttids)
val_inputs = torch.tensor(val_inputs)
val_tags = torch.tensor(val_tags)
val_masks = torch.tensor(val_masks)
#val_ttids = torch.tensor(val_ttids)

torch.cuda.empty_cache() 

print('Creating data loaders for train and dev sets')
train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)
valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)


print('Loading BERT')
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tags_vals))
#output_dir = "./models/"
# Step 2: Re-load the saved model and vocabulary

# Example for a Bert model
#model = BertForTokenClassification.from_pretrained(output_dir,num_labels=len(tags_vals))

print('Pushing model to GPU')
model.cuda();


FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters()) 
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=1e-4)



from seqeval.metrics import f1_score
from seqeval.metrics import classification_report
from torch.nn import CrossEntropyLoss

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



print('Initiating training')
epochs = 14
max_grad_norm = 1.0
F_scores = np.zeros(epochs,float)
patience = 6
for e in range(epochs):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    predictions , true_labels = [], []
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        #loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        blogits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        closs = CrossEntropyLoss(ignore_index=-100)
        logits = blogits.view(-1,len(tags_vals))
        labels = b_labels.view(-1)
        loss = closs(logits, labels)
        # backward pass
        blogits = blogits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(blogits, axis=2)])
        true_labels.append(label_ids)
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i if l_ii!=-100]
    c_f1 = f1_score(valid_tags, pred_tags)
    rep = classification_report(valid_tags, pred_tags)
    print('Training classification report:',rep)
    print("Training F1-Score: {}".format(c_f1))
    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels= batch
        
        with torch.no_grad():
            #tmp_eval_loss = model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask, labels=b_labels)
            closs = CrossEntropyLoss(ignore_index=-100)
            logits = model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)
            blogits = logits.view(-1,len(tags_vals))
            labels = b_labels.view(-1)
            tmp_eval_loss = closs(blogits, labels)

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i if l_ii!=-100]
    c_f1 = f1_score(valid_tags, pred_tags)
    rep = classification_report(valid_tags, pred_tags)
    print(rep)
    print("F1-Score: {}".format(c_f1))
    #k epochs no improvement
    F_scores[e] = c_f1
    prev_Fs = np.arange(e-patience,e)
    prev_indices = prev_Fs[prev_Fs>=0]
    if all(F_scores[e] - F_scores[prev_indices] < 0 ) and e > patience:
        print('Done training')
        break
    else:
        print('Still training')
        #save checkpoint

        
        
        
        
        
print('Initiating validation')
model.eval()
predictions = []
true_labels = []
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
for batch in valid_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        #tmp_eval_loss = model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask, labels=b_labels)
        logits = model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)
        closs = CrossEntropyLoss(ignore_index=-100)
        blogits = logits.view(-1, len(tags_vals))
        labels = b_labels.view(-1)
        tmp_eval_loss = closs(blogits,labels)


    logits = logits.detach().cpu().numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    label_ids = b_labels.to('cpu').numpy()
    true_labels.append(label_ids)
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy

    nb_eval_examples += b_input_ids.size(0)
    nb_eval_steps += 1

pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]
valid_tags = [[tags_vals[l_ii] for l_ii in l_i if l_ii!=-100] for l in true_labels for l_i in l]
print("Validation loss: {}".format(eval_loss/nb_eval_steps))
print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
print("Validation F1-Score: {}".format(f1_score(valid_tags, pred_tags)))
rep = classification_report(valid_tags, pred_tags)
print(rep)




print('Saving model')
from pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME

output_dir = "./model_nottid/"

# Step 1: Save a model, configuration and vocabulary that you have fine-tuned

# If we have a distributed model, save only the encapsulated model
# (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
model_to_save = model.module if hasattr(model, 'module') else model

# If we save using the predefined names, we can load using `from_pretrained`
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
train_tokenizer.save_vocabulary(output_dir)
