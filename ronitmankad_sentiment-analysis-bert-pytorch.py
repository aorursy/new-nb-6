import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from pytorch_pretrained_bert import BertTokenizer, BertConfig

from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification

from tqdm import tqdm, trange

import pandas as pd

import io

import numpy as np

import matplotlib.pyplot as plt




import torch

from pytorch_pretrained_bert.tokenization import BertTokenizer

import os

print(os.listdir('../input/'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_gpu = torch.cuda.device_count()

torch.cuda.get_device_name(0)
train_df = pd.read_csv('../input/nlp-hack/train.csv')

test_df = pd.read_csv('../input/nlp-hack/test.csv')
# train_df['target'] = train_df['target'].map({0:0, 4:1})

# sentences = train_df.text.values

# sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]

# labels = train_df.target.values
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

# print ("Tokenize the first sentence:")

# print (tokenized_texts[0])
# MAX_LEN = 128

# input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],

#                           maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")



# input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]



# input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# attention_masks = []



# # Create a mask of 1s for each token followed by 0s for padding

# for seq in input_ids:

#   seq_mask = [float(i>0) for i in seq]

#   attention_masks.append(seq_mask)
# train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 

#                                                             random_state=2018, test_size=0.1)

# train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,

#                                              random_state=2018, test_size=0.1)
# train_inputs = torch.tensor(train_inputs)

# validation_inputs = torch.tensor(validation_inputs)

# train_labels = torch.tensor(train_labels)

# validation_labels = torch.tensor(validation_labels)

# train_masks = torch.tensor(train_masks)

# validation_masks = torch.tensor(validation_masks)
# batch_size = 32



# # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 

# # with an iterator the entire dataset does not need to be loaded into memory



# train_data = TensorDataset(train_inputs, train_masks, train_labels)

# train_sampler = RandomSampler(train_data)

# train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)



# validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)

# validation_sampler = SequentialSampler(validation_data)

# validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

def save_checkpoint():

    checkpoint = {

        'model':model, 

    }

    torch.save(checkpoint, './bert_model.pt')

    

def load_checkpoint(filepath, inference = False):

    checkpoint = torch.load('../input/bertmodelsaved/bert_model_final.pt')

    model = checkpoint['model']

    if inference:

        for parameter in model.parameter():

            parameter.require_grad = False

        model.eval()

    model.to(device)

    return model

        
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

model = load_checkpoint('../input/bertmodelsaved/bert_model_final.pt')

model.cuda()
for param in model.bert.parameters():

  param.requires_grad = False



for name, param in model.named_parameters():                

    if param.requires_grad:

        print(name)
# param_optimizer = list(model.named_parameters())

# no_decay = ['bias', 'gamma', 'beta']

# optimizer_grouped_parameters = [

#     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],

#      'weight_decay_rate': 0.01},

#     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],

#      'weight_decay_rate': 0.0}

# ]



# optimizer = BertAdam(optimizer_grouped_parameters,

#                      lr=2e-5,

#                      warmup=.1)



# def flat_accuracy(preds, labels):

#     pred_flat = np.argmax(preds, axis=1).flatten()

#     labels_flat = labels.flatten()

#     return np.sum(pred_flat == labels_flat) / len(labels_flat)
# train_loss_set = []



# # Number of training epochs (authors recommend between 2 and 4)

# epochs = 4



# # trange is a tqdm wrapper around the normal python range

# for _ in trange(epochs, desc="Epoch"):

  

  

#   # Training

  

#   # Set our model to training mode (as opposed to evaluation mode)

#   model.train()

  

#   # Tracking variables

#   tr_loss = 0

#   nb_tr_examples, nb_tr_steps = 0, 0

  

#   # Train the data for one epoch

#   for step, batch in enumerate(train_dataloader):

#     # Add batch to GPU

#     batch = tuple(t.to(device) for t in batch)

#     # Unpack the inputs from our dataloader

#     b_input_ids, b_input_mask, b_labels = batch

#     # Clear out the gradients (by default they accumulate)

#     optimizer.zero_grad()

#     # Forward pass

#     loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

#     train_loss_set.append(loss.item())    

#     # Backward pass

#     loss.backward()

#     # Update parameters and take a step using the computed gradient

#     optimizer.step()

    

    

#     # Update tracking variables

#     tr_loss += loss.item()

#     nb_tr_examples += b_input_ids.size(0)

#     nb_tr_steps += 1



#   print("Train loss: {}".format(tr_loss/nb_tr_steps))

    

    

#   # Validation



#   # Put model in evaluation mode to evaluate loss on the validation set

#   model.eval()



#   # Tracking variables 

#   eval_loss, eval_accuracy = 0, 0

#   nb_eval_steps, nb_eval_examples = 0, 0



#   # Evaluate data for one epoch

#   for batch in validation_dataloader:

#     # Add batch to GPU

#     batch = tuple(t.to(device) for t in batch)

#     # Unpack the inputs from our dataloader

#     b_input_ids, b_input_mask, b_labels = batch

#     # Telling the model not to compute or store gradients, saving memory and speeding up validation

#     with torch.no_grad():

#       # Forward pass, calculate logit predictions

#       logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    

#     # Move logits and labels to CPU

#     logits = logits.detach().cpu().numpy()

#     label_ids = b_labels.to('cpu').numpy()



#     tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    

#     eval_accuracy += tmp_eval_accuracy

#     nb_eval_steps += 1



#   print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

#   save_checkpoint()
sentences = test_df.text.values

sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]

labels = np.random.rand(len(sentences))
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

MAX_LEN = 128

# Pad our input tokens

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],

                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention masks

attention_masks = []



# Create a mask of 1s for each token followed by 0s for padding

for seq in input_ids:

  seq_mask = [float(i>0) for i in seq]

  attention_masks.append(seq_mask) 



prediction_inputs = torch.tensor(input_ids)

prediction_masks = torch.tensor(attention_masks)

prediction_labels = torch.tensor(labels)

  

batch_size = 32  





prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)

prediction_sampler = SequentialSampler(prediction_data)

prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
model.eval()



# Tracking variables 

predictions , true_labels = [], []



# Predict 

for batch in prediction_dataloader:

  # Add batch to GPU

  batch = tuple(t.to(device) for t in batch)

  # Unpack the inputs from our dataloader

  b_input_ids, b_input_mask, b_labels = batch

  # Telling the model not to compute or store gradients, saving memory and speeding up prediction

  with torch.no_grad():

    # Forward pass, calculate logit predictions

    logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)



  # Move logits and labels to CPU

  logits = logits.detach().cpu().numpy()

  label_ids = b_labels.to('cpu').numpy()

  

  # Store predictions and true labels

  predictions.append(logits)

  true_labels.append(label_ids)
my_submission = pd.DataFrame()

my_submission['Id'] = test_df['Id']
final_preds = []

for p in predictions:

    for i in p:

        final_preds.append(np.argmax(i))
my_submission['target'] = final_preds
my_submission['target'] = my_submission['target'].map({0:0, 1:4})
my_submission.head()
my_submission.to_csv('my_sub.csv', index=False)