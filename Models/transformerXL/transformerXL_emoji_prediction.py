# Importing all libraries

import matplotlib.pyplot as plt
import torch
from transformers import TransfoXLTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import TransfoXLForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random
import numpy as np
import pandas as pd
import os
import logging

_logger = logging.getLogger(__name__)

epochs = 2

# Checking for GPU device

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

_logger.info(f'We are using a {device} device')

# Loading the dataset

data_path = '\\'.join(os.path.dirname(__file__).split('\\')[:-2])
data_path = os.path.join(data_path, 'Data')

mappings_path = os.path.join(data_path, 'mapping.txt')

X_train_path = os.path.join(data_path, 'train_text.txt')
y_train_path = os.path.join(data_path, 'train_labels.txt')

X_test_path = os.path.join(data_path, 'test_text.txt')
y_test_path = os.path.join(data_path, 'test_labels.txt')

X_val_path = os.path.join(data_path, 'val_text.txt')
y_val_path = os.path.join(data_path, 'val_labels.txt')

def get_data_from_path(path):
    '''
    Function that reads the data from the given path to file and returns the list
    @path : String containing the path of the file
    '''
    with open(path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    for i in range(len(content)):
        content[i] = content[i][:-1]
    return content

X_train = get_data_from_path(X_train_path)
y_train = list(map(int, get_data_from_path(y_train_path)))

X_test = get_data_from_path(X_test_path)
y_test = list(map(int, get_data_from_path(y_test_path)))

X_val = get_data_from_path(X_val_path)
y_val = list(map(int, get_data_from_path(y_val_path)))

mappings = get_data_from_path(mappings_path)
num_classes = len(mappings)

# Creating a tokenizer

tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103', do_lower_case=True)

# Calculating the maximum length of the encoded tokens

max_length = 0

for i in range(len(X_train)):
    input_ids = tokenizer(X_train[i], add_special_tokens=True)['input_ids']
    max_length = max(max_length, len(input_ids))

# Tokenizing the data

def get_dataloader(X, y, is_train=False, batch_size=32):
    '''
    Function that takes in the data and returns the dataloader for the dataset
    @X: Features of the data
    @y: Labels of the data
    @is_train: True if the data is used for training
    @batch_size: Size of the batch
    '''
    input_ids = []
    attention_masks = []
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    for i in range(len(X)):
        encoded_dict = tokenizer(
            X[i],
            add_special_tokens = True,
            max_length = max_length * 2,
            padding = 'max_length',
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(y)
    labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).to(torch.float)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    if is_train:
        return DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
    else:
        return DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

train_dataloader = get_dataloader(X_train, y_train, is_train=True, batch_size=1)
val_dataloader = get_dataloader(X_val, y_val, is_train=False, batch_size=1)
test_dataloader = get_dataloader(X_test, y_test, is_train=False, batch_size=1)

# Loading the transfoXL Base model

model = TransfoXLForSequenceClassification.from_pretrained(
    'transfo-xl-wt103',
    problem_type="multi_label_classification",
    num_labels = 20,
    output_attentions = False,
    output_hidden_states = False
)

model.to(device)

# Creating the optimizer instance

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

total_steps = len(train_dataloader) * epochs
schedular = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

def flat_accuracy(preds, labels):
    '''
    Takes the prediction and true labels and calculates the accuracy
    '''
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.argmax(labels, axis=1).flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Training the model

seed_val = 21
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed(seed_val)

training_stats = []
best_val_loss = float('inf')

total_t0 = time.time()

for epoch_i in range(0, epochs):
    _logger.info('')
    _logger.info(f'======= Epoch {epoch_i+1} / {epochs} =======')
    _logger.info('Training...')

    t0 = time.time()
    total_training_loss = 0
    total_train_accuracy = 0

    model.train()
    for step, batch in enumerate(train_dataloader):
        if step%10 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            _logger.info(f'   Batch {step} of {len(train_dataloader)}. Elapsed: {elapsed}')
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()
        result = model(
            b_input_ids,
            labels = b_labels,
            return_dict = True
            )

        loss = result.loss
        logits = result.logits
        total_training_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedular.step()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_train_accuracy += flat_accuracy(logits, label_ids)
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)

    avg_train_loss = total_training_loss / len(train_dataloader)
    training_time = format_time(time.time()-t0)
    _logger.info('')
    _logger.info(f'   Average training loss : {avg_train_loss}')
    _logger.info(f'   Average training accuracy : {avg_train_accuracy}')
    _logger.info(f'   Training epoch took : {training_time}')

    _logger.info('')
    _logger.info('Running Validation...')

    t0 = time.time()
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            result = model(b_input_ids,
                           labels = b_labels,
                           return_dict = True)
        loss = result.loss
        logits = result.logits
        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    avg_val_loss = total_eval_loss / len(val_dataloader)
    validation_time = format_time(time.time() - t0)
    _logger.info(f'   Validation Loss: {avg_val_loss}')
    _logger.info(f'   Validation Accuracy: {avg_val_accuracy}')
    _logger.info(f'   Validation took: {validation_time}')

    training_stats.append(
        {
            'epoch' : epoch_i + 1,
            'training_loss' : avg_train_loss,
            'training_accuracy' : avg_train_accuracy,
            'val_loss' : avg_val_loss,
            'val_accuracy' : avg_val_accuracy,
            'training_time' : training_time,
            'validation_time' : validation_time
        }
    )
    if avg_val_loss < best_val_loss:
        torch.save(model.state_dict(), f'transfoXL_{avg_val_loss}')
        best_val_loss = avg_val_loss

_logger.info('')
_logger.info('TRAINING COMPLETE!')

_logger.info(f'Total training time took {format_time(time.time()-total_t0)} (h:mm:ss)')

# Printing some stats

df_stats = pd.DataFrame(data=training_stats).set_index('epoch')
df_stats.to_csv('Models/transformerXL/transfoXL.csv', encoding='utf-8')

# Plotting the training and validation loss of the model over training epochs

plt.rcParams['figure.figsize'] = (12,6)

plt.plot(df_stats['training_loss'], 'b-o', label='Training')
plt.plot(df_stats['val_loss'], 'g-o', label='Validation')

plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.xticks(list(range(epochs)))

plt.savefig('Models/transformerXL/transfoXL.png')
plt.close()

# Making predictions on the test dataset

print(f'Predicting labels for {len(test_dataloader)} test sentences...')

model.eval()
predictions, true_labels = [], []

for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        result = model(
                b_input_ids,
                return_dict = True
            )
        logits = result.logits
        logits = logits.detach().cpu().numpy()
        pred_labels = np.argmax(logits, axis=1)
        label_ids = np.argmax(b_labels.to('cpu').numpy(), axis=1)

        predictions.extend(pred_labels.tolist())
        true_labels.extend(label_ids.tolist())

print('Completed prediction')

accuracy = 0
for i in range(len(true_labels)):
    if true_labels[i] == predictions[i]:
        accuracy += 1
accuracy /= len(true_labels)
with open('Models/transformerXL/transfoXL_test.txt', 'w') as f:
    f.write(str(accuracy))