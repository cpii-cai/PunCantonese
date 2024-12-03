import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import torch.multiprocessing
from tqdm import tqdm
from focalloss import focal_loss  


from argparser import parse_arguments
from dataset import Dataset_cant,Dataset_cant_jyupin, Dataset_cant_multitask_sequence
from model import DeepPunctuation,DeepPunctuationCRF, DeepPunctuationCRF_jyupin, DeepPunctuation_jyupin
from config import *
import augmentation
from transformers import XLMModel, XLMTokenizer,XLMRobertaModel,XLMRobertaTokenizer

torch.multiprocessing.set_sharing_strategy('file_system')   # https://github.com/pytorch/pytorch/issues/11201

args = parse_arguments()

# for reproducibility
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# tokenizer

#tokenizer = MODELS[args.pretrained_model][1].from_pretrained(args.pretrained_model)
#tokenizer = XLMRobertaTokenizer.from_pretrained("../../../xlm-roberta-large")  
tokenizer = MODELS[args.pretrained_model][1].from_pretrained("../../../"+args.pretrained_model)

augmentation.tokenizer = tokenizer

augmentation.sub_style = args.sub_style
augmentation.alpha_sub = args.alpha_sub
augmentation.alpha_del = args.alpha_del
token_style = MODELS[args.pretrained_model][3]
ar = args.augment_rate
sequence_len = args.sequence_length
aug_type = args.augment_type

punctuation_dict = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3,'EXCLAMATION': 4}
if args.multitask:
    punctuation_dict = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3,'EXCLAMATION': 4,'INFORMAL':5,'FORMAL':6}
    

# Datasets
if args.jyutping:
    if args.multitask:
        train_set = Dataset_cant_jyupin(os.path.join(args.data_path, 'cant_jyupin_multi/train_wiki.txt'), tokenizer=tokenizer, sequence_len=sequence_len,
                            token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type,multitask=True)
        val_set = Dataset_cant_jyupin(os.path.join(args.data_path, 'cant_jyupin_multi/val.txt'), tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=False,multitask=True)
        test_set_ref = Dataset_cant_jyupin(os.path.join(args.data_path, 'cant_jyupin_multi/test.txt'), tokenizer=tokenizer, sequence_len=sequence_len,
                            token_style=token_style, is_train=False,multitask=True)
        test_set = [val_set, test_set_ref]
    else:
        train_set = Dataset_cant_jyupin(os.path.join(args.data_path, 'cant_jyupin/train_wiki.txt'), tokenizer=tokenizer, sequence_len=sequence_len,
                            token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type)
        val_set = Dataset_cant_jyupin(os.path.join(args.data_path, 'cant_jyupin/val.txt'), tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=False)
        test_set_ref = Dataset_cant_jyupin(os.path.join(args.data_path, 'cant_jyupin/test.txt'), tokenizer=tokenizer, sequence_len=sequence_len,
                            token_style=token_style, is_train=False)
        test_set = [val_set, test_set_ref]

elif args.multitask:
    train_set = Dataset_cant_multitask_sequence(os.path.join(args.data_path, 'cant_multitask/train.txt'), tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type)
    val_set = Dataset_cant_multitask_sequence(os.path.join(args.data_path, 'cant_multitask/val.txt'), tokenizer=tokenizer, sequence_len=sequence_len,
                      token_style=token_style, is_train=False)
    test_set_ref = Dataset_cant_multitask_sequence(os.path.join(args.data_path, 'cant_multitask/test.txt'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False)
    test_set = [val_set, test_set_ref]
else:
    train_set = Dataset_cant(os.path.join(args.data_path, 'cant_informal_wiki/train.txt'), tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type)
    val_set = Dataset_cant(os.path.join(args.data_path, 'cant_informal_wiki/val.txt'), tokenizer=tokenizer, sequence_len=sequence_len,
                      token_style=token_style, is_train=False)
    test_set_ref = Dataset_cant(os.path.join(args.data_path, 'cant_informal_wiki/test.txt'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False)
    test_set = [val_set, test_set_ref]

# Data Loaders
data_loader_params = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': 1
}
train_loader = torch.utils.data.DataLoader(train_set, **data_loader_params)
val_loader = torch.utils.data.DataLoader(val_set, **data_loader_params)
test_loaders = [torch.utils.data.DataLoader(x, **data_loader_params) for x in test_set]

# logs
os.makedirs(args.save_path, exist_ok=True)
model_save_path = os.path.join(args.save_path, args.name+'weights.pt')
log_path = os.path.join(args.save_path, args.name + '_logs.txt')


# select gpu
#os.environ["CUDA_VISIBLE_DEVICES"]="1"


# Model
device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
if args.use_crf:
    if args.jyutping:
        deep_punctuation = DeepPunctuationCRF_jyupin(args.pretrained_model, freeze_bert=args.freeze_bert, lstm_dim=args.lstm_dim,punctuation_dict=punctuation_dict)
    else:
        deep_punctuation = DeepPunctuationCRF(args.pretrained_model, freeze_bert=args.freeze_bert, lstm_dim=args.lstm_dim,punctuation_dict=punctuation_dict)

else:
    if args.jyutping:
        deep_punctuation = DeepPunctuation_jyupin(args.pretrained_model, freeze_bert=args.freeze_bert, lstm_dim=args.lstm_dim,punctuation_dict=punctuation_dict)
    else:
        deep_punctuation = DeepPunctuation(args.pretrained_model, freeze_bert=args.freeze_bert, lstm_dim=args.lstm_dim,punctuation_dict=punctuation_dict)

deep_punctuation.to(device)


#loss
criterion = nn.CrossEntropyLoss()
#criterion1 = focal_loss()

if args.loss == 'CE':
    criterion = nn.CrossEntropyLoss()
elif args.loss == 'focal':
    if args.multitask:
        criterion = focal_loss(alpha=[0.75,0.75,0.75,0.25,0.25,0.75,0.75],num_classes = 7)
    else:
        criterion = focal_loss(alpha=[0.75,0.75,0.75,0.25,0.25],num_classes = 5)




optimizer = torch.optim.Adam(deep_punctuation.parameters(), lr=args.lr, weight_decay=args.decay)


def validate(data_loader):
    """
    :return: validation accuracy, validation loss
    """
    num_iteration = 0
    deep_punctuation.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        if args.jyutping:
            for x, y, att, y_mask, jyupin in tqdm(data_loader, desc='eval'):
                x, y, att, y_mask, jyupin = x.to(device), y.to(device), att.to(device), y_mask.to(device), jyupin.to(device)
                y_mask = y_mask.view(-1)
                if args.use_crf:
                    y_predict = deep_punctuation(x, att, y, jyupin)
                    loss = deep_punctuation.log_likelihood(x, att, y, jyupin)
                    y_predict = y_predict.view(-1)
                    y = y.view(-1)
                else:
                    y_predict = deep_punctuation(x, att, jyupin)
                    y = y.view(-1)
                    y_predict = y_predict.view(-1, y_predict.shape[2])
                    loss = criterion(y_predict, y)
                    #loss = criterion(y_predict, y)+criterion1(y_predict, y)
                    y_predict = torch.argmax(y_predict, dim=1).view(-1)
                val_loss += loss.item()
                num_iteration += 1
                y_mask = y_mask.view(-1)
                correct += torch.sum(y_mask * (y_predict == y).long()).item()
                total += torch.sum(y_mask).item()
        else:
            for x, y, att, y_mask in tqdm(data_loader, desc='eval'):
                x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
                y_mask = y_mask.view(-1)
                if args.use_crf:
                    y_predict = deep_punctuation(x, att, y)
                    loss = deep_punctuation.log_likelihood(x, att, y)
                    y_predict = y_predict.view(-1)
                    y = y.view(-1)
                else:
                    y_predict = deep_punctuation(x, att)
                    y = y.view(-1)
                    y_predict = y_predict.view(-1, y_predict.shape[2])
                    loss = criterion(y_predict, y)
                    #loss = criterion(y_predict, y)+criterion1(y_predict, y)
                    y_predict = torch.argmax(y_predict, dim=1).view(-1)
                val_loss += loss.item()
                num_iteration += 1
                y_mask = y_mask.view(-1)
                #correct += torch.sum(y_mask * (y_predict == y).long()).item()
                #yx
                correct += torch.sum(y_mask[1:] * (y_predict[1:] == y[1:]).long()).item()
                total += torch.sum(y_mask).item()
    return correct/total, val_loss/num_iteration


def test(data_loader):
    """
    :return: precision[numpy array], recall[numpy array], f1 score [numpy array], accuracy, confusion matrix
    """
    num_iteration = 0
    deep_punctuation.eval()
    # +1 for overall result
    tp = np.zeros(1+len(punctuation_dict), dtype=np.int)
    fp = np.zeros(1+len(punctuation_dict), dtype=np.int)
    fn = np.zeros(1+len(punctuation_dict), dtype=np.int)
    cm = np.zeros((len(punctuation_dict), len(punctuation_dict)), dtype=np.int)
    correct = 0
    total = 0
    with torch.no_grad():
        if args.jyutping:
            for x, y, att, y_mask, jyupin in tqdm(data_loader, desc='test'):
                x, y, att, y_mask, jyupin = x.to(device), y.to(device), att.to(device), y_mask.to(device), jyupin.to(device)
                y_mask = y_mask.view(-1)
                if args.use_crf:
                    y_predict = deep_punctuation(x, att, y, jyupin)
                    y_predict = y_predict.view(-1)
                    y = y.view(-1)
                else:
                    y_predict = deep_punctuation(x, att, jyupin)
                    y = y.view(-1)
                    y_predict = y_predict.view(-1, y_predict.shape[2])
                    y_predict = torch.argmax(y_predict, dim=1).view(-1)
                num_iteration += 1
                y_mask = y_mask.view(-1)
                correct += torch.sum(y_mask * (y_predict == y).long()).item()
                total += torch.sum(y_mask).item()
                for i in range(y.shape[0]):
                    if y_mask[i] == 0:
                        # we can ignore this because we know there won't be any punctuation in this position
                        # since we created this position due to padding or sub-word tokenization
                        continue
                    cor = y[i]
                    prd = y_predict[i]
                    if cor == prd:
                        tp[cor] += 1
                    else:
                        fn[cor] += 1
                        fp[prd] += 1
                    cm[cor][prd] += 1
        else:
            for x, y, att, y_mask in tqdm(data_loader, desc='test'):
                x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
                y_mask = y_mask.view(-1)
                if args.use_crf:
                    y_predict = deep_punctuation(x, att, y)
                    y_predict = y_predict.view(-1)
                    y = y.view(-1)
                else:
                    y_predict = deep_punctuation(x, att)
                    y = y.view(-1)
                    y_predict = y_predict.view(-1, y_predict.shape[2])
                    y_predict = torch.argmax(y_predict, dim=1).view(-1)
                num_iteration += 1
                y_mask = y_mask.view(-1)
                correct += torch.sum(y_mask * (y_predict == y).long()).item()
                total += torch.sum(y_mask).item()
                for i in range(y.shape[0]):
                    if y_mask[i] == 0:
                        # we can ignore this because we know there won't be any punctuation in this position
                        # since we created this position due to padding or sub-word tokenization
                        continue
                    cor = y[i]
                    prd = y_predict[i]
                    if cor == prd:
                        tp[cor] += 1
                    else:
                        fn[cor] += 1
                        fp[prd] += 1
                    cm[cor][prd] += 1

    # ignore first index which is for no punctuation
    tp[-1] = np.sum(tp[1:5])
    fp[-1] = np.sum(fp[1:5])
    fn[-1] = np.sum(fn[1:5])
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, correct/total, cm


def train():
    with open(log_path, 'a') as f:
        f.write(str(args)+'\n')
    best_val_acc = 0
    for epoch in range(args.epoch):
        train_loss = 0.0
        train_iteration = 0
        correct = 0
        total = 0
        deep_punctuation.train()
        if args.jyutping:
            for x, y, att, y_mask, jyupin in tqdm(train_loader, desc='train'):
                x, y, att, y_mask, jyupin= x.to(device), y.to(device), att.to(device), y_mask.to(device), jyupin.to(device)
                y_mask = y_mask.view(-1)
                if args.use_crf:
                    loss = deep_punctuation.log_likelihood(x, att, y, jyupin)
                    y_predict = deep_punctuation(x, att, y, jyupin)
                    y_predict = y_predict.view(-1)
                    y = y.view(-1)
                else:
                    y_predict = deep_punctuation(x, att, jyupin)
                    y_predict = y_predict.view(-1, y_predict.shape[2])
                    y = y.view(-1)
                    loss = criterion(y_predict, y)
                    #loss = criterion(y_predict, y)+criterion1(y_predict, y)
                    y_predict = torch.argmax(y_predict, dim=1).view(-1)

                    correct += torch.sum(y_mask * (y_predict == y).long()).item()

                optimizer.zero_grad()
                train_loss += loss.item()
                train_iteration += 1
                loss.backward()

                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(deep_punctuation.parameters(), args.gradient_clip)
                optimizer.step()

                y_mask = y_mask.view(-1)

                total += torch.sum(y_mask).item()
        else:
            for x, y, att, y_mask in tqdm(train_loader, desc='train'):
                x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
                y_mask = y_mask.view(-1)
                if args.use_crf:
                    loss = deep_punctuation.log_likelihood(x, att, y)
                    # y_predict = deep_punctuation(x, att, y)
                    # y_predict = y_predict.view(-1)
                    y = y.view(-1)
                else:
                    y_predict = deep_punctuation(x, att)
                    y_predict = y_predict.view(-1, y_predict.shape[2])
                    y = y.view(-1)
                    loss = criterion(y_predict, y)
                    #loss = criterion(y_predict, y)+criterion1(y_predict, y)
                    y_predict = torch.argmax(y_predict, dim=1).view(-1)

                    correct += torch.sum(y_mask * (y_predict == y).long()).item()

                optimizer.zero_grad()
                train_loss += loss.item()
                train_iteration += 1
                loss.backward()

                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(deep_punctuation.parameters(), args.gradient_clip)
                optimizer.step()

                y_mask = y_mask.view(-1)

                total += torch.sum(y_mask).item()
        train_loss /= train_iteration
        log = 'epoch: {}, Train loss: {}, Train accuracy: {}'.format(epoch, train_loss, correct / total)
        with open(log_path, 'a') as f:
            f.write(log + '\n')
        print(log)

        val_acc, val_loss = validate(val_loader)
        log = 'epoch: {}, Val loss: {}, Val accuracy: {}'.format(epoch, val_loss, val_acc)
        with open(log_path, 'a') as f:
            f.write(log + '\n')
        print(log)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(deep_punctuation.state_dict(), model_save_path)

    print('Best validation Acc:', best_val_acc)
    deep_punctuation.load_state_dict(torch.load(model_save_path))
    for loader in test_loaders:
        precision, recall, f1, accuracy, cm = test(loader)
        log = 'Precision: ' + str(precision) + '\n' + 'Recall: ' + str(recall) + '\n' + 'F1 score: ' + str(f1) + \
              '\n' + 'Accuracy:' + str(accuracy) + '\n' + 'Confusion Matrix' + str(cm) + '\n'
        print(log)
        with open(log_path, 'a') as f:
            f.write(log)
        log_text = ''
        for i in range(1, 5):
            log_text += str(precision[i] * 100) + ' ' + str(recall[i] * 100) + ' ' + str(f1[i] * 100) + ' '
        with open(log_path, 'a') as f:
            f.write(log_text[:-1] + '\n\n')




if __name__ == '__main__':
    train()
