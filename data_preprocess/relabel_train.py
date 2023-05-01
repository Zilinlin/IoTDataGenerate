#written by Zilin Shen and Daniel de Mello
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
from label_packet import generate_label_data, label_packets
from random_perturb import random_perturb
from pso import PSO
from lstm import Lstm
from seq2seq.utils import softmax, print_header, get_events
from seq2seq.seq2seq_attention import Seq2seqAttention
import argparse
import copy
import tensorflow as tf
import random

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)



def get_args(jupyter_args = None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--permute_truncated', required=False, action='store_true', help="")
    parser.add_argument('--use_prob_embedding', required=False, action='store_true', help="")
    parser.add_argument('--sequence_length', required=False, type=int, default=10, help="")
    parser.add_argument('--rv', required=False, type=int, default=1, help="")
    parser.add_argument('--ps_epochs', required=False, type=int, default=50, help="")


    if jupyter_args is not None:
        args = parser.parse_args(jupyter_args)
    else: 
        args = parser.parse_args()

    return args

#jupyter_args = ['--permute_truncated', '--use_prob_embedding']
args = get_args()
seq2seq_config = {"sequence_length": args.sequence_length, 
                  "permute_truncated": args.permute_truncated,
                  "use_prob_embedding": args.use_prob_embedding,
                  "rv": args.rv
                  }   

# -----------get the preprocessed training and testing saved as .npy files
test_label_infection = np.load('preprocessed/test_label_infection.npy')
train_label_infection = np.load('preprocessed/train_label_infection.npy')
test_data_infection = np.load('preprocessed/test_data_infection.npy')
train_data_infection = np.load('preprocessed/train_data_infection.npy')

test_label_reconnaissance = np.load('preprocessed/test_label_reconnaissance.npy')
train_label_reconnaissance = np.load('preprocessed/train_label_reconnaissance.npy')
test_data_reconnaissance = np.load('preprocessed/test_data_reconnaissance.npy')
train_data_reconnaissance = np.load('preprocessed/train_data_reconnaissance.npy')

test_label_attack = np.load('preprocessed/test_label_attack.npy')
train_label_attack = np.load('preprocessed/train_label_attack.npy')
test_data_attack = np.load('preprocessed/test_data_attack.npy')
train_data_attack = np.load('preprocessed/train_data_attack.npy')

all_data = {"infection": 
                        {
                        'train': [train_data_infection, train_label_infection], 
                        'test': [test_data_infection, test_label_infection]
                        },
            "attack": 
                        {
                        'train': [train_data_attack, train_label_attack], 
                        'test': [test_data_attack, test_label_attack]
                        },
            "reconnaissance": 
                        {
                        'train': [train_data_reconnaissance, train_label_reconnaissance], 
                        'test': [test_data_reconnaissance, test_label_reconnaissance]
                        }
            }

# ----------------create per-step detectors----------------------
ps_attack = Lstm("ps-detector-attack")
ps_attack.add_dataset(all_data['attack']) 

ps_recon = Lstm("ps-detector-recon")
ps_recon.add_dataset(all_data['reconnaissance']) 

ps_infec = Lstm("ps-detector-infec")
ps_infec.add_dataset(all_data['infection']) 

# ----------------train per-step detectors----------------------
metrics_dict = {}
for detector in [ps_attack, ps_recon, ps_infec]:
    #train data
    train_data = detector.dataset['train']
    train_examples = train_data[0]
    train_labels = train_data[1]

    #test data
    test_data = detector.dataset['test']
    test_examples = test_data[0]
    test_labels = test_data[1]
        
    features_len = train_examples.shape[1]
    print('features len is ', features_len)
    
    print_header("Training {} detector".format(detector.name))
    detector.learning(features_len, train_examples, train_labels, kind='', epochs=args.ps_epochs)
                    
    print_header("Measureing {} detector performance on test data".format(detector.name))
    _, _, metrics_dict_ps = detector.detection(test_examples, test_labels, kind='')
    metrics_dict[detector.name] = metrics_dict_ps

# -----------------seq2seq stage----------------------
#get events
events = get_events(ps_attack, ps_recon, ps_infec, all_data['infection']['train'][0])

#init seq2seq
seq2seq = Seq2seqAttention('seq2seq')

#train seq2seq
seq2seq.learning(events, all_data['infection']['train'][1], seq2seq_config)

#get seq2seq tagged events
events_preds, tagged_seq2seq = seq2seq.analysis(events, all_data['infection']['train'][1], seq2seq_config)

#get per-step infection detector tagged windows
preds_ps_infec = ps_infec.predict(all_data['infection']['train'][0], kind='')
preds_ps_infec = np.array(preds_ps_infec).squeeze()
print('preds ps infec shape is', preds_ps_infec.shape)
tagged_ps_infec = []
for idx, pred in enumerate(preds_ps_infec):
    if pred>0.5:
        tagged_ps_infec.append(idx)


# -----------------relabeling---------------------
#strategy 1
retrain_pos = []
retrain_neg = []
for idx in tagged_ps_infec:
    if idx in tagged_seq2seq:
        retrain_pos.append(idx)
    else:
        retrain_neg.append(idx)

#overwritte original dataset with new positive and negative labels
retrain_labels = copy.deepcopy(all_data['infection']['train'][1])
retrain_data = copy.deepcopy(all_data['infection']['train'][0])
for idx, l in enumerate(retrain_labels):
    if idx in retrain_pos:
        retrain_labels[idx] = 1
    if idx in retrain_neg:
        retrain_labels[idx] = 0


# -----------------retrain per-step infection detector with new labels---------------------
#def retrain_detector(detector, retrain_data, retrain_labels, test_data, test_labels):


features_len = retrain_data.shape[1]
print('features len is ', features_len)

print_header("Retraining {} detector".format('infection'))
ps_infec.learning(features_len, retrain_data, retrain_labels, kind='', epochs=args.ps_epochs)
                
print_header("Measureing {} detector performance on test data".format('infection'))
_, _, metrics_dict_new = ps_infec.detection(detector.dataset['test'][0], detector.dataset['test'][1], kind='')

print_header("Per-step infection detector metrics BEFORE relabeling")
print( metrics_dict[ps_infec.name])

print_header("Per-step infection detector metrics AFTER relabeling")
print(metrics_dict_new)