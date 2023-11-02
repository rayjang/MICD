import pandas as pd
import json
from keybert import KeyBERT
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
import logging
from typing import Iterable, Dict
import torch.nn.functional as F
from torch import nn, Tensor
from sentence_transformers.losses.ContrastiveLoss import SiameseDistanceMetric
from sentence_transformers.SentenceTransformer import SentenceTransformer
# custom model
from sentence_transformers import SentenceTransformer, models
from torch import nn
import pandas as pd
import numpy as np
import numpy as np
from scipy.stats import entropy
from math import log, e
import pandas as pd
import torch
import random
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, evaluation
from torch.utils.data import DataLoader
import pickle
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import entropy
from math import log, e
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import copy
import logging
import warnings


warnings.filterwarnings(action='ignore')

DATASET = 'bank'
model_nm = 'encoder_bank'

with open(f'train_kw_{DATASET}.pickle', 'rb') as f:
    train = pickle.load(f)
with open(f'test_kw_{DATASET}.pickle', 'rb') as f:
    test = pickle.load(f)


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.ERROR,
    handlers=[LoggingHandler()],
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

def testing(model_nm, first_p, min_v, a_p):
    model = SentenceTransformer(model_nm)
    
    def mixup(x1, x2, lam):
        mixed_x = lam*x1 + (1-lam)*x2
        return mixed_x


    sgd = LogisticRegression()


    n_classes_test = test["label"].nunique()
    n_shot = 5

    # get mean accuacies by conducting experiment 20 times with different random seed
    random_seeds =[i for i in range(1,20)]


    acc_1= []
    acc_2= []
    acc_3= []
    acc_4= []
    acc_5= []

    # get mean accuacies by conducting experiment 20 times with different random seed
    for rs in random_seeds:

        c = list(zip(test["sentence"].tolist(), test["label"].tolist()))
        random.shuffle(c)
        x_test, y_test = zip(*c)

        label_list = list(Counter(y_test).keys())
        x_support = []
        y_support = []
        x_query = []
        y_query = []

        for shot in range(1,n_shot+1):

            if shot == 1:
                for l in label_list:
                    l_idx = [i for i, _ in enumerate(y_test) if _ == l]
                    x_support.extend([x_test[l_idx[0]]])
                    y_support.extend([y_test[l_idx[0]]])
                    x_query.extend([x_test[i] for i in l_idx[1:]])
                    y_query.extend([y_test[i] for i in l_idx[1:]])

                x_query = list(x_query)
                y_query = list(y_query)

                x_emb_support = model.encode(x_support)
                x_emb_query = model.encode(x_query)

                sgd = LogisticRegression()

                sgd.fit(x_emb_support, y_support)
                y_pred_query = sgd.predict(x_emb_query)

                acc_1.append(accuracy_score(y_query, y_pred_query))


                nxt_x_query = []
                nxt_y_query = []

                for _l in label_list:
                    l_idx_s = [i for i, _ in enumerate(y_support) if _ == _l]
                    x_support_tmp = x_support[l_idx_s[0]]
                    x_emb_support_tmp = model.encode(x_support_tmp)

                    l_idx = [i for i, _ in enumerate(y_query) if _ == _l]
                    x_c = [x_query[i] for i in l_idx]
                    y_c = [y_query[i] for i in l_idx]
                    x_c_emb = model.encode(x_c)


                    cos_list = []
                    for i, xeq in enumerate(x_c_emb):
                        
                        cos = F.cosine_similarity(torch.tensor(x_emb_support_tmp)[None,:], torch.unsqueeze(torch.tensor(xeq), dim=0))
                        cos_list.append(cos)

                    v = min(np.array(cos_list), key=lambda x:abs(x-np.percentile([ x.detach().cpu().numpy()[0] for x in cos_list], min(min_v,int(a_p*shot)))))
                    cos_idx = list(cos_list).index(v)

                    x_support = list(x_support)
                    y_support = list(y_support)

                    x_support.append(x_c[cos_idx])
                    y_support.append(y_c[cos_idx])

                    nxt_x_query.extend(x_c[:cos_idx])
                    nxt_x_query.extend(x_c[cos_idx+1:])
                    nxt_y_query.extend(y_c[:cos_idx])
                    nxt_y_query.extend(y_c[cos_idx+1:])
                x_query = nxt_x_query
                y_query = nxt_y_query


            else:
                c = list(zip(x_support, y_support))
                random.shuffle(c)
                x_support, y_support = zip(*c)


                x_emb_support = model.encode(x_support)
                x_emb_query = model.encode(x_query)

                nn = 5
                sgd = KNeighborsClassifier(n_neighbors=nn)


                augmented_x_emb_support = []
                augmented_y_support = []

                for i in range(len(x_emb_support)):
                    for j in range(len(x_emb_support)):
                        if y_support[i] == y_support[j]:
                            aug_x = mixup(x_emb_support[i], x_emb_support[j], 0.5)
                            augmented_x_emb_support.append(aug_x)
                            augmented_y_support.append(y_support[i])

                            aug_x = mixup(x_emb_support[i], x_emb_support[j], 0.6)
                            augmented_x_emb_support.append(aug_x)
                            augmented_y_support.append(y_support[i])

                            aug_x = mixup(x_emb_support[i], x_emb_support[j], 0.7)
                            augmented_x_emb_support.append(aug_x)
                            augmented_y_support.append(y_support[i])

                            aug_x = mixup(x_emb_support[i], x_emb_support[j], 0.8)
                            augmented_x_emb_support.append(aug_x)
                            augmented_y_support.append(y_support[i])

                            aug_x = mixup(x_emb_support[i], x_emb_support[j], 0.9)
                            augmented_x_emb_support.append(aug_x)
                            augmented_y_support.append(y_support[i])


                x_emb_support = np.concatenate([x_emb_support, np.array(augmented_x_emb_support)])
                y_support_inferece = list(y_support)
                y_support_inferece.extend(augmented_y_support)


                x_emb_support = [ x_emb_support[i] for i in range(x_emb_support.shape[0])]
                c = list(zip(x_emb_support, y_support_inferece))
                random.shuffle(c)
                x_emb_support, y_support_inferece = zip(*c)


                sgd.fit(x_emb_support, y_support_inferece)
                x_emb_query = model.encode(x_query)
                y_pred_query = sgd.predict(x_emb_query)

                if shot ==2:
                    acc_2.append(accuracy_score(y_query, y_pred_query))
                elif shot ==3:
                    acc_3.append(accuracy_score(y_query, y_pred_query))
                elif shot ==4:
                    acc_4.append(accuracy_score(y_query, y_pred_query))
                elif shot ==5:
                    acc_5.append(accuracy_score(y_query, y_pred_query))


                nxt_x_query = []
                nxt_y_query = []
                for _l in label_list:

                    l_idx_s = [i for i, _ in enumerate(y_support) if _ == _l]
                    x_support_tmp = [x_support[lis] for lis in l_idx_s]
                    x_emb_support_tmp = model.encode(x_support_tmp)

                    l_idx = [i for i, _ in enumerate(y_query) if _ == _l]
                    x_c = [x_query[i] for i in l_idx]
                    y_c = [y_query[i] for i in l_idx]
                    x_c_emb = model.encode(x_c)

                    x_emb_support = list(x_emb_support)


                    cos_list = []
                    for i, xeq in enumerate(x_c_emb):
                        
                        cos = F.cosine_similarity(torch.mean(torch.tensor(x_emb_support_tmp),0,False)[None,:], torch.unsqueeze(torch.tensor(xeq), dim=0))
                        
                        cos_list.append(cos)
                    v = min(np.array(cos_list), key=lambda x:abs(x-np.percentile([ x.detach().cpu().numpy()[0] for x in cos_list], min(min_v,int(a_p*shot)))))
                    cos_idx = list(cos_list).index(v)

                    x_support = list(x_support)
                    y_support = list(y_support)

                    x_support.append(x_c[cos_idx])
                    y_support.append(y_c[cos_idx])

                    nxt_x_query.extend(x_c[:cos_idx])
                    nxt_x_query.extend(x_c[cos_idx+1:])
                    nxt_y_query.extend(y_c[:cos_idx])
                    nxt_y_query.extend(y_c[cos_idx+1:])
                x_query = nxt_x_query
                y_query = nxt_y_query


    print(f'1 - acc_score. Fit: {np.mean(acc_1)}\n')
    print(f'2 - acc_score. Fit: {np.mean(acc_2)}\n')
    print(f'3 - acc_score. Fit: {np.mean(acc_3)}\n')
    print(f'4 - acc_score. Fit: {np.mean(acc_4)}\n')
    print(f'5 - acc_score. Fit: {np.mean(acc_5)}\n')

testing(model_nm, 10, 60, 10)