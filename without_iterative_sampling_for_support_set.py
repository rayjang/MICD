# select dataset out of bank, wos, medium, and rnd.
DATASET = 'bank'
from enum import Enum
from keybert import KeyBERT
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
from sklearn.model_selection import train_test_split

def get_jsonl_data(jsonl_path: str):
    assert jsonl_path.endswith(".jsonl")
    out = list()
    with open(jsonl_path, 'r', encoding="utf-8") as file:
        for line in file:
            #print(line.strip())
            j = json.loads(line.strip())
            out.append(j)
            
    return out

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.WARN,
    handlers=[LoggingHandler()],
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# iterative 20 times with different random seeds
set_seed(42)


# read data
DATASET_NM = DATASET
DATA_DIR = './data/' + DATASET_NM + '/'

train_json  = get_jsonl_data(DATA_DIR + 'train.jsonl')
test_json = get_jsonl_data(DATA_DIR + 'test.jsonl')

train = pd.DataFrame(train_json)
test = pd.DataFrame(test_json)

for i in range(train.shape[0]):
    if len(train.loc[i,'label']) == 1:
        train.loc[i,'label'] = '00'+train.loc[i,'label']
    elif len(train.loc[i,'label']) == 2:
        train.loc[i,'label'] = '0'+train.loc[i,'label']
        
for i in range(test.shape[0]):
    if len(test.loc[i,'label']) == 1:
        test.loc[i,'label'] = '00'+test.loc[i,'label']
    elif len(test.loc[i,'label']) == 2:
        test.loc[i,'label'] = '0'+test.loc[i,'label']

        
train['keyword1'] = ''
train['keyword2'] = ''
test['keyword1'] = ''
test['keyword2'] = ''


# by using KeyBERT, we extract 2 primary keywords per each samples
kw_model = KeyBERT(model=f'./transformer_models/{DATASET}/fine-tuned')

for i in train.index:
    text = train.loc[i,'sentence']
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1),stop_words='english', highlight=False, top_n=2)
    if len(keywords) == 0:
        continue
    
    if len(keywords) ==1:
        train.loc[i, 'keyword1'] = keywords[0][0]
    else:
        train.loc[i, 'keyword1'] = keywords[0][0]
        train.loc[i, 'keyword2'] = keywords[1][0]

for i in test.index:
    text = test.loc[i,'sentence']
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1),stop_words='english', highlight=False, top_n=2)
    if len(keywords) == 0:
        continue
    if len(keywords) ==1:
        test.loc[i, 'keyword1'] = keywords[0][0]
    else:
        test.loc[i, 'keyword1'] = keywords[0][0]
        test.loc[i, 'keyword2'] = keywords[1][0]
        

import pickle
with open(f'train_kw_{DATASET}.pickle', 'wb') as f:
    pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)
with open(f'test_kw_{DATASET}.pickle', 'wb') as f:
    pickle.dump(test, f, pickle.HIGHEST_PROTOCOL)


sentence_list = list(train['sentence'])
label_list = list(train['label'])

kw1_list = list(train['keyword1'])
kw2_list = list(train['keyword2'])


c = list(zip(sentence_list, label_list, kw1_list, kw2_list))
random.shuffle(c)
sentence_list, label_list, kw1_list, kw2_list = zip(*c)


# split into train and valid data from original train data
from sklearn.model_selection import train_test_split
x_train, x_eval, y_train, y_eval = train_test_split(sentence_list, label_list, test_size=0.1, random_state=42)


def testing(epoch, m, alpha, diversity_p):
    
    # sentence pairs generation with hard positive sampling
    def sentence_pairs_generation(sentences, labels, kw1_list, kw2_list, pairs):
        # initialize two empty lists to hold the (sentence, sentence) pairs and
        # labels to indicate if a pair is positive or negative
        num_gen_p = 1
        num_gen_n = 1

        numClassesList = np.unique(labels)
        idx = [np.where(labels == i)[0] for i in numClassesList]

        mod_num = 0
        for idxA in range(len(sentences)):      
            currentSentence = sentences[idxA]
            label = labels[idxA]

            current_pos_cnt = 0
            kkkk = idx[np.where(numClassesList==label)[0][0]]
            random.shuffle(kkkk)
            for i in kkkk:
                if current_pos_cnt == num_gen_p+1:
                    break
                # print(len(sentences))
                # print(i)
                posSentence = sentences[i]
                if (kw1_list[idxA].lower() not in posSentence.lower() ) and ( kw2_list[idxA].lower() not in posSentence.lower()):
                    pairs.append(InputExample(texts=[currentSentence, posSentence], label=1.0))
                    current_pos_cnt += 1

            #### neg sampling
            current_neg_cnt = 0

            while current_neg_cnt < num_gen_n*2:
                negIdx = np.where(labels != label)[0]
                negSentence = sentences[np.random.choice(negIdx)]
                pairs.append(InputExample(texts=[currentSentence, negSentence], label=0.0))
                current_neg_cnt += 1

        return (pairs)

    # generate sentence pairs for training and validation
    train_examples = [] 
    train_examples = sentence_pairs_generation(np.array(sentence_list), np.array(label_list), kw1_list, kw2_list,train_examples)

    valid_examples = []
    valid_examples = sentence_pairs_generation(np.array(x_eval), np.array(y_eval), kw1_list, kw2_list,valid_examples)

    eval_s1 = []
    eval_s2 = []
    eval_label = []

    for i in range(len(valid_examples)):
        eval_s1.append(valid_examples[i].texts[0])
        eval_s2.append(valid_examples[i].texts[1])
        eval_label.append(valid_examples[i].label)

    evaluator = evaluation.EmbeddingSimilarityEvaluator(eval_s1, eval_s2, eval_label)

    with open(f'train_examples_{DATASET}.pickle', 'wb') as f:
        pickle.dump(train_examples, f, pickle.HIGHEST_PROTOCOL)
    with open(f'evaluator_{DATASET}.pickle', 'wb') as f:
        pickle.dump(evaluator, f, pickle.HIGHEST_PROTOCOL)    


    class SiameseDistanceMetric(Enum):
        EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
        MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
        COSINE_DISTANCE = lambda x, y: 1-F.cosine_similarity(x, y)

    # our proposed contrastive loss
    class LargeN_ContrastiveLoss(nn.Module):
        def __init__(self, model: SentenceTransformer, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, margin: float = 0.3, m: float = 0.8, alpha: float = 0.1):
            super(Online_rcl_ContrastiveLoss, self).__init__()
            self.model = model
            self.margin = margin
            self.m = m
            self.alpha = alpha
            self.distance_metric = distance_metric

        def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, size_average=False):
            embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

            distance_matrix = self.distance_metric(embeddings[0], embeddings[1])
            negs = distance_matrix[labels == 0]
            poss = distance_matrix[labels == 1]


            negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
            positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

            if len(poss) > 0 and len(negs) > 0:
                intra_class_reg = poss[poss < diversity_p*m].sum()
            else:
                intra_class_reg = 0


            positive_loss = torch.log(positive_pairs.pow(2).sum()+1)
            negative_loss = F.relu(m - negative_pairs).pow(2).sum()


            loss = positive_loss + negative_loss + self.alpha*intra_class_reg
            return loss


    model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
    # S-BERT adaptation
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
    train_loss = LargeN_ContrastiveLoss(model)
   
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epoch, warmup_steps=20, show_progress_bar=False, evaluation_steps=201, evaluator=evaluator, save_best_model=True, output_path=f'./model_wos_contra_{str(epoch)}_{str(m)}_{str(alpha)}_{str(diversity_p)}')
   


    sgd = LogisticRegression()

    n_classes_test = test["label"].nunique()
    n_shot = 5
    random_seeds = [i for i in range(1,20)]

    acc_list = []

    # get mean accuacies by conducting experiment 20 times with different random seed
    for rs in random_seeds:
        set_seed(rs)
        c = list(zip(test["sentence"].tolist(), test["label"].tolist(), test["keyword1"].tolist(), test["keyword2"].tolist()))
        random.shuffle(c)
        x_test, y_test, keyword1, keyword2 = zip(*c)


        label_list = list(Counter(y_test).keys())
        x_support = []
        y_support = []
        k_support = []

        x_query = []
        y_query = []
        k1_query = []
        k2_query = []



        for l in label_list:
            l_idx = [i for i, _ in enumerate(y_test) if _ == l]
            # randomly select N*k samples
            x_support.extend([x_test[i] for i in l_idx[:5]])
            y_support.extend([y_test[i] for i in l_idx[:5]])
            k_support.extend([x_test[i] for i in l_idx[:5]])

            x_query.extend([x_test[i] for i in l_idx[5:]])
            y_query.extend([y_test[i] for i in l_idx[5:]])
            k1_query.extend([keyword1[i] for i in l_idx[5:]])
            k2_query.extend([keyword2[i] for i in l_idx[5:]])

        x_query = list(x_query)
        y_query = list(y_query)

        c = list(zip(x_support, y_support))
        random.shuffle(c)
        x_support, y_support = zip(*c)

        x_emb_support = model.encode(x_support)
        x_emb_query = model.encode(x_query)

        sgd.fit(x_emb_support, y_support)
        y_pred_query = sgd.predict(x_emb_query)

        acc_list.append(accuracy_score(y_query, y_pred_query))
        f1_list.append(f1_score(y_query, y_pred_query, average='micro'))

    # mean accuracy of our model without intra-class mix-up augmentation
    print('Acc. original', np.mean(acc_list))
    org_acc = np.mean(acc_list)


    def mixup(x1, x2, lam):
        mixed_x = lam*x1 + (1-lam)*x2
        return mixed_x

    sgd = LogisticRegression()

    n_classes_test = test["label"].nunique()
    n_shot = 5


    random_seeds = [i for i in range(1,20)]
    acc_list = []

    # get mean accuacies by conducting experiment 20 times with different random seed
    for rs in random_seeds:
        set_seed(rs)
        c = list(zip(test["sentence"].tolist(), test["label"].tolist(), test["keyword1"].tolist(), test["keyword2"].tolist()))
        random.shuffle(c)
        x_test, y_test, keyword1, keyword2 = zip(*c)


        label_list = list(Counter(y_test).keys())
        x_support = []
        y_support = []
        k_support = []
        
        x_query = []
        y_query = []
        k1_query = []
        k2_query = []
        

        # shot =1
        for l in label_list:
            l_idx = [i for i, _ in enumerate(y_test) if _ == l]
            
            x_support.extend([x_test[i] for i in l_idx[:5]])
            y_support.extend([y_test[i] for i in l_idx[:5]])
            k_support.extend([x_test[i] for i in l_idx[:5]])

            x_query.extend([x_test[i] for i in l_idx[5:]])
            y_query.extend([y_test[i] for i in l_idx[5:]])
            k1_query.extend([keyword1[i] for i in l_idx[5:]])
            k2_query.extend([keyword2[i] for i in l_idx[5:]])

        x_query = list(x_query)
        y_query = list(y_query)

        c = list(zip(x_support, y_support))
        random.shuffle(c)
        x_support, y_support = zip(*c)

        x_emb_support = model.encode(x_support)
        x_emb_query = model.encode(x_query)

        # added
        current_y = max(y_support)


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

        y_support = list(y_support)
        x_emb_support = np.concatenate([x_emb_support, np.array(augmented_x_emb_support)])
        y_support.extend(augmented_y_support)


        x_emb_support = [ x_emb_support[i] for i in range(x_emb_support.shape[0])]
        c = list(zip(x_emb_support, y_support))
        random.shuffle(c)
        x_emb_support, y_support = zip(*c)
        #added


        sgd.fit(x_emb_support, y_support)
        y_pred_query = sgd.predict(x_emb_query)

        for i, y in enumerate(y_pred_query):
            if len(y) == 6:
                y_pred_query[i] = y[:3]

        acc_list.append(accuracy_score(y_query, y_pred_query))
    # mean accuracy of our model with intra-class mix-up augmentation
    print('Acc. mixup', np.mean(acc_list))

testing(1, 0.6, 0.005, 0.1)






