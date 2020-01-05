import argparse
from collections import Counter
import code
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import WeightedAvgModel, SNLIModel


import numpy as np
import pandas as pd
import pkuseg

from elmoformanylangs import Embedder

e = Embedder('/share/data/lang/users/zeweichu/projs/faqbot/zhs.model')
sents = ["今天天气真好啊",
        "潮水退了就知道谁没穿裤子"]

seg = pkuseg.pkuseg()

allPre = []
with open("../nlpcc-iccpol-2016.kbqa.kb") as fin:
    for line in fin:
        allPre.append(line.split(" ||| ")[1])

def getAnswerPatten(inputPath = '../nlpcc-iccpol-2016.kbqa.training-data'):
    inputEncoding = 'utf8'

    data = []
    with open(inputPath, 'r', encoding=inputEncoding) as fi:
        pattern = re.compile(r'[·•\-\s]|(\[[0-9]*\])') #pattern to clean predicate, in order to be consistent with KB clean method
        for line in fi:
            if line.find('<q') == 0:  #question line
                qRaw = line[line.index('>') + 2:].strip()
                continue
            elif line.find('<t') == 0:  #triple line
                triple = line[line.index('>') + 2:]
                s = triple[:triple.index(' |||')].strip()
                triNS = triple[triple.index(' |||') + 5:]
                p = triNS[:triNS.index(' |||')]
                p, num = pattern.subn('', p)
                if qRaw.find(s) != -1:
                    qRaw = qRaw.replace(s,'', 1)
                # 把问题中出现的subject去掉

                # <question id=1>   《机械设计基础》这本书的作者是谁？
                # <triple id=1>   机械设计基础 ||| 作者 ||| 杨可桢，程光蕴，李仲生
                # <answer id=1>   杨可桢，程光蕴，李仲生

                #  这本书的作者是谁？ ||| 作者
               
                data.append([qRaw, p])
         
            else: continue
    return data

data = getAnswerPatten()
# code.interact(local=locals())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SNLIModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
# code.interact(local=locals())

num_epochs = 10
batch_size = 64
margin = 0.5
for epoch in range(num_epochs):
    print("starting epoch {}".format(epoch))
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]

        # sample negative predicates
        questions = [q for q, _ in batch_data]
        predicates = [p for _, p in batch_data]
        negative_predicates = []
        for p in predicates:
            neg_pre = random.sample(allPre, 1)[0]
            while neg_pre == p:
                neg_pre = random.sample(allPre, 1)[0]
            negative_predicates.append(neg_pre)

        questions = [seg.cut(sent) for sent in questions]
        q_embeddings = np.concatenate([np.expand_dims(emb.mean(1), 0) for emb in e.sents2elmo(questions, -2)])

        predicates = [seg.cut(sent) for sent in predicates]
        p_embeddings =  np.concatenate([np.expand_dims(emb.mean(1), 0) for emb in e.sents2elmo(predicates, -2)])
        
        negative_predicates = [seg.cut(sent) for sent in negative_predicates]
        neg_p_embeddings =  np.concatenate([np.expand_dims(emb.mean(1), 0) for emb in e.sents2elmo(negative_predicates, -2)])

        q_embeddings = torch.Tensor(q_embeddings).long().to(DEVICE)
        p_embeddings = torch.Tensor(p_embeddings).long().to(DEVICE)
        neg_p_embeddings = torch.Tensor(neg_p_embeddings).long().to(DEVICE)
       
        pos_score = model(q_embeddings, p_embeddings)
        neg_score = model(q_embeddings, neg_p_embeddings)
       
        loss = -(pos_score - neg_score - margin)
        loss[loss < 0] = 0
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i // batch_size % 100 ==0:
            print("batch {} loss {}".format(i // batch_size, loss.item()))

    model_path = "snli_checkpoints/elmo_sim_model-epoch{}.py".format(epoch)
    print("saving model to {}".format(model_path))
    torch.save(model.state_dict(), model_path)
