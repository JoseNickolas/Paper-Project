from zmq import device
from mimic import Mimic3
from word2vec import Word2Vec, SkipGramDataset
from siamese_cnn import SiameseCNN

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm, trange


# constants
DATASET_PATH = 'Dataset/all_files/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# set hyperparamers

# word2vec
emb_dim = 100
alpha = 1
beta = 10
sg_lr = 0.1
sg_epochs = 1
sg_batch = 512

# siamese
feature_maps = 100
kernel_size = 5
spp_levels = (4, 2, 1)
out_dim = 10
scnn_epochs = 1000
scnn_batch = 20
sc_lr = 0.1
margin = 5



# create the main dataset of medical concept sequences (preprocessing) 
dataset = Mimic3(DATASET_PATH)
num_codes = dataset.num_codes() # total number of unique medical codes


# train word2vec
word2vec = Word2Vec(emb_dim, num_codes)
word2vec.to(DEVICE)

sg_dataset = SkipGramDataset(dataset, alpha=alpha, beta=beta)
sg_dataloader = DataLoader(sg_dataset, batch_size=sg_batch)


criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(lr=sg_lr, params=word2vec.parameters())


for i in trange(sg_epochs, desc='Training word2vec'):
    for center, context in tqdm(sg_dataloader, desc=f'epoch {i}'):
        center, context = center.to(DEVICE), context.to(DEVICE)
        
        pred = word2vec(center)
        loss = criterion(pred, context)
        
        loss.backward()
        optim.step()
        optim.zero_grad()



# Siamese CNN with SPP
scnn = SiameseCNN(feature_maps, kernel_size, spp_levels, out_dim)
scnn.to(DEVICE)


optim = torch.optim.Adam(lr=sc_lr, params=scnn.parameters())


for epoch in range(scnn_epochs):
    
    patient_reps = []
    patient_cohorts = []
    
    for k, (code, cohort) in enumerate(dataset):
        
        code = torch.tensor(code).long().to(DEVICE)
        # cohort = torch.tensor(cohort).long().to(DEVICE)
        
        emb = word2vec.emb(code)  # ncodes, emb_dim
        emb = emb.view(1, 1, emb.shape[0], emb.shape[1]) # 1 (batch), 1 (channel), ncodes, emb_dim
        patent_rep = scnn(emb)

        patient_reps.append(patent_rep)
        patient_cohorts.append(cohort)
        
        if k % scnn_batch == scnn_batch-1:
            
            patient_reps = torch.cat(patient_reps, dim=0) # scnn_batch, out_dim
            pdist = F.cosine_similarity(patient_reps, patient_reps[:, None, :], dim=-1)
            
            y = torch.zeros((scnn_batch, scnn_batch))
            for i in range(scnn_batch):
                for j in range(scnn_batch):
                    if len(np.intersect1d(patient_cohorts[i], patient_cohorts[j])) != 0:
                        y[i, j] = 1
            y = y.to(DEVICE)
                        
            loss = 0.5 * (1 - y) * pdist**2 + 0.5 * y * torch.maximum(torch.tensor(0.), margin - pdist)
            loss = loss.mean()
            
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            patient_reps = []
            patient_cohorts = []

        

# Evaluate
