import os
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#In case gpu dont hjave sufficient memory
#DEVICE = torch.device('cpu')

X = []
if os.path.exists("yoochoose_full_unscaled.dmp"):
    with open("yoochoose_full_unscaled.dmp", 'rb') as pcf:
        X = pickle.load(pcf)
        X = X.astype('int16')
else:
    with pd.read_csv(os.getcwd() +"/featureSet_full_yoochoose_2021-05-17_05_59_58.csv", # use appropriate file name
                     index_col = [0], chunksize= 2000) as reader:
        for chunk in reader:
            chunk = chunk.fillna(0).astype('uint16')
            if len(X) == 0:
                X = [np.zeros(chunk.shape[1])]
            X = np.vstack([X, chunk]).astype('uint16')
            print(f"({X.shape[0]}/{X.shape[1]}) Parsed...")
    with open("yoochoose_full_unscaled.dmp", 'wb') as pcf:
        pickle.dump(X, pcf)


train_dl = DataLoader( TensorDataset(torch.Tensor(X)), batch_size=100, shuffle=False)
del(X)

class RBM():
    def __init__(self, visible_dim, hidden_dim, gaussian_hidden_distribution=False):
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.gaussian_hidden_distribution = gaussian_hidden_distribution

        # intialize parameters
        self.W = torch.randn(visible_dim, hidden_dim).to(DEVICE) * 0.1
        self.h_bias = torch.zeros(hidden_dim).to(DEVICE)  
        self.v_bias = torch.zeros(visible_dim).to(DEVICE)

        # parameters for learning with momentum
        self.W_momentum = torch.zeros(visible_dim, hidden_dim).to(DEVICE)
        self.h_bias_momentum = torch.zeros(hidden_dim).to(DEVICE)
        self.v_bias_momentum = torch.zeros(visible_dim).to(DEVICE)

    def predict_hidden(self, v):
        activation = torch.mm(v, self.W) + self.h_bias
        if self.gaussian_hidden_distribution:
            return activation, torch.normal(activation, torch.tensor([1]).to(DEVICE))
        else:
            p = torch.sigmoid(activation)
            return p, torch.bernoulli(p)

    def predict_visible(self, h):
        activation = torch.mm(h, self.W.t()) + self.v_bias
        p = torch.sigmoid(activation)
        return p
    
    def update_weights(self, v0, vk, ph0, phk, lr, momentum_coef, weight_decay, batch_size):
        self.W_momentum *= momentum_coef
        self.W_momentum += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)

        self.h_bias_momentum *= momentum_coef
        self.h_bias_momentum += torch.sum((ph0 - phk), 0)

        self.v_bias_momentum *= momentum_coef
        self.v_bias_momentum += torch.sum((v0 - vk), 0)

        self.W += lr*self.W_momentum/batch_size
        self.h_bias += lr*self.h_bias_momentum/batch_size
        self.v_bias += lr*self.v_bias_momentum/batch_size

        self.W -= self.W * weight_decay

models = [] # store trained RBM models
visible_dim = 16076 #43070 for diginetica
rbm_train_dl = train_dl
for hidden_dim in [12800, 3200, 800, 200]:
    # The GPU I have 8GB memory therefore can't allocate memory for 1st and 2nd step
    DEVICE = torch.device('cuda' if (torch.cuda.is_available() and visible_dim < 20000) else 'cpu')
    # configs - we have a different configuration for the last layer
    num_epochs = 30 if hidden_dim == 200 else 10
    lr = 1e-3 if hidden_dim == 200 else 0.1
    use_gaussian = hidden_dim == 200
    best_loss = 1e6
    # train RBM
    rbm = RBM(visible_dim=visible_dim, hidden_dim=hidden_dim, 
              gaussian_hidden_distribution=use_gaussian)
    loss = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        train_loss = 0
        for i, data_list in enumerate(rbm_train_dl):
            v0, pvk = data_list[0].to(DEVICE), data_list[0].to(DEVICE)

            _, hk = rbm.predict_hidden(v0)
            pvk = rbm.predict_visible(hk)

            ph0, _ = rbm.predict_hidden(v0)
            phk, _ = rbm.predict_hidden(pvk)
            # update weights
            rbm.update_weights(v0, pvk, ph0, phk, lr, 
                               momentum_coef=0.5 if epoch < 5 else 0.9, 
                               weight_decay=2e-4, 
                               batch_size=v0.shape[0])
            train_loss += loss(v0, pvk)
        print(f"{str(datetime.now())} - Layer: {hidden_dim} epoch {epoch}: {train_loss/len(train_dl)}")
    models.append(rbm)

    # reconstruct new data loader based on the output of trained model
    new_data = [rbm.predict_hidden(data_list[0].to(DEVICE))[0].cpu().detach().numpy() for data_list in rbm_train_dl]
    hidden_embeddings = np.concatenate(new_data)
    del(new_data)
    with open(f"yoochoose_{hidden_dim}_input.dmp", 'wb') as pcf:
        pickle.dump(hidden_embeddings, pcf)
    rbm_train_dl = DataLoader(
        TensorDataset(torch.Tensor(hidden_embeddings)), 
        batch_size=100, shuffle=False
    )
    visible_dim = hidden_dim

class DeepAE(nn.Module):
    def __init__(self, models):
        super(DeepAE, self).__init__()

        # build encoders and decoders based on weights from each 
        self.encoders = nn.ParameterList([nn.Parameter(model.W.clone()) for model in models])
        self.encoder_biases = nn.ParameterList([nn.Parameter(model.h_bias.clone()) for model in models])
        self.decoders = nn.ParameterList([nn.Parameter(model.W.clone()) for model in reversed(models)])
        self.decoder_biases = nn.ParameterList([nn.Parameter(model.v_bias.clone()) for model in reversed(models)])

    def forward(self, v):
        p_h = self.encode(v)
        return self.decode(p_h)

    def encode(self, v):
        p_v = v
        for i in range(len(self.encoders)):
            activation = torch.mm(p_v, self.encoders[i]) + self.encoder_biases[i]
            p_v = torch.sigmoid(activation)
        return activation

    def decode(self, h):
        p_h = h
        for i in range(len(self.encoders)):
            activation = torch.mm(p_h, self.decoders[i].t()) + self.decoder_biases[i]
            p_h = torch.sigmoid(activation)
        return p_h

# Returning back to CPU since we dont have sufficient memory to hold all models in VRAM
DEVICE = torch.device('cpu')
dae = DeepAE(models).to(DEVICE)
path = os.path.join(os.getcwd(), f'dae_model_backup_pretrained.pth')
torch.save(dae.state_dict(), path)
optimizer = torch.optim.Adam(dae.parameters(), 1e-3)
loss = nn.MSELoss()

for epoch in range(50):
    training_loss = []
    for i, features_list in enumerate(train_dl): 
        features = features_list[0].to(DEVICE)
        pred = dae(features)
        batch_loss = loss(features, pred) # difference between actual and reconstructed   
        training_loss.append(batch_loss.item())
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    avg_loss = np.mean(training_loss)
    print(f"{str(datetime.now())} - DAE Epoch {epoch}: {avg_loss}")
    if epoch % 5 == 4:
        path = os.path.join(os.getcwd(), f'dae_model_backup_{epoch}.pth')
        torch.save(dae.state_dict(), path)

path = os.path.join(os.getcwd(), f'dae_model_backup_final.pth')
torch.save(dae.state_dict(), path)
dae.eval()
final_embeddings = []
for items in train_dl:
    embeddings = dae.encode(items[0])
    final_embeddings.append(embeddings.detach().numpy())

final_embeddings = np.concatenate(final_embeddings)
col_names = pd.read_csv(os.getcwd() + "/featureSet_full_diginetica_2021-05-10_10_21_57.csv", index_col = [0], usecols = [0]) # use appropriate file name
indexes = np.insert(col_names.index.values, 0, 0)
final_embeddings = pd.DataFrame(final_embeddings, index= indexes, columns=list(range(200))) # 200 is the vector size of last RBM layer. Change according to that.
with open("yoochoose_full_embeddings_AE2.dmp", 'wb') as pcf:
        pickle.dump(final_embeddings, pcf)
