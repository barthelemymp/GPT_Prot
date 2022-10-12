import argparse
import numpy as np
import argparse
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
import math
import wandb
from src.ProteinTransformer import *
from src.ProteinsDataset import *
from src.MatchingLoss import *
from src.utils import *

import json
from scipy.stats import spearmanr


def get_params(params):

    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument('--trainset', type=str, help="Path to the dataset for training")
    parser.add_argument('--valset', type=str, help="Path to the dataset for validation")
    parser.add_argument('--save', type=str, default="", help="path to save model, if empty will not save")
    parser.add_argument('--load', type=str, default="", help="path to load model, if empty will not save")
    parser.add_argument('--modelconfig', type=str, default="shallow.config.json", help="hyperparameter")
    parser.add_argument('--outputfile', type=str, default="output.txt", help="file to print scores")
    
    args = parser.parse_args(params)

    return args

def main(params):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Params
    opts = get_params(params)
    f = open(opts.outputfile, "w")
    if opts.save:
        save_model=True
        model_path_save = opts.save
    else:
        save_model = False
    if opts.load:
        load_model=True
        model_path_load = opts.load
    else:
        load_model = False
        
    with open(opts.modelconfig, "r") as read_file:
        print("loading hyperparameter")
        modelconfig = json.load(read_file)
        

    Align=modelconfig["Align"]
    pds_train = ProteinTranslationDataset(opts.trainset, device=device, Align=Align, returnlabel=False)
    pds_val = ProteinTranslationDataset(opts.valset, device=device, Unalign=Unalign, returnlabel=True)
    len_input = pds_train[0].shape[0]

    
    ntrain = len(pds_train)
    nval = len(pds_val)

    # dval1,dval2 = distanceTrainVal(pds_train, pds_val)
    # print("median", (dval1+dval2).min(dim=0)[0].median())
    # maskValclose = (dval1+dval2).min(dim=0)[0]<(dval1+dval2).min(dim=0)[0].median()
    # maskValclose = maskValclose.cpu().numpy()
    # maskValfar = (dval1+dval2).min(dim=0)[0]>=(dval1+dval2).min(dim=0)[0].median()
    # maskValfar = maskValfar.cpu().numpy()
    
    train_iterator = DataLoader(pds_train, batch_size=modelconfig["batch_size"],
                    shuffle=False, num_workers=0, collate_fn=default_collate)
    val_iterator = DataLoader(pds_val, batch_size=modelconfig["batch_size"],
                    shuffle=False, num_workers=0, collate_fn=default_collate)
    
    src_pad_idx =pds_train.padIndex

    trg_position_embedding = PositionalEncoding(modelconfig["embedding_size"], max_len=len_input, device=device)
            
           
    model = Transformer(
        modelconfig["embedding_size"],
        modelconfig["trg_vocab_size"],
        pad_idx,
        modelconfig["num_heads"],
        modelconfig["num_decoder_layers"],
        modelconfig["forward_expansion"],
        modelconfig["dropout"],
        trg_position_embedding,
        device,
    ).to(device)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
 
    optimizer = optim.AdamW(model.parameters(), lr=modelconfig["learning_rate"], weight_decay=modelconfig["wd"])

    if load_model:
        load_checkpoint(torch.load(model_path_load), model, optimizer)
    
    
    criterion = nn.CrossEntropyLoss(ignore_index=pds_train.padIndex)
    criterion_raw = nn.CrossEntropyLoss(reduction='none')
    num_epochs = modelconfig["num_epochs"]
    for epoch in range(modelconfig["num_epochs"]+1):
        print(f"[Epoch {epoch} / {num_epochs}]")
        model.train()
        lossesCE = []
        for batch_idx, batch in enumerate(train_iterator):
            optimizer.zero_grad()
            loss = LLLoss(batch, model, criterion)
            lossesCE.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
            optimizer.step()
                
            
        mean_lossCETrain = sum(lossesCE) / len(lossesCE)

        model.eval()
        lossesCE_eval = []
        lossesMatching_eval = []
        accuracyVal = 0
        mes = []

        with  torch.no_grad():
            for batch_idx, batch in enumerate(val_iterator):
                target, me = batch[0], batch[1]
                bs = target.shape[1]
                # target, me= batch[0], batch[1]
                # target = target.to(device)
                # output = model(target[:-1, :])
                # # accuracyVal += accuracy(batch, output, onehot=False).item()
                # output = output.reshape(-1, output.shape[2]) #keep last dimension
                # targets_Original= target
                # targets_Original = targets_Original[1:].reshape(-1)
                # loss_eval = criterion(output, targets_Original)
                loss_full = LLLoss(target, model, criterion_raw).reshape(-1,bs).mean(dim=0)
                for x, xme in zip(loss_full, me):
                    lossesCE_eval.append(x.item())
                    mes.append(xme.item())
            spearmann = spearmanr(lossesCE_eval, mes)


        out = "epoch: "+str(epoch)+", Train loss CE: " + str(mean_lossCETrain) +  ", spearmann: "+ str(spearmann)
        
        # if epoch%200==0:
        #     model.eval()
        #     criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.padIndex, reduction='none')
        #     scoreHungarianVal = HungarianMatchingBS(pds_val, model, 100)
        #     scoHVal = scipy.optimize.linear_sum_assignment(scoreHungarianVal)
        #     scoreMatchingVal = sum(scoHVal[0]==scoHVal[1])
        #     scoreMatchingValClose = sum((scoHVal[0]==scoHVal[1])[maskValclose])
        #     scoreMatchingValFar = sum((scoHVal[0]==scoHVal[1])[maskValfar])
        #     out+= ", scoreMatching Val :" +str(scoreMatchingVal)+", scoreMatchingValClose: " +str(scoreMatchingValClose)+", scoreMatchingVal Far: "+ str(scoreMatchingValFar)
        #     if save_model:
        #         checkpoint = {
        #             "state_dict": model.state_dict(),
        #             "optimizer": optimizer.state_dict(),
        #         }
        #         save_checkpoint(checkpoint, filename=model_path_save)
        # out+="\n"
        f.write(out)
    print("End Training")
    f.close()
    
    

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])