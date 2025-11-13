import torch
import torchvision.transforms as aT
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt


def compare_csv(label,preds):
    labels = pd.read_csv(label)
    predictions = pd.read_csv(preds)
    #replace "_REC_" and "_REC " with "_"
    labels['Fichier'] = labels['Fichier'].str.replace("_REC_", "_").str.replace("_REC ", "_").str.replace("_Rec_", "_")
    predictions['filename'] = predictions['filename'].str.replace("_REC_", "_").str.replace("_REC ", "_").str.replace("_Rec_", "_")

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    cpt = 0
    for i,row in labels.iterrows():

        file = row['Fichier']+".wav"
        
    
        loc = row['Localisation']
        label = 1 if row['Manual_ID']=='Lamantin' else 0 

        sub_preds = predictions[predictions['filename'] == file]

        if len(sub_preds) == 0:
            print(f"No predictions for {file}")
            cpt += 1
            continue
    
        strats = sub_preds['start_time'].values
        # print(loc)
        # print(strats)
        # print()
        # print(min(strats-max(loc-5,0)))
        squared_diffs = (strats+5 - loc) ** 2
        min_index = np.argmin(squared_diffs)
        best_start = strats[min_index]
        

        pred = sub_preds[sub_preds['start_time'] == best_start]

        if pred['id_preds'].values[0] == label:
            if label == 1:
                TP += 1
            else:
                TN += 1
        else:
            if label == 1:
                FN += 1
            else:
                FP += 1
    print(f"Total files with no predictions: {cpt}")   
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")


    
def plot(dataset, idx = None):

    for i in range(10):
        
        
        # img = img.permute(1,2,0)
    
        fig, axs = plt.subplots(3,figsize=(5,10))
        samples = random.sample(range(len(dataset)), k=3)
        for a,idx in zip(axs,samples) :
            
    
            img,label = dataset[idx]
            print(label)
            img = img[0,:,:].squeeze()
            imin, imax = img.min(), img.max()
        
            img = (img - imin) / (imax - imin + 1e-7)
            a.imshow(img,cmap='magma')
            a.set_title(["Pas lamantin","Lamantin"][label])
            a.set_axis_off()
        plt.savefig(f'samples_{i}.png')
    

           
    
