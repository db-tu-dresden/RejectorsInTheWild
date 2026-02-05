import os
import numpy as np
import pandas as pd
from glob import glob 
from tqdm import tqdm

import sys
sys.path.append('../src')
import helper

dataset = 'routerBench' #'openLLMLeaderboard'
test = pd.read_pickle(f'../{dataset}/data/test.pkl')

RUNS = 10
for run in tqdm(range(RUNS)):
    rejectors = glob(f'../results/{dataset}/{run}/rejectors/*')
    rejectors = [f for f in rejectors if not f.endswith(".pkl")] #filters bank.pkl
    bank = pd.read_pickle(f'../results/{dataset}/{run}/rejectors/bank.pkl')
    for rejector in tqdm(rejectors):
        rejector_name = rejector.split('/')[-1]
        trainer = helper.prepare_rejector_for_testing(rejector)

        os.makedirs(f'../results/{dataset}/{run}/predictions/', exist_ok=True)
        os.makedirs(f'../results/{dataset}/{run}/bank/', exist_ok=True)
        
        #collect test predictions
        logits = helper.test_model(trainer, test)
        logitPath = f'../results/{dataset}/{run}/predictions/{rejector_name}'
        np.save(logitPath, logits)

        #collect bank predictions
        logits = helper.test_model(trainer, bank)
        logitPath = f'../results/{dataset}/{run}/bank/{rejector_name}'
        np.save(logitPath, logits)

        
        