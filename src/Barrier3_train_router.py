import numpy as np
import pandas as pd

import sys
sys.path.append('../src')
import helper

bank = 0.05
RUNS = 10
dataset = 'routerBench' #openLLMLeaderboard

for run in range(RUNS):
    out =  f'../results/{dataset}/{run}/rejectors/'
    df_train = pd.read_pickle(f'../{dataset}/data/train.pkl')
    helper.train_router(out, df_train, bank)