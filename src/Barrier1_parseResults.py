import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import sys
sys.path.append('../src')
import helper

files = glob('../results/HelmLiteRejectors/*/*/*/*.npy')

def get_logits_and_labels(score_path):
    logits = np.load(score_path)
    parts  = score_path.split('/')
    model, lr = parts[3], parts[4]
    labels = np.load(f'../results/HelmLiteRejectors/{model}/{lr}/labels.npy')
    return logits, labels

records = []
for path in tqdm(files):
    parts = path.split('/')
    model = parts[3]
    f1, bleu, similarity = None, None, None    
    try:
        f1, bleu = parts[4].split('_')
    except:
        similarity = parts[4]
    run     = parts[5]
    epoch   = parts[6].split('_')[-1].split('.')[0]

    try:
        logits, labels = get_logits_and_labels(path)
        m = helper.compute_eval_metrics(logits, labels)
    except:
        continue
    records.append([
        model, f1, bleu, similarity, run, epoch,
        m['auroc'], m['aupr'], m['brier'], m['accuracy_0.5'],
        m['coverage_curve'], m['accuracy_curve']
    ])

df = pd.DataFrame(records, columns=[
    'model','f1','bleu', 'similarity', 'run','epoch',
    'auroc','aupr','brier','acc_0.5',
    'coverage','accuracy_curve'
])

df.to_pickle('../results/HelmLiteRejectors/results.pkl')

df = df.drop(columns=['aupr', 'brier', 'acc_0.5', 'coverage', 'accuracy_curve'])
df.to_pickle('../results/HelmLiteRejectors/results_compressed.pkl')