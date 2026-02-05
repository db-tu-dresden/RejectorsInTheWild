import random
import argparse
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import torch
from datasets import Dataset
from sklearn.metrics import roc_curve, auc, accuracy_score
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

models = glob("../helmBenchmark/*.pkl")

task_to_metric = {
    "legalbench":     "stats_quasi_exact_match",               
    "mmlu":           "stats_exact_match",                
    "narrative_qa":   "stats_f1_score",                   
    "commonsense":    "stats_exact_match",               
    "wmt_14":         "stats_bleu_4",                    
    "gsm":            "stats_final_number_exact_match",  
    "math":           "stats_math_equiv_chain_of_thought",
    "med_qa":         "stats_quasi_exact_match",                   
    "natural_qa":     "stats_f1_score"                   
}

model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.squeeze(logits)
    preds  = (torch.sigmoid(torch.tensor(logits).to(device)) >= 0.5).cpu().numpy().astype(int)
    acc    = accuracy_score(labels, preds)
    _, _, roc_auc = compute_roc_auc(logits, labels)
    return {"accuracy": acc, "roc_auc": roc_auc}

def compute_roc_auc(scores: np.ndarray, labels: np.ndarray):
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, tpr, auc(fpr, tpr)

def tokenize_fn(examples):
    return tokenizer(examples["text"],padding="max_length",truncation=True,max_length=512)

def train_model(out_dir, batch_size, lr, epochs, model, tokenizer, train, test, metric):
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        weight_decay=0.1,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        no_cuda=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,       
        compute_metrics=metric,
    )

    trainer.train()
    return trainer

thresholds = {
        "stats_quasi_exact_match":             1,
        "stats_exact_match":                   1,
        "stats_f1_score":                      0.9, #0.1
        "stats_bleu_4":                        0.1, #0.4
        "stats_final_number_exact_match":      1,
        "stats_math_equiv_chain_of_thought":   1
}

def load_data(data, thresholds, cm):
    df = pd.read_pickle(data)

    def decide_correctness(row):
        metric = task_to_metric[row['task']]

        if cm is not None and metric in ['stats_f1_score', 'stats_bleu_4']:
            return row[cm]
        return int(row[metric] >= thresholds[metric])

    df['correctness'] = df.apply(decide_correctness, axis=1)
    return df

def prepare_df(train_df, test_df):
    texts_train  = train_df["request_prompt"].tolist()
    labels_train = train_df["correctness"].astype(float).tolist()
    texts_test   = test_df["request_prompt"].tolist()
    labels_test  = test_df["correctness"].astype(float).tolist()
        
    train_ds = Dataset.from_dict({"text": texts_train, "label": labels_train})
    test_ds  = Dataset.from_dict({"text": texts_test,  "label": labels_test})
        
    train_tok = train_ds.map(tokenize_fn, batched=True)
    test_tok  = test_ds.map(tokenize_fn, batched=True)
    return train_tok, test_tok, labels_test

dfs = []
names = []
for model in tqdm(models): 
    dfs.append(load_data(model, thresholds, 'Een'))  
    #dfs.append(load_data(model, thresholds, None)) use this for other labeling rules like F1:0.1 and BLEU: 0.4
    names.append(model.split('/')[-1].split('.pkl')[0])

data =[]

for i in tqdm(range(len(dfs))): 
    df = dfs[i]
    df_shuf = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = df_shuf.iloc[:10000]
    test_df  = df_shuf.iloc[10000:12000]
    train_tok, test_tok, labels_test = prepare_df(train_df, test_df)
    data.append([names[i], train_tok, test_tok, labels_test])


EPOCHS = 3

for i in range(len(data)):
    
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1
    ).to(device)

    name = data[i][0]
    train_tok = data[i][1]
    test_tok = data[i][2]
    out = f"../results/HelmLiteHeatmap/{name}/" #adjust name according to selected labeling rule
        
    trainer = train_model(out, 32, 2e-5, EPOCHS, model, tokenizer, train_tok, test_tok, compute_metrics)

    for k in range(len(data)):
        compareName = data[k][0]
        compareTokens = data[k][2]
        labels = data[k][3]
        preds = trainer.predict(compareTokens)
        logits = preds.predictions
        
        base = out + compareName
        try:
            os.mkdir(base)
        except: 
            pass
        logitPath = base + '/Shuffle_logits'
        labelPath = base + '/Shuffle_labels'
        np.save(logitPath, logits)
        np.save(labelPath, labels)