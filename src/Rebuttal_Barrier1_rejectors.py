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
import ast
import re
import string
from collections import Counter
import sacrebleu

def extract_ref(x):
    if isinstance(x, str):
        x = ast.literal_eval(x)
    return x[0]['output']['text']


def normalize_text(s):
    s = str(s).lower()
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ' '.join(s.split())
    return s


def token_f1(pred, ref):
    pred_tokens = normalize_text(pred).split()
    ref_tokens = normalize_text(ref).split()

    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

models = glob("../helmBenchmark/*.pkl")
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

def prepare_data(data):
    df = pd.read_pickle(data)
    df['bleu_complete'] = np.nan
    df['f1_complete'] = np.nan
    
    mask = df['predicted_text'].notna()
    tmp = df.loc[mask].copy()
    tmp['ref'] = tmp['references'].apply(extract_ref)
    
    tmp['bleu_complete'] = [
        sacrebleu.sentence_bleu(pred, [ref]).score
        for pred, ref in zip(tmp['predicted_text'], tmp['ref'])
    ]
    
    tmp['f1_complete'] = [
        token_f1(pred, ref)
        for pred, ref in zip(tmp['predicted_text'], tmp['ref'])
    ]
    
    df['bleu_complete'] = tmp['bleu_complete']
    df['bleu_complete'] = df['bleu_complete']/100
    df['f1_complete'] = tmp['f1_complete']

    bleu_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4]
    f1_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    combs = []
    
    for tb in bleu_thresholds:
        for tf in f1_thresholds:
            or_col = ((df["bleu_complete"] >= tb) | (df["f1_complete"] >= tf)).astype(int)
            and_col = ((df["bleu_complete"] >= tb) & (df["f1_complete"] >= tf)).astype(int)
            rule_name_or = 'or_' + str(tb) + '_' + str(tf)
            rule_name_and = 'and_' + str(tb) + '_' + str(tf)
            df[rule_name_or] = or_col
            df[rule_name_and] = and_col
            combs.append(rule_name_or)
            combs.append(rule_name_and)

    return df, combs
    
def compute_correctness(df, train_size, test_size, task_to_metric):
    def decide_correctness(row):
        metric = task_to_metric[row['task']]
        return int(row[metric])

    df['correctness'] = df.apply(decide_correctness, axis=1)
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = df.iloc[:train_size]
    test_df  = df.iloc[train_size:train_size+test_size]
    
    texts_train  = train_df["request_prompt"].tolist()
    labels_train = train_df["correctness"].astype(float).tolist()
    texts_test   = test_df["request_prompt"].tolist()
    labels_test  = test_df["correctness"].astype(float).tolist()
    
    train_ds = Dataset.from_dict({"text": texts_train, "label": labels_train})
    test_ds  = Dataset.from_dict({"text": texts_test,  "label": labels_test})
    
    train_tok = train_ds.map(tokenize_fn, batched=True)
    test_tok  = test_ds.map(tokenize_fn, batched=True)
    return train_tok, test_tok, labels_test
    
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

    preds = trainer.predict(test)
    logits = preds.predictions
    fname = os.path.join(out_dir, f"logits.npy")
    np.save(fname, logits)
    
def main():
    for i in range(10):
        model = models[i]
        df, combs = prepare_data(model)
        
        for comb in combs:
            print(comb)
            task_to_metric = {
                "legalbench":     "stats_quasi_exact_match",               
                "mmlu":           "stats_exact_match",                
                "narrative_qa":   comb,                   
                "commonsense":    "stats_exact_match",               
                "wmt_14":         comb,                    
                "gsm":            "stats_final_number_exact_match",  
                "math":           "stats_math_equiv_chain_of_thought",
                "med_qa":         "stats_quasi_exact_match",                   
                "natural_qa":     comb
            }
            
            train_tok, test_tok, labels_test = compute_correctness(df, 10000, 2000, task_to_metric)
    
            model = DistilBertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1
            ).to(device)
    
            out = f'../rebuttal/model{i}/{comb}'
            
            train_model(out, 32, 2e-5, 3, model, tokenizer, train_tok, test_tok, compute_metrics)
    
            np.save(f'../rebuttal/model{i}/{comb}/labels.npy', labels_test)

if __name__ == "__main__":
    main()