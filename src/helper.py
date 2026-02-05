import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, average_precision_score, brier_score_loss
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def prepare_df(df):
    tokens  = df["prompt"].tolist()
    labels = df["correctness"].astype(float).tolist()
    print(len(tokens))
        
    ds = Dataset.from_dict({"text": tokens, "label": labels})
        
    tokens = ds.map(tokenize_fn, batched=True)
    return tokens, labels

def train_model(out_dir, batch_size, lr, epochs, model, tokenizer, train, metric):
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        weight_decay=0.1,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        disable_tqdm=True,
        fp16=torch.cuda.is_available(),
        no_cuda=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        tokenizer=tokenizer,       
        compute_metrics=metric,
    )

    trainer.train()
    return trainer

def coverage_accuracy_curve(scores, labels):
    idx = np.argsort(-scores)
    sorted_labels = labels[idx]
    
    cum_preds = np.arange(1, len(scores)+1) # [1,2,3,…,n]
    cum_correct = np.cumsum(sorted_labels)  # how many true positives so far

    coverages  = cum_preds / len(scores) #duplicates?
    accuracies = cum_correct / cum_preds
    return coverages, accuracies

def compute_eval_metrics(logits, labels):
    scores = 1 / (1 + np.exp(-logits)) #like torch.sigmoid(torch.tensor(logits)).numpy()
    scores = np.concatenate(scores)

    _, _, auroc = compute_roc_auc(scores, labels)
    aupr  = average_precision_score(labels, scores)
    brier = brier_score_loss(labels, scores)

    preds_05 = (scores >= 0.5).astype(int)
    acc_05   = accuracy_score(labels, preds_05)

    coverages, accuracies = coverage_accuracy_curve(scores, labels)

    return {
        'auroc'           : auroc,
        'aupr'            : aupr,
        'brier'           : brier,
        'accuracy_0.5'    : acc_05,
        'coverage_curve'  : np.array(coverages),
        'accuracy_curve'  : np.array(accuracies),
    }

def test_model(trainer, data):
    inputs  = data["prompt"].tolist()
    ds = Dataset.from_dict({"text": inputs})
    tok = ds.map(tokenize_fn, batched=True)

    preds = trainer.predict(tok)
    logits = preds.predictions

    return logits

def train_router(out, df, bank_size=0, EPOCHS=3):
    model_names = [c for c in df.columns if c not in ['sample_id', 'eval_name', 'prompt']]

    os.makedirs(os.path.dirname(out), exist_ok=True)

    if bank_size > 0:
        nan_rate = df.isna().mean(axis=1)
        sorted_rows = df.loc[nan_rate.sort_values().index]
        n = len(sorted_rows)
        split_point = int(n * bank_size) 
        
        bank = sorted_rows.iloc[:split_point] #bank should have as few NaN values as possible
        df  = sorted_rows.iloc[split_point:]
        bank.to_pickle(f'{out}bank.pkl')
        
    for model in tqdm(model_names):
        temp = df[['prompt', model]]
        temp = temp.rename(columns={model: 'correctness'})
        temp = temp.dropna(subset=['correctness'])

        out_ = f'{out}{model}'

        train_rejector(out_, temp, EPOCHS)
        print(f'rejector for model: "{model}" was trained')

def train_rejector(out, df_train, EPOCHS):
    tokens, labels = prepare_df(df_train)
    
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1
    ).to(device)

    trainer = train_model(out, 32, 2e-5, EPOCHS, model, tokenizer, tokens, compute_metrics)
    trainer.save_model(out)

def prepare_rejector_for_testing(path):
    rejector = DistilBertForSequenceClassification.from_pretrained(
            path,
            num_labels=1
    ).to(device)
    
    eval_args = TrainingArguments(
        output_dir="./",
        per_device_eval_batch_size=32,
    )
    
    eval_Trainer = Trainer(
        model=rejector,
        tokenizer=tokenizer,
        args=eval_args
    )
    return eval_Trainer