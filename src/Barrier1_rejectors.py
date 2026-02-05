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

def prepare_data(data, train_size, test_size, thresholds, cm):
    df = pd.read_pickle(data)

    #df['correctness'] = df.apply(lambda row: int(row[task_to_metric[row['task']]] >= thresholds[task_to_metric[row['task']]]),axis=1)
    def decide_correctness(row):
        metric = task_to_metric[row['task']]

        if cm is not None and metric in ['stats_f1_score', 'stats_bleu_4']:
            return row[cm]
        return int(row[metric] >= thresholds[metric])

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

class LogitsSaverCallback(TrainerCallback):
    def __init__(self, test_dataset, output_dir):
        self.test_dataset = test_dataset
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        preds = self.trainer.predict(self.test_dataset)
        logits = preds.predictions
        
        epoch = int(state.epoch or 0)
        fname = os.path.join(self.output_dir, f"logits_epoch_{epoch}.npy")
        np.save(fname, logits)

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

    logits_callback = LogitsSaverCallback(test_dataset=test, output_dir=out_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,       
        compute_metrics=metric,
        callbacks=[logits_callback],
    )
    
    logits_callback.set_trainer(trainer)

    trainer.train()

def multi_run_model(selected_data, train_size, test_size, RUNS, tokenizer, EPOCHS, f1, bleu, thresholds, cm):

    train_tok, test_tok, labels_test = prepare_data(selected_data, train_size, test_size, thresholds, cm)
    name = selected_data.split('/')[-1].split('.pkl')[0]
    
    for run in tqdm(range(RUNS)):
        if cm:
            out = f"../results/HelmLiteRejectors/{name}/{cm}/{run}"
        else:
            out = f"../results/HelmLiteRejectors/{name}/{str(f1)}_{str(bleu)}/{run}"
        
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1
        ).to(device)

        train_model(out, 32, 2e-5, EPOCHS, model, tokenizer, train_tok, test_tok, compute_metrics)

    if cm:
        np.save(f'../results/HelmLiteRejectors/{name}/{cm}/labels.npy', labels_test)
    else:
        np.save(f'../results/HelmLiteRejectors/{name}/{str(f1)}_{str(bleu)}/labels.npy', labels_test)

def main():
    parser = argparse.ArgumentParser(description="Process two required args (model data border1 / border2) and two optional args (f1 and bleu threshold).")
    parser.add_argument("arg1", type=int, help="First border")
    parser.add_argument("arg2", type=int, help="Second border")
    parser.add_argument("--f1", type=float, default=0.6, help="f1 threshold (default: 0.6)")
    parser.add_argument("--bleu", type=float, default=0.2, help="bleu threshold (default: 0.2)")
    parser.add_argument("--cm", type=str, choices=["Vee","Eee","Ven","Een"],help="alternative correctness metrics (Vee, Eee, Ven, Een)")
    args = parser.parse_args()

    border1 = args.arg1
    border2 = args.arg2
    f1_threshold = args.f1
    bleu_threshold = args.bleu

    cm = args.cm

    thresholds = {
        "stats_quasi_exact_match":             1,
        "stats_exact_match":                   1,
        "stats_f1_score":                      f1_threshold,
        "stats_bleu_4":                        bleu_threshold,
        "stats_final_number_exact_match":      1,
        "stats_math_equiv_chain_of_thought":   1
    }

    print(border1, border2, f1_threshold, bleu_threshold, cm)
        
    for i in models[border1:border2]:
        selected_data = i
        multi_run_model(selected_data, 10000, 2000, 10, tokenizer, 10, f1_threshold, bleu_threshold, thresholds, cm)

if __name__ == "__main__":
    main()