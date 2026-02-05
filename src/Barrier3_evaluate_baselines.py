import os
import sys
import numpy as np
import pandas as pd

sys.path.append("../src/routerbench")
from embedding.cache import EmbeddingCache
from routers import knn_router, mlp_router, abstract_router

dataset = 'routerBench' # 'openLLMLeaderboard'

test = pd.read_pickle(f'../{dataset}/data/test.pkl')
train = pd.read_pickle(f'../{dataset}/data/train.pkl')

test.columns = (test.columns.str.lower().str.replace("__", "_"))
train.columns = (train.columns.str.lower().str.replace("__", "_"))

if dataset == 'routerBench':
    models = train.keys()[3:].tolist()
elif dataset == 'openLLMLeaderboard':
    models = train.keys()[2:-1].tolist()

models = [m.lower().replace("__", "_") for m in models]
models = sorted(models)
train = train[['prompt'] + models]
test = test[['prompt'] + models]


#baselines
EMB_MODEL = "all-MiniLM-L12-v2" 
LOCAL_CACHE_FILE = f"../temp_embedding_cache/embedding_cache_{EMB_MODEL}.pkl"
N_NEIGHBORS = 40
HIDDEN_LAYERS = [1] #[100,100]
RUNS = 10

for run in range(RUNS):
    print(run)
    
    def _fake_avg_len(self):
        return {m: 100 for m in self.models_to_route}
        
    knn_router.KNNRouter.calculate_average_response_length_per_model = _fake_avg_len
    mlp_router.MLPRouter.calculate_average_response_length_per_model = _fake_avg_len
    
    my_cache = EmbeddingCache(
        connection_string=None,                
        local_cache_path=LOCAL_CACHE_FILE,     
        local_mode=True,                       
    )
    
    knn = knn_router.KNNRouter(
        embedding_model=EMB_MODEL,
        cache=my_cache,
        train_file=train,       
        n_neighbors=N_NEIGHBORS,
        distance_metric="cosine",
        models_to_route=models,
    )

    mlp = mlp_router.MLPRouter(
        embedding_model=EMB_MODEL,
        cache=my_cache,
        train_file=train,
        random_state=run,
        hidden_layer_sizes=HIDDEN_LAYERS,
        models_to_route=models,
    )
    
    test_prompts = test["prompt"].tolist()
    
    predicted_models_knn, knn_perf = knn.batch_route_prompts(test_prompts)
    predicted_models_mlp, mlp_perf = mlp.batch_route_prompts(test_prompts)
    

    M = test[models].fillna(0).to_numpy()
    idx = np.arange(len(test))
    model_to_pos = {m: i for i, m in enumerate(models)}
    
    chosen_idx_knn = np.fromiter((model_to_pos[m] for m in predicted_models_knn), dtype=int)
    chosen_idx_mlp = np.fromiter((model_to_pos[m] for m in predicted_models_mlp), dtype=int)
    
    knn_acc = M[idx, chosen_idx_knn].mean()
    mlp_acc = M[idx, chosen_idx_mlp].mean()
    
    print(f"KNN router accuracy: {knn_acc}")
    print(f"MLP router accuracy: {mlp_acc}")

    np.savez(
        f"../results/baselines/{dataset}/router_results_{run}_{str(HIDDEN_LAYERS)}.npz",
        predicted_models_knn=predicted_models_knn,
        predicted_models_mlp=predicted_models_mlp,
        chosen_idx_knn=chosen_idx_knn,
        chosen_idx_mlp=chosen_idx_mlp,
        knn_acc=knn_acc,
        mlp_acc=mlp_acc
    )