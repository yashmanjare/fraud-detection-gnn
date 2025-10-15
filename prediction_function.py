# prediction_function.py
import os
import io
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
import torch.nn.functional as F

# Optional import for torch_geometric — only required if GNN model is used.
try:
    from torch_geometric.data import Data
except Exception:
    Data = None  # we'll check before using

# ---------- Helpers ----------
def _standardize_df(df, numeric_cols=None):
    """Return scaled numpy array and column names."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(0.).astype(float).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, numeric_cols, scaler

def _build_knn_graph(X, k=5):
    """Return edge_index for torch_geometric from k-NN graph on X (n_samples x n_features)."""
    nbrs = NearestNeighbors(n_neighbors=min(k+1, X.shape[0]), algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    # build edges: for each i, connect to indices[i, 1:]
    src = []
    dst = []
    for i in range(indices.shape[0]):
        for j in range(1, indices.shape[1]):  # skip self at position 0
            src.append(i)
            dst.append(indices[i, j])
    # convert to torch tensor edge_index shape [2, num_edges]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)  # make undirected by duplicating
    return edge_index

def _try_load_model(model_path, map_location='cpu'):
    """Try loading a model in a flexible way:
       - Try torch.jit.load (scripted model)
       - Try torch.load, and if it's an nn.Module return it, if state_dict return that
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 1) Try torch.jit.load
    try:
        scripted = torch.jit.load(model_path, map_location=map_location)
        scripted.eval()
        return scripted
    except Exception:
        pass

    # 2) Try torch.load
    loaded = torch.load(model_path, map_location=map_location)
    # If the object is a Module (saved with torch.save(model)), return it
    if isinstance(loaded, torch.nn.Module):
        loaded.eval()
        return loaded
    # Otherwise the loaded object may be a state_dict
    if isinstance(loaded, dict):
        # Heuristic: check if keys look like state_dict keys
        some_keys = list(loaded.keys())[:5]
        if all(isinstance(k, str) for k in some_keys):
            # It's likely a state_dict; return the dict so the caller can handle it
            return loaded
    # Otherwise return what we got
    return loaded

# ---------- Main prediction function ----------
def predict_from_dataframe(df,
                           model_path=None,
                           use_gnn=True,
                           knn_k=5,
                           threshold=0.5,
                           device='cpu'):
    """
    Predict fraud on a dataframe of transactions.

    Args:
        df (pd.DataFrame): Input transaction dataframe (columns should be numeric features used by model).
        model_path (str or None): Path to .pth/.pt model file. 
        use_gnn (bool): Whether to attempt GNN prediction. 
        knn_k (int): number of neighbors for graph construction when using GNN.
        threshold (float 0..1): probability threshold to label as 'Fraud'.
        device (str): 'cpu' or 'cuda' (if available and model supports it).

    Returns:
        pd.DataFrame: original df with added columns:
            - Fraud_Probability: [0..1] (higher => more likely fraud)
            - Prediction_Label: 'Fraud'/'Not Fraud'
    """
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty")

    # 1) pick numeric columns (user data may have non-numeric like IDs/time; keep numeric features)
    X_scaled, numeric_cols, scaler = _standardize_df(df)
    n_nodes = X_scaled.shape[0]

    # 2) Try to load model if requested
    model = None
    state_dict = None
    if use_gnn and model_path is not None:
        try:
            loaded = _try_load_model(model_path, map_location=device)
            if isinstance(loaded, dict):
                # state_dict: We cannot reconstruct model architecture automatically
                state_dict = loaded
                model = None
            elif isinstance(loaded, torch.nn.Module):
                model = loaded.to(device)
            else:
                # unknown object loaded (e.g., a dict with more metadata)
                # try to find a model inside
                if hasattr(loaded, 'state_dict'):
                    try:
                        loaded.eval()
                        model = loaded.to(device)
                    except Exception:
                        model = None
        except Exception as e:
            model = None

    # 3) If we have a GNN model and torch_geometric available, construct graph and run inference
    probs = None
    used_method = None
    if model is not None and Data is not None:
        try:
            # Build kNN graph
            edge_index = _build_knn_graph(X_scaled, k=knn_k).to(device)

            # Construct node features: use scaled features as torch.float
            x = torch.tensor(X_scaled, dtype=torch.float32, device=device)

            # Create PyG Data object
            pyg_data = Data(x=x, edge_index=edge_index)

            model.eval()
            with torch.no_grad():
                # Support models that accept either (x, edge_index) or a single data object
                try:
                    out = model(pyg_data.x, pyg_data.edge_index)
                except TypeError:
                    try:
                        out = model(pyg_data)
                    except Exception as e:
                        raise RuntimeError(f"Model call failed: {e}")

                # If output has shape (N, C), convert to probability of class 1
                if out is None:
                    raise RuntimeError("Model returned None")
                out_cpu = out.cpu()
                if out_cpu.dim() == 2 and out_cpu.size(1) >= 2:
                    # assume binary classification; take softmax prob for class 1
                    probs_tensor = F.softmax(out_cpu, dim=1)[:, 1]
                    probs = probs_tensor.numpy()
                else:
                    # single-output regression/logit - apply sigmoid
                    probs = torch.sigmoid(out_cpu.view(-1)).numpy()

            used_method = 'gnn'
        except Exception as e:
            # If anything fails during GNN inference, we'll fallback
            print(f"[predict_from_dataframe] GNN inference failed: {e}")
            probs = None

    # 4) If GNN wasn't usable, fallback to IsolationForest unsupervised anomaly scoring
    if probs is None:
        # IsolationForest expects 2D numeric input; we already have X_scaled
        iso = IsolationForest(n_estimators=200, contamination='auto', random_state=42)
        iso.fit(X_scaled)
        # anomaly score: lower = more abnormal. `score_samples` gives higher = more normal.
        scores = iso.score_samples(X_scaled)  # higher => normal; lower => anomalous
        # convert to 0..1 "fraud probability" by flipping and min-max scaling
        scores_min = scores.min()
        scores_max = scores.max() if scores.max() != scores.min() else scores_min + 1e-6
        norm = (scores_max - scores) / (scores_max - scores_min)
        # clamp
        norm = np.clip(norm, 0.0, 1.0)
        probs = norm
        used_method = 'isolation_forest'

    # 5) Form output dataframe
    out_df = df.copy().reset_index(drop=True)
    out_df['Fraud_Probability'] = np.round(probs.astype(float), 6)
    out_df['Prediction_Label'] = np.where(out_df['Fraud_Probability'] >= float(threshold), 'Fraud', 'Not Fraud')
    out_df['Model_Method'] = used_method

    return out_df, {'used_method': used_method, 'numeric_cols': numeric_cols}
