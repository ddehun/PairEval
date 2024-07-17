import json
import os
import pickle
import random
from typing import Any, Dict, List

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def report_correlation(output: List[Dict[str, Any]], return_dict: bool = False):
    ans, pred = [e["score"] for e in output], [e["pred"] for e in output]
    spr, pear = (
        get_spearman_correlation(ans, pred)["correlation"],
        get_pearson_correlation(ans, pred)["correlation"],
    )
    if return_dict:
        return {"spearman": spr, "pearson": pear}
    print("Spearoman: ", spr)
    print("Pearson  : ", pear)


def get_spearman_correlation(a: List[float], b: List[float]) -> Dict[str, float]:
    assert len(a) == len(b)
    assert all([isinstance(e, float) for e in a + b])
    res = spearmanr(a, b)
    return {"correlation": round(100 * res.correlation, 2), "pvalue": res.pvalue}


def get_pearson_correlation(a: List[float], b: List[float]) -> Dict[str, float]:
    assert len(a) == len(b)
    assert all([isinstance(e, float) for e in a + b])
    res = pearsonr(a, b)
    return {"correlation": round(100 * res[0], 2), "pvalue": res[1]}


def send_dict_tensors_to_device(
    data: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }


def set_seed(random_seed: int = 42) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def write_jsonl(data: List[Dict[str, Any]], fname: str, overwrite: bool = False):
    assert overwrite or not os.path.exists(fname), fname
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as f:
        for el in data:
            f.write(json.dumps(el) + "\n")


def write_json(data: List[Dict[str, Any]], fname: str, overwrite: bool = False):
    assert overwrite or not os.path.exists(fname)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as f:
        json.dump(data, f)


def read_jsonl(fname: str):
    with open(fname, "r") as f:
        return [json.loads(e) for e in f.readlines()]


def read_json(fname: str):
    with open(fname, "r") as f:
        return json.load(f)


def write_pickle(data: Any, fname: str, overwrite: bool = False):
    assert overwrite or not os.path.exists(fname)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "wb") as f:
        pickle.dump(data, f)
