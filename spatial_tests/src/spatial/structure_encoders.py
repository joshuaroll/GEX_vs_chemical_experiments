"""Pluggable molecular structure encoders: SMILES -> embedding vector.

Registry so the structure arm can swap encoders without touching the comparison:
    get_encoder("ecfp4" | "chemberta" | "unimol_v1" | "unimol_v2")(smiles_list) -> [n, d]

- ecfp4      : 2048-bit Morgan r=2 (rdkit). Always available; the fixed-descriptor baseline.
- chemberta  : DeepChem/ChemBERTa-77M-MLM mean-pooled token states (transformers). Learned, ~384-d.
- unimol_v1  : Uni-Mol 3D molecular representation (needs `unimol_tools`).
- unimol_v2  : Uni-Mol2 (needs `unimol_tools` >= the v2 release).

UniMol backends import `unimol_tools` lazily and raise a clear install hint if absent,
so the registry stays swappable even when the package is not installed.
"""
from __future__ import annotations
import numpy as np

_REGISTRY = {}


def register(name):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def available():
    return sorted(_REGISTRY)


def get_encoder(name):
    if name not in _REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; available: {available()}")
    return _REGISTRY[name]()


# ---------------------------------------------------------------- ECFP4
class _ECFP4:
    dim = 2048
    name = "ecfp4"

    def __call__(self, smiles_list):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        out, ok = [], []
        for s in smiles_list:
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            if m is None:
                out.append(np.zeros(self.dim, np.float32)); ok.append(False); continue
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=self.dim)
            arr = np.zeros(self.dim, np.int8); Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            out.append(arr.astype(np.float32)); ok.append(True)
        return np.vstack(out), np.array(ok)


@register("ecfp4")
def _mk_ecfp4():
    return _ECFP4()


# ---------------------------------------------------------------- ChemBERTa
class _ChemBERTa:
    name = "chemberta"
    model_id = "DeepChem/ChemBERTa-77M-MLM"

    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.mdl = AutoModel.from_pretrained(self.model_id).to(self.dev).eval()
        self.dim = self.mdl.config.hidden_size

    def __call__(self, smiles_list, batch=64):
        t = self.torch
        embs, ok = [], []
        for i in range(0, len(smiles_list), batch):
            chunk = [str(s) for s in smiles_list[i:i + batch]]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.dev)
            with t.no_grad():
                h = self.mdl(**enc).last_hidden_state  # [b, L, d]
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)  # mean-pool over real tokens
            embs.append(pooled.cpu().numpy())
            ok.extend([True] * len(chunk))
        return np.vstack(embs).astype(np.float32), np.array(ok)


@register("chemberta")
def _mk_chemberta():
    return _ChemBERTa()


# ---------------------------------------------------------------- UniMol v1 / v2
class _UniMol:
    def __init__(self, version):
        try:
            from unimol_tools import UniMolRepr
        except Exception as e:
            raise ImportError(
                f"UniMol {version} needs `unimol_tools` (pip install unimol_tools). "
                f"Registered and swappable; install to activate. Original error: {e}")
        model_name = "unimolv1" if version == 1 else "unimolv2"
        self.rep = UniMolRepr(data_type="molecule", model_name=model_name, remove_hs=False)
        self.name = f"unimol_v{version}"

    def __call__(self, smiles_list):
        r = self.rep.get_repr([str(s) for s in smiles_list], return_atomic_reprs=False)
        emb = np.asarray(r["cls_repr"], np.float32)  # [n, d]
        return emb, np.ones(len(smiles_list), bool)


@register("unimol_v1")
def _mk_unimol_v1():
    return _UniMol(1)


@register("unimol_v2")
def _mk_unimol_v2():
    return _UniMol(2)
