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


# ---------------------------------------------------------------- other fingerprints
# Generic hashed-fingerprint wrapper over rdFingerprintGenerator (the modern rdkit API),
# plus MACCS keys. Each is a fixed molecular descriptor (no learning) — the classic
# baselines to sit ECFP4 / ChemBERTa / UniMol against.
class _FPGen:
    """SMILES -> fixed-length bit fingerprint via a named rdFingerprintGenerator."""

    def __init__(self, name, kind, dim, radius=3):
        self.name, self.kind, self.dim, self.radius = name, kind, dim, radius
        self._gen = None

    def _generator(self):
        if self._gen is None:
            from rdkit.Chem import rdFingerprintGenerator as G
            if self.kind == "morgan":
                self._gen = G.GetMorganGenerator(radius=self.radius, fpSize=self.dim)
            elif self.kind == "atompair":
                self._gen = G.GetAtomPairGenerator(fpSize=self.dim)
            elif self.kind == "topotorsion":
                self._gen = G.GetTopologicalTorsionGenerator(fpSize=self.dim)
            elif self.kind == "rdkit":
                self._gen = G.GetRDKitFPGenerator(fpSize=self.dim)
            else:
                raise ValueError(self.kind)
        return self._gen

    def __call__(self, smiles_list):
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        gen = self._generator()
        out, ok = [], []
        for s in smiles_list:
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            if m is None:
                out.append(np.zeros(self.dim, np.float32)); ok.append(False); continue
            arr = np.zeros(self.dim, np.int8)
            Chem.DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m), arr)
            out.append(arr.astype(np.float32)); ok.append(True)
        return np.vstack(out), np.array(ok)


class _MACCS:
    dim = 167
    name = "maccs"

    def __call__(self, smiles_list):
        from rdkit import Chem
        from rdkit.Chem import MACCSkeys
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        out, ok = [], []
        for s in smiles_list:
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            if m is None:
                out.append(np.zeros(self.dim, np.float32)); ok.append(False); continue
            arr = np.zeros(self.dim, np.int8)
            Chem.DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), arr)
            out.append(arr.astype(np.float32)); ok.append(True)
        return np.vstack(out), np.array(ok)


@register("ecfp6")
def _mk_ecfp6():
    return _FPGen("ecfp6", "morgan", 2048, radius=3)


@register("atompair")
def _mk_atompair():
    return _FPGen("atompair", "atompair", 2048)


@register("topotorsion")
def _mk_topotorsion():
    return _FPGen("topotorsion", "topotorsion", 2048)


@register("rdkit_fp")
def _mk_rdkit_fp():
    return _FPGen("rdkit_fp", "rdkit", 2048)


@register("maccs")
def _mk_maccs():
    return _MACCS()


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
        # small batch: the 84M v2 attention blows past 32 GB at the default batch_size=32
        self.rep = UniMolRepr(data_type="molecule", model_name=model_name,
                              remove_hs=False, batch_size=8)
        self.name = f"unimol_v{version}"
        self.dim = None  # inferred on first call (v1 CLS=512; v2 differs)

    def __call__(self, smiles_list):
        # rdkit-validate first so the returned CLS reprs stay index-aligned to the input
        # (unimol_tools 0.1.6 returns a list of per-mol CLS ndarrays; older builds a dict).
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        smis = [str(s) for s in smiles_list]
        valid = [i for i, s in enumerate(smis) if Chem.MolFromSmiles(s) is not None]
        ok = np.zeros(len(smis), bool)
        if not valid:
            return np.zeros((len(smis), self.dim or 512), np.float32), ok
        r = self.rep.get_repr([smis[i] for i in valid], return_atomic_reprs=False)
        reps = r["cls_repr"] if isinstance(r, dict) else r
        reps = [np.asarray(x, np.float32).ravel() for x in reps]
        if len(reps) != len(valid):
            raise RuntimeError(f"UniMol returned {len(reps)} reprs for {len(valid)} valid mols")
        self.dim = reps[0].shape[0]
        out = np.zeros((len(smis), self.dim), np.float32)
        for j, i in enumerate(valid):
            out[i] = reps[j]; ok[i] = True
        return out, ok


@register("unimol_v1")
def _mk_unimol_v1():
    return _UniMol(1)


@register("unimol_v2")
def _mk_unimol_v2():
    return _UniMol(2)
