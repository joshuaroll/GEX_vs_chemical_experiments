#!/usr/bin/env python
"""Extract the MultiDCP-CheMoE latent (global_features, 306-d) per drug.

global_features = cat([drug_embed(128), cell_hidden(50), dose_embed(128)]) is the
model's internal representation BEFORE the expert/gene-output stage. It is the
input to model.gating_network, so a forward-pre-hook captures it exactly without
editing the frozen model.

Reuses gate4a's proven upstream model build + the production basal pipeline.
Default checkpoint: the canonical frozen best_model.pt (transformer encoder).

Validation main confirms: extraction works, and (architecture claim) the drug
block varies across drugs while the cell_hidden + dose blocks are constant within
one fixed basal context.
"""
from __future__ import annotations
import argparse, os, sys, pathlib, tempfile
import numpy as np

REPO = pathlib.Path("/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests")
MDCP_SRC = "/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src"
CONFIG_PATH = REPO / "configs" / "liver_p2.yaml"
CANONICAL_CKPT = "/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/best_model.pt"
DRUG_DIM, CELL_DIM, DOSE_DIM = 128, 50, 128  # global_features = 306


class MultiDCPLatentExtractor:
    """Loads the frozen model once; extract_global_features(smiles) -> [306]."""

    def __init__(self, gpu="2", checkpoint=CANONICAL_CKPT, linear_encoder=False):
        sys.path.insert(0, str(REPO))
        sys.path.insert(0, os.path.join(MDCP_SRC, "models"))
        sys.path.insert(0, os.path.join(MDCP_SRC, "utils"))
        import yaml, torch
        from scripts.cache_region_de import _resolve_device
        from multidcp_ae_pdg_utils import initialize_model_registry
        import multidcp_chemoe_pdg as mc
        from data_utils_pdg import convert_smile_to_feature, create_mask_feature

        self.torch = torch
        self.device = torch.device(_resolve_device(gpu))
        cfg = yaml.safe_load(open(CONFIG_PATH))
        self.norm_lo, self.norm_hi = cfg["basal_normalization"]["manifold_range"]
        self.gene_order = (REPO / cfg["gene_order_file"]).read_text().split()
        self.N_PDG = len(self.gene_order)
        self._smi2feat = convert_smile_to_feature
        self._mask = create_mask_feature

        reg = initialize_model_registry()
        reg.update({"num_gene": self.N_PDG, "pert_idose_input_dim": 2,
                    "dropout": 0.3, "linear_encoder_flag": linear_encoder})
        model = mc.MultiDCP_CheMoE_AE(device=self.device, model_param_registry=reg)
        model.to(self.device).double()
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        res = model.load_state_dict(state, strict=False)
        self.strict_ok = (len(res.missing_keys) == 0 and len(res.unexpected_keys) == 0)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self._gene_t = torch.arange(self.N_PDG, device=self.device)

        # forward-pre-hook on gating_network captures global_features (its input)
        self._gf = {}
        def _hook(_m, args):
            self._gf["v"] = args[0].detach().float().cpu().numpy()
        model.model.gating_network.register_forward_pre_hook(_hook)

    def set_basal(self, basal_np):
        self._basal = self.torch.as_tensor(
            basal_np, dtype=self.torch.float64, device=self.device).unsqueeze(0)

    def extract(self, smiles):
        return self.extract_full(smiles)[0]  # [306] global_features (back-compat)

    def extract_full(self, smiles):
        """One forward -> (global_features[306], treated_pred[N_PDG]).

        The gating-network hook captures global_features (its input); the model's
        return value is the absolute predicted treated GEX over N_PDG genes. Both
        come off the SAME forward, so the latent and the gene-output are read from
        one identical computation in one identical basal.
        """
        t = self.torch
        drug = self._smi2feat([smiles], self.device)
        mask = self._mask(drug, self.device)
        dose = t.tensor([[1.0, 0.0]], dtype=t.float64, device=self.device)
        with t.no_grad():
            out = self.model(input_cell_gex=self._basal, input_drug=drug, input_gene=self._gene_t,
                             mask=mask, input_pert_idose=dose, job_id="perturbed", epoch=0)
        pred = out[0] if isinstance(out, (tuple, list)) else out  # (pred, cell_hidden)
        return self._gf["v"][0].copy(), pred.squeeze(0).float().cpu().numpy()  # [306], [N_PDG]


def load_reference_basal(ext):
    """One fixed liver basal context (periportal), production-normalized."""
    from scripts.cache_region_de import _build_region_basal_map
    from scripts.run_p1_eda import YU2022_L5_ZIP, _load_yu2022_l5
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = pathlib.Path(tmp_str)
        adata, _ = _load_yu2022_l5(YU2022_L5_ZIP, tmp / "yu2022")
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        bm = _build_region_basal_map(X, list(adata.var_names), "human",
                                     ext.gene_order, float(ext.norm_lo), float(ext.norm_hi),
                                     mouse_to_human=None)
    return bm["periportal"]


def _validate(gpu):
    ext = MultiDCPLatentExtractor(gpu=gpu)
    print(f"strict-load 0/0: {ext.strict_ok}")
    ext.set_basal(load_reference_basal(ext))
    # 5 distinct drug SMILES
    smis = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "CN1CCC[C@H]1c1cccnc1",
            "Clc1ccccc1C2=NCC(=O)Nc3ccc(cc23)[N+](=O)[O-]", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]
    G = np.vstack([ext.extract(s) for s in smis])  # [5, 306]
    drug, cell, dose = G[:, :DRUG_DIM], G[:, DRUG_DIM:DRUG_DIM+CELL_DIM], G[:, DRUG_DIM+CELL_DIM:]
    def block_var(B): return float(B.std(0).mean())
    print(f"global_features shape: {G.shape}")
    print(f"  drug block  [0:128]   across-drug std (mean): {block_var(drug):.4e}  (should be > 0)")
    print(f"  cell block  [128:178] across-drug std (mean): {block_var(cell):.4e}  (should be ~0: drug-independent)")
    print(f"  dose block  [178:306] across-drug std (mean): {block_var(dose):.4e}  (should be ~0: fixed dose)")
    print(f"  per-drug global_features pairwise Pearson (drug0 vs others): "
          f"{[round(float(np.corrcoef(G[0],G[i])[0,1]),4) for i in range(1,5)]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="2")
    _validate(ap.parse_args().gpu)
