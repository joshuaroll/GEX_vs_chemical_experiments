"""
MolFormer frozen chemical encoder — SMILES -> 768-dim embedding.

Model: ibm-research/MoLFormer-XL-both-10pct (HuggingFace)
HF commit SHA: 7b12d946c181a37f6012b9dc3b002275de070314
License: Apache-2.0 (IBM Research)

trust_remote_code=True is required: model uses custom tokenizer/architecture
(configuration_molformer.py, modeling_molformer.py, tokenization_molformer.py).
Security audit (2026-05-20): IBM Research publisher, Apache-2.0, standard ML files only,
no exec/eval/shell calls in model code. Approved for use.

NO fine-tuning. Model is frozen in eval mode. Do NOT call model.train() or optimizer.step().
"""
import torch
from transformers import AutoModel, AutoTokenizer

HF_MODEL_ID = "ibm-research/MoLFormer-XL-both-10pct"
HF_MODEL_SHA = "7b12d946c181a37f6012b9dc3b002275de070314"
EMBED_DIM = 768


class MolFormerWrapper:
    """Frozen MolFormer encoder. Converts SMILES strings to 768-dim embeddings."""

    def __init__(self, device: torch.device, cache_dir: str = None):
        """Load MolFormer from HuggingFace hub (or local cache).

        Args:
            device: torch.device to run inference on.
            cache_dir: Optional HF cache directory override.
        """
        self.device = device
        kwargs = dict(trust_remote_code=True)
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, **kwargs)
        self.model = AutoModel.from_pretrained(
            HF_MODEL_ID,
            deterministic_eval=True,
            **kwargs,
        )
        # Workaround: transformers >= 5.x does not correctly restore non-persistent
        # buffers (inv_freq, cos_cached, sin_cached) during from_pretrained().
        # The buffers are registered with persistent=False so they are excluded from
        # the checkpoint; transformers 5.x leaves them uninitialized (garbage values)
        # after weight loading, causing NaN in the linear attention rotary embeddings.
        # Fix: re-run _set_cos_sin_cache BEFORE moving to device. (transformers <=4.x was fine.)
        self._reinitialize_rotary_embeddings()
        self.model = self.model.to(device)
        self.model.eval()
        # Freeze all parameters — no gradients needed
        for param in self.model.parameters():
            param.requires_grad = False

    def _reinitialize_rotary_embeddings(self):
        """Re-initialize rotary embedding buffers post-load.

        transformers >= 5.x leaves non-persistent buffers (inv_freq, cos_cached,
        sin_cached) as uninitialized memory after from_pretrained(), causing NaN.
        This method re-runs the initialization for all rotary embedding modules.
        """
        for name, module in self.model.named_modules():
            if hasattr(module, 'inv_freq') and hasattr(module, '_set_cos_sin_cache') and hasattr(module, 'dim'):
                module.inv_freq = 1.0 / (
                    module.base ** (torch.arange(0, module.dim, 2).float() / module.dim)
                )
                module._set_cos_sin_cache(
                    seq_len=module.max_position_embeddings,
                    device=module.inv_freq.device,
                    dtype=torch.get_default_dtype(),
                )

    @torch.no_grad()
    def encode(self, smiles_list: list, batch_size: int = 64) -> torch.Tensor:
        """Encode a list of SMILES strings to a [N, 768] embedding tensor.

        Args:
            smiles_list: List of canonical SMILES strings.
            batch_size: Batch size for HF tokenizer/model forward pass.

        Returns:
            Tensor of shape [len(smiles_list), 768] on CPU.
        """
        all_embeddings = []
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=202,  # MolFormer max_position_embeddings
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # Use pooler_output (768-dim CLS embedding) if available
            if outputs.pooler_output is not None:
                emb = outputs.pooler_output  # [batch, 768]
            else:
                emb = outputs.last_hidden_state[:, 0, :]  # CLS token fallback
            all_embeddings.append(emb.cpu())
        return torch.cat(all_embeddings, dim=0)  # [N, 768]


def load_molformer(device: torch.device, cache_dir: str = None) -> MolFormerWrapper:
    """Convenience factory. Returns a MolFormerWrapper ready for encode()."""
    return MolFormerWrapper(device=device, cache_dir=cache_dir)
