from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def doc_text(title: str, text: str) -> str:
    title = (title or "").strip()
    return f"{title} {text}".strip() if title else str(text).strip()


class DenseEncoder:

    def __init__(self, model_dir: str | Path, pooling: str, normalize: bool, max_len: int, device="cuda",
                 dtype: torch.dtype = torch.float32):
        assert pooling in ("mean", "cls", "last")
        self.tok = AutoTokenizer.from_pretrained(str(model_dir), padding_side="left" if pooling == "last" else "right")
        self.model = AutoModel.from_pretrained(str(model_dir), dtype=dtype).to(device).eval()
        self.pooling, self.normalize, self.max_len, self.device = pooling, normalize, max_len, device
        self.dim = self.model.config.hidden_size

    def lengths(self, texts: list[str], chunk: int = 20000) -> np.ndarray:
        out = []
        for i in range(0, len(texts), chunk):
            enc = self.tok(texts[i:i + chunk], truncation=True, max_length=self.max_len,
                           add_special_tokens=True)["input_ids"]
            out.append(np.fromiter((len(x) for x in enc), dtype=np.int32, count=len(enc)))
        return np.concatenate(out) if out else np.zeros(0, np.int32)

    def _forward(self, texts: list[str]):
        x = self.tok(texts, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
        h = self.model(**x).last_hidden_state
        return x, h

    def pool(self, x, h) -> torch.Tensor:
        h = h.float()
        if self.pooling == "mean":
            m = x["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1)
        elif self.pooling == "last":
            assert bool((x["attention_mask"][:, -1] == 1).all()), "last-token pooling expects left padding"
            e = h[:, -1]
        else:
            e = h[:, 0]
        if self.normalize:
            e = F.normalize(e, dim=-1)
        return e

    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        x, h = self._forward(texts)
        return self.pool(x, h).float().cpu().numpy()


def length_batches(lengths: np.ndarray, max_tokens: int, max_batch: int = 1024) -> list[np.ndarray]:
    order = np.lexsort((np.arange(len(lengths)), lengths))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = int(lengths[i])
        new_max = max(cur_max, L)
        if cur and (new_max * (len(cur) + 1) > max_tokens or len(cur) >= max_batch):
            batches.append(np.array(cur, dtype=np.int64))
            cur, new_max = [], L
        cur.append(int(i))
        cur_max = new_max
    if cur:
        batches.append(np.array(cur, dtype=np.int64))
    return batches


class BGEM3Heads:

    def __init__(self, model_dir: str | Path, max_len: int = 512, device="cuda"):
        model_dir = Path(model_dir)
        self.enc = DenseEncoder(model_dir, pooling="cls", normalize=True, max_len=max_len, device=device)
        hd = self.enc.dim
        self.colbert = torch.nn.Linear(hd, hd).to(device).eval()
        self.sparse = torch.nn.Linear(hd, 1).to(device).eval()
        self.colbert.load_state_dict(torch.load(model_dir / "colbert_linear.pt", map_location=device, weights_only=True))
        self.sparse.load_state_dict(torch.load(model_dir / "sparse_linear.pt", map_location=device, weights_only=True))
        t = self.enc.tok
        self.unused = {i for i in (t.cls_token_id, t.eos_token_id, t.pad_token_id, t.unk_token_id) if i is not None}
        self.device = device

    @torch.no_grad()
    def encode(self, texts: list[str]):
        x, h = self.enc._forward(texts)
        dense = F.normalize(h[:, 0], dim=-1)
        w = torch.relu(self.sparse(h)).squeeze(-1)
        ids = x["input_ids"]
        mask = x["attention_mask"]
        sparse = []
        w_cpu, ids_cpu, mask_cpu = w.cpu().numpy(), ids.cpu().numpy(), mask.cpu().numpy()
        for b in range(ids_cpu.shape[0]):
            d: dict[int, float] = {}
            for tid, wt, m in zip(ids_cpu[b], w_cpu[b], mask_cpu[b]):
                if not m or tid in self.unused or wt <= 0:
                    continue
                tid = int(tid)
                if wt > d.get(tid, 0.0):
                    d[tid] = float(wt)
            sparse.append(d)
        cv = self.colbert(h[:, 1:]) * mask[:, 1:].unsqueeze(-1).to(h.dtype)
        cv = F.normalize(cv, dim=-1)
        ntok = mask.sum(1)
        colbert = [cv[b, : int(ntok[b]) - 1] for b in range(cv.shape[0])]
        return dense.float().cpu().numpy(), sparse, colbert


def lexical_score(q: dict[int, float], p: dict[int, float]) -> float:
    return float(sum(w * p[t] for t, w in q.items() if t in p))


def maxsim_matrix(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return (q @ p.T).max(dim=1).values


def colbert_score(q: torch.Tensor, p: torch.Tensor) -> float:
    return float(maxsim_matrix(q, p).sum() / q.shape[0])
