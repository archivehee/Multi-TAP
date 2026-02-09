from typing import Dict, Tuple, Optional, Union, Sequence
from collections import OrderedDict
import pickle
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from ..utils.io_item import build_item_feature_table


def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _as_table_from_dict_or_array(obj,
                                 id_map: Optional[Dict[str, int]] = None,
                                 pad_one_based: bool = False) -> torch.Tensor:
    """
    - For dict[int->vec] or dict[str->vec], rebuild a table sized max_id+1
    - For np.ndarray[num, dim], convert directly to a tensor
    """
    if isinstance(obj, dict):
        # Some dict formats wrap embeddings as {"emb": np.ndarray, ...} or {"embeddings": {...}}
        if "emb" in obj:
            return _as_table_from_dict_or_array(obj["emb"], id_map=id_map, pad_one_based=pad_one_based)
        if "embeddings" in obj:
            return _as_table_from_dict_or_array(obj["embeddings"], id_map=id_map, pad_one_based=pad_one_based)
        # Build table by converting keys to integer ids
        try:
            keys = [int(k) for k in obj.keys()]
            dim = len(np.asarray(next(iter(obj.values())), dtype=np.float32))
            size = max(keys) + 1
            table = np.zeros((size, dim), dtype=np.float32)
            for k, v in obj.items():
                table[int(k)] = np.asarray(v, dtype=np.float32)
            if pad_one_based:
                padded = np.zeros((table.shape[0] + 1, dim), dtype=np.float32)
                padded[1:] = table
                table = padded
            return torch.from_numpy(table)
        except ValueError:
            if id_map is None:
                raise ValueError("Dict keys cannot be converted to int. Check user/item ID mappings.")
            dim = len(np.asarray(next(iter(obj.values())), dtype=np.float32))
            size = max(id_map.values()) + 1 + (1 if pad_one_based else 0)
            table = np.zeros((size, dim), dtype=np.float32)
            offset = 1 if pad_one_based else 0
            missing = 0
            for k, v in obj.items():
                if k not in id_map:
                    missing += 1
                    continue
                idx = id_map[k] + offset
                table[idx] = np.asarray(v, dtype=np.float32)
            if missing:
                print(f"[warn] {_as_table_from_dict_or_array.__name__}: {missing} ids missing from map; filled zeros.")
            return torch.from_numpy(table)
    elif isinstance(obj, np.ndarray):
        return torch.from_numpy(obj.astype(np.float32, copy=False))
    else:
        # Support UniCDR-style {"embeddings": np.array} format
        if isinstance(obj, dict) and "embeddings" in obj:
            emb = obj["embeddings"]
            return _as_table_from_dict_or_array(emb)
        raise ValueError("Unsupported PKL format.")


def _load_user2id_map(path: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw, idx = line.split()[:2]
            mapping[raw] = int(idx)
    return mapping


def _build_alignment(target_map: Optional[Dict[str, int]], source_map: Optional[Dict[str, int]]) -> Optional[torch.Tensor]:
    """
    Build a tensor mapping target indices to source indices.
    """
    if not target_map or not source_map:
        return None
    size = max(target_map.values()) + 1
    align = torch.full((size,), -1, dtype=torch.long)
    shared = 0
    for raw_id, tgt_idx in target_map.items():
        src_idx = source_map.get(raw_id)
        if src_idx is None:
            continue
        align[tgt_idx] = src_idx
        shared += 1
    if shared == 0:
        return None
    return align



def _persona_table_from_unified_pkl(pkl_obj, user_id_map: Optional[Dict[str, int]] = None) -> Tuple[torch.Tensor, int]:
    """
    Returns: (tensor [num_users, 5, persona_dim], persona_dim)
    """
    if not isinstance(pkl_obj, dict) or "embeddings" not in pkl_obj:
        raise ValueError("persona PKL must be {'embeddings': {uid: {'vecs': (5, D)}}}.")
    embs: Dict[str, Dict] = pkl_obj["embeddings"]
    first_key = next(iter(embs.keys()))
    persona_dim = int(np.asarray(embs[first_key]["vecs"]).shape[1])

    if user_id_map is None:
        # Legacy format: uid is a numeric string
        try:
            ids = sorted([int(k) for k in embs.keys()])
        except ValueError as e:
            raise ValueError("Persona embeddings and user2id mapping mismatch: provide user2id.txt.") from e
        num_users = max(ids) + 1 if ids else 0
        table = np.zeros((num_users, 5, persona_dim), dtype=np.float32)
        for uid in ids:
            table[uid] = np.asarray(embs[str(uid)]["vecs"], dtype=np.float32)
        return torch.from_numpy(table), persona_dim

    num_users = max(user_id_map.values()) + 1 if user_id_map else 0
    table = np.zeros((num_users, 5, persona_dim), dtype=np.float32)
    for raw_uid, idx in user_id_map.items():
        entry = embs.get(raw_uid)
        if entry is None:
            continue
        table[idx] = np.asarray(entry["vecs"], dtype=np.float32)
    return torch.from_numpy(table), persona_dim

# =========================
# Encoders
# =========================
class ItemEncoder(nn.Module):
    """
    feature_table: [I, 1664] (row 0 is padding)
    encode(iid) -> [B,256] or [B,K,256]
    """
    def __init__(self, feature_table: torch.Tensor, dropout: float = 0.1, train_emb: bool = True):
        super().__init__()
        D_in = feature_table.shape[1]  # 1664
        assert D_in == 1664, f"expect 1664-dim features, got {D_in}"
        if train_emb:
            self.feat = nn.Embedding.from_pretrained(feature_table, freeze=False)
        else:
            self.feat = FrozenEmbedding(feature_table)
        self.norm = nn.LayerNorm(D_in)
        self.mlp = nn.Sequential(
            nn.Linear(D_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
        )

    def forward(self, iid: torch.LongTensor) -> torch.Tensor:
        x = self.feat(iid)  # [..., 1664]
        if x.dim() == 3:  # [B,K,1664]
            B, K, D = x.shape
            x = x.view(B*K, D)
            x = self.mlp(self.norm(x))
            return x.view(B, K, -1)
        return self.mlp(self.norm(x))  # [B,256]


class UserEncoder(nn.Module):
    def __init__(
        self,
        persona_tables: Dict[str, torch.Tensor],
        domain_tables: Dict[str, torch.Tensor],
        aggregator_mode: str = "self_attention",
        dropout: float = 0.1,
        train_persona: bool = True,
        train_lgn: bool = True,
        default_domain: str = "target",
    ):
        super().__init__()
        if not persona_tables:
            raise ValueError("persona_tables must contain at least one domain.")
        if not domain_tables:
            raise ValueError("domain_tables must contain at least one domain.")

        self.persona_tables = nn.ParameterDict()
        for key, table in persona_tables.items():
            assert table.ndim == 3 and table.shape[1] == 5, "persona table expected [U,5,3072]"
            assert table.shape[2] == 3072, "persona dim expected 3072"
            self.persona_tables[key] = nn.Parameter(table.clone(), requires_grad=train_persona)
        self.persona_default = default_domain if default_domain in self.persona_tables else next(iter(self.persona_tables.keys()))

        def _mk_emb(weight: torch.Tensor):
            if train_lgn:
                return nn.Embedding.from_pretrained(weight, freeze=False)
            return FrozenEmbedding(weight)

        self.domain_embeddings = nn.ModuleDict()
        for key, table in domain_tables.items():
            assert table.shape[1] == 128, "LGN user dim expected 128"
            self.domain_embeddings[key] = _mk_emb(table)
        self.domain_default = default_domain if default_domain in self.domain_embeddings else next(iter(self.domain_embeddings.keys()))

        self.agg_modules = nn.ModuleDict()
        agg_state = None
        for key in self.persona_tables.keys():
            agg = PersonaAggregator(persona_dim=3072, out_dim=128,
                                    mode=aggregator_mode, dropout=dropout)
            if agg_state is None:
                agg_state = agg.state_dict()
            else:
                agg.load_state_dict(agg_state)
            self.agg_modules[key] = agg
        self.agg_default = (default_domain if default_domain in self.agg_modules
                            else (next(iter(self.agg_modules.keys())) if self.agg_modules else None))
        self.fuse_norm = nn.LayerNorm(256)
        self.fuse_mlp  = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
        )

    def _lookup_persona5(self, uid: torch.LongTensor, domain: Optional[str] = None) -> torch.Tensor:
        key = domain if (domain and domain in self.persona_tables) else self.persona_default
        return self.persona_tables[key].index_select(0, uid)

    def _agg_module(self, domain: Optional[str] = None) -> "PersonaAggregator":
        if not self.agg_modules:
            raise RuntimeError("No persona aggregator modules have been initialized.")
        key = domain if (domain and domain in self.agg_modules) else self.agg_default
        return self.agg_modules[key]

    def aggregate_persona(self, uid: torch.LongTensor, domain: Optional[str] = None) -> torch.Tensor:
        persona5 = self._lookup_persona5(uid, domain=domain)
        return self._agg_module(domain)(persona5)

    def domain_embedding(self, uid: torch.LongTensor, domain: Optional[str] = None) -> torch.Tensor:
        key = domain if (domain and domain in self.domain_embeddings) else self.domain_default
        return self.domain_embeddings[key](uid)

    def concat_persona_domain(self, persona_vec: torch.Tensor, domain_vec: torch.Tensor) -> torch.Tensor:
        return torch.cat([persona_vec, domain_vec], dim=-1)

    def concat_for_domain(
        self,
        uid: torch.LongTensor,
        domain: str,
        persona_cache: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        persona_vec = persona_cache if persona_cache is not None else self.aggregate_persona(uid, domain=domain)
        domain_vec = self.domain_embedding(uid, domain=domain)
        return self.concat_persona_domain(persona_vec, domain_vec)

    def forward(self, uid: torch.LongTensor, domain: Optional[str] = None, return_concat: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        persona_vec = self.aggregate_persona(uid, domain=domain)
        domain_vec = self.domain_embedding(uid, domain=domain)
        concat = self.concat_persona_domain(persona_vec, domain_vec)
        fused = self.fuse_mlp(self.fuse_norm(concat))
        if return_concat:
            return fused, concat
        return fused

    def freeze_persona_transform(self, domain: str):
        if domain not in self.agg_modules:
            return
        for param in self.agg_modules[domain].parameters():
            param.requires_grad = False




class FrozenEmbedding(nn.Module):
    """Behaves like nn.Embedding with fixed weights (no gradients)."""
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight)  # ensures requires_grad=False

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        return F.embedding(idx, self.weight)



class PersonaAggregator(nn.Module):
    def __init__(self, persona_dim: int, out_dim: int, mode: str = "mean", dropout: float = 0.0):
        super().__init__()
        assert mode in ["self_attention"]
        self.mode = mode
        self.persona_dim = persona_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        if mode == "self_attention":
            # Q/K/V linear transforms (Q uses one mask token)
            self.w_q = nn.Linear(persona_dim, persona_dim, bias=False)
            self.w_k = nn.Linear(persona_dim, persona_dim, bias=False)
            self.w_v = nn.Linear(persona_dim, persona_dim, bias=False)
            # Trainable mask token
            self.mask_token = nn.Parameter(torch.randn(1, persona_dim) * 0.02)
            # FFN (2 layers)
            self.ffn = nn.Sequential(
                nn.Linear(persona_dim, persona_dim),
                nn.ReLU(inplace=True),
                nn.Linear(persona_dim, persona_dim),
            )

        self.proj = nn.Linear(persona_dim, out_dim)

    def forward(self, persona5: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            pooled = persona5.mean(dim=1)  # [B, Dp]
        else:
            B, N, D = persona5.shape  # N=5
            # Q: replicate mask token across the batch
            q = self.mask_token.expand(B, -1)          # [B, D]
            q = self.w_q(q).unsqueeze(1)               # [B, 1, D]
            k = self.w_k(persona5)                     # [B, 5, D]
            v = self.w_v(persona5)                     # [B, 5, D]
            # scaled dot-product attention
            att = torch.matmul(q, k.transpose(1, 2)) / (D ** 0.5)  # [B, 1, 5]
            alpha = F.softmax(att, dim=-1)                          # [B, 1, 5]
            # Aggregate values
            out = torch.matmul(alpha, v).squeeze(1)                 # [B, D]
            if self.dropout is not None:
                out = self.dropout(out)
            # FFN
            out = self.ffn(out)                                     # [B, D]
            pooled = out

        # Projection (match dims)
        return self.proj(pooled)  # [B, out_dim]


class DoppelgangerAttention(nn.Module):
    """
    Use target/self persona self-attention output (128-d) to
    fuse target copy (Query) with source output (Key/Value) via cross-attention.
    """
    def __init__(self, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.out_dim = out_dim
        self.w_q = nn.Linear(out_dim, out_dim, bias=False)
        self.w_k = nn.Linear(out_dim, out_dim, bias=False)
        self.w_v = nn.Linear(out_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, tgt_vec: torch.Tensor, src_vec: torch.Tensor) -> torch.Tensor:
        """
        tgt_vec: [B, out_dim] (copy of target persona self-attention output)
        src_vec: [B, out_dim] (source persona self-attention output)
        """
        q = self.w_q(tgt_vec).unsqueeze(1)  # [B,1,D]
        k = self.w_k(src_vec).unsqueeze(1)  # [B,1,D]
        v = self.w_v(src_vec).unsqueeze(1)  # [B,1,D]
        att = torch.matmul(q, k.transpose(1, 2)) / (self.out_dim ** 0.5)  # [B,1,1]
        alpha = F.softmax(att, dim=-1)
        dop = torch.matmul(alpha, v).squeeze(1)  # [B,D]
        if self.dropout is not None:
            dop = self.dropout(dop)
        dop = self.ffn(dop)
        return dop



class AltTAPModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        # Paths/options
        persona_pkl   = cfg["persona_pkl"]
        lgn_user_pkl  = cfg["lgn_user_pkl"]
        aggregator    = cfg["aggregator"]   
        dropout       = float(cfg["dropout"])

        def _resolve_map_dir(path: Optional[str]) -> Optional[str]:
            return os.path.abspath(path) if path else None

        target_map_dir = _resolve_map_dir(cfg.get("target_map_dir") or cfg.get("map_dir"))
        source_map_dir = _resolve_map_dir(cfg.get("source_map_dir") or target_map_dir)

        def _load_map(map_dir: Optional[str]) -> Optional[Dict[str, int]]:
            if not map_dir:
                return None
            user2id_path = os.path.join(map_dir, "user2id.txt")
            if os.path.exists(user2id_path):
                return _load_user2id_map(user2id_path)
            return None

        target_user_map = _load_map(target_map_dir)
        source_user_map = _load_map(source_map_dir)
        if source_user_map is None:
            source_user_map = target_user_map

        persona_cache: Dict[Tuple[str, str], torch.Tensor] = {}

        def _prepare_persona_table(path: str,
                                   map_key: Optional[str],
                                   user_id_map: Optional[Dict[str, int]]) -> torch.Tensor:
            key = (os.path.abspath(path), map_key or "__nomap__")
            if key in persona_cache:
                return persona_cache[key]
            per_obj = _load_pkl(path)
            table, _ = _persona_table_from_unified_pkl(per_obj, user_id_map=user_id_map)
            persona_cache[key] = table
            return table

        persona_tables: Dict[str, torch.Tensor] = OrderedDict()
        target_persona_path = cfg.get("target_persona_pkl") or persona_pkl
        persona_tables["target"] = _prepare_persona_table(target_persona_path, target_map_dir, target_user_map)
        source_persona_path = cfg.get("source_persona_pkl")
        if source_persona_path:
            persona_tables["source"] = _prepare_persona_table(source_persona_path, source_map_dir, source_user_map)

        lgn_cache: Dict[Tuple[str, str], torch.Tensor] = {}

        def _load_lgn_table(path: str,
                            map_key: Optional[str],
                            user_id_map: Optional[Dict[str, int]]) -> torch.Tensor:
            key = (os.path.abspath(path), map_key or "__nomap__")
            if key in lgn_cache:
                return lgn_cache[key]
            u_obj = _load_pkl(path)
            table = _as_table_from_dict_or_array(u_obj, id_map=user_id_map, pad_one_based=False)
            lgn_cache[key] = table
            return table

        domain_tables: Dict[str, torch.Tensor] = OrderedDict()
        target_lgn_path = cfg.get("target_lgn_user_pkl") or lgn_user_pkl
        domain_tables["target"] = _load_lgn_table(target_lgn_path, target_map_dir, target_user_map)
        source_lgn_path = cfg.get("source_lgn_user_pkl")
        if source_lgn_path:
            domain_tables["source"] = _load_lgn_table(source_lgn_path, source_map_dir, source_user_map)

        alignment = None
        if "target" in persona_tables and "source" in persona_tables:
            alignment = _build_alignment(target_user_map, source_user_map)
        self.has_source_alignment = alignment is not None
        if alignment is not None:
            self.register_buffer("source_alignment", alignment)
        else:
            self.source_alignment = None

        # ---- Load item sources ----
        self.item_encoders = nn.ModuleDict()
        target_item_features = build_item_feature_table(
            map_dir=target_map_dir or cfg["map_dir"],
            lgn_item_pkl=cfg["lgn_item_pkl"],
            txt_item_pkl=cfg["txt_item_pkl"],
            lgn_dim=128,
            txt_dim=512,
        )
        self.item_encoders["target"] = ItemEncoder(
            target_item_features,
            dropout=dropout,
            train_emb=cfg.get("train_item_emb", True),
        )
        source_item_lgn = cfg.get("source_lgn_item_pkl")
        source_item_txt = cfg.get("source_txt_item_pkl")
        if source_item_lgn and source_item_txt:
            src_item_features = build_item_feature_table(
                map_dir=source_map_dir or target_map_dir or cfg["map_dir"],
                lgn_item_pkl=source_item_lgn,
                txt_item_pkl=source_item_txt,
                lgn_dim=128,
                txt_dim=512,
            )
            self.item_encoders["source"] = ItemEncoder(
                src_item_features,
                dropout=dropout,
                train_emb=cfg.get("train_item_emb", True),
            )

        self.user_enc = UserEncoder(persona_tables=persona_tables,
                                    domain_tables=domain_tables,
                                    aggregator_mode=aggregator, dropout=dropout,
                                    train_persona=cfg.get("train_persona_emb", True),
                                    train_lgn=cfg.get("train_lgn_emb", True),
                                    default_domain="target")
        self.doppel_agg = DoppelgangerAttention(out_dim=128, dropout=dropout)
        self.use_info_nce = bool(cfg.get("use_info_nce", False))
        self.info_nce_mode = str(cfg.get("info_nce_mode", "doppelganger")).lower()
        self.info_nce_tau = float(cfg.get("info_nce_tau", 1.0))
        self.info_nce_weight = float(cfg.get("info_nce_weight", 1.0))
        self.info_nce_detach_persona = bool(cfg.get("info_nce_detach_persona", False))
        self.source_domain_key = "source"
        self.target_domain_key = "target"
        self.source_persona_key = "source" if "source" in self.user_enc.persona_tables else None
        self.target_persona_key = "target" if "target" in self.user_enc.persona_tables else self.user_enc.persona_default
        need_domains = (self.source_domain_key in self.user_enc.domain_embeddings and
                        self.target_domain_key in self.user_enc.domain_embeddings)
        need_personas = (self.source_persona_key is not None and self.target_persona_key is not None)
        valid_modes = {"doppelganger", "lgn", "persona", "both"}
        if self.info_nce_mode not in valid_modes:
            print(f"[warn] Unknown info_nce_mode='{self.info_nce_mode}', fallback to 'doppelganger'.")
            self.info_nce_mode = "doppelganger"
        if self.use_info_nce:
            need_lgn = self.info_nce_mode in {"lgn", "both"}
            need_per = self.info_nce_mode in {"doppelganger", "persona", "both"}
            if (need_lgn and not need_domains) or (need_per and not need_personas):
                print("[warn] Missing source/target inputs for InfoNCE; disabling it.")
                self.use_info_nce = False

    def freeze_source_persona_transform(self):
        if self.source_persona_key is None:
            return
        self.user_enc.freeze_persona_transform(self.source_persona_key)

    def _item_encoder(self, domain: Optional[str] = None) -> ItemEncoder:
        if domain and domain in self.item_encoders:
            return self.item_encoders[domain]
        return self.item_encoders[self.target_domain_key]

    # BPR for training
    def bpr_loss(self,
                 uid: torch.LongTensor,
                 pos_iid: torch.LongTensor,
                 neg_iid: torch.LongTensor,
                 domain: Optional[str] = None) -> torch.Tensor:
        u256    = self.user_enc(uid, domain=domain)  # [B,256]
        item_encoder = self._item_encoder(domain)
        pos256  = item_encoder(pos_iid)             # [B,256]
        neg256  = item_encoder(neg_iid)             # [B,256]
        s_pos   = (u256 * pos256).sum(dim=-1)        # [B]
        s_neg   = (u256 * neg256).sum(dim=-1)        # [B]
        return -torch.mean(torch.log(torch.sigmoid(s_pos - s_neg) + 1e-12))

    def info_nce_loss(self, uid: torch.LongTensor) -> torch.Tensor:
        if not self.use_info_nce:
            return torch.zeros((), device=uid.device)
        if self.has_source_alignment and self.source_alignment is not None:
            aligned = self.source_alignment.index_select(0, uid)
            valid_mask = aligned >= 0
            if not torch.any(valid_mask):
                return torch.zeros((), device=uid.device)
            src_uid = aligned[valid_mask]
            tgt_uid = uid[valid_mask]
        else:
            src_uid = uid
            tgt_uid = uid
        if self.info_nce_mode == "doppelganger":
            src_persona = self.user_enc.aggregate_persona(src_uid, domain=self.source_persona_key)
            tgt_persona = self.user_enc.aggregate_persona(tgt_uid, domain=self.target_persona_key)
            if self.info_nce_detach_persona:
                src_persona = src_persona.detach()
                tgt_persona = tgt_persona.detach()
            src_vec = self.doppel_agg(tgt_persona, src_persona)
            tgt_vec = tgt_persona
        elif self.info_nce_mode == "lgn":
            src_vec = self.user_enc.domain_embedding(src_uid, domain=self.source_domain_key)
            tgt_vec = self.user_enc.domain_embedding(tgt_uid, domain=self.target_domain_key)
        elif self.info_nce_mode == "persona":
            src_vec = self.user_enc.aggregate_persona(src_uid, domain=self.source_persona_key)
            tgt_vec = self.user_enc.aggregate_persona(tgt_uid, domain=self.target_persona_key)
            if self.info_nce_detach_persona:
                src_vec = src_vec.detach()
                tgt_vec = tgt_vec.detach()
        else:  # both
            src_persona = self.user_enc.aggregate_persona(src_uid, domain=self.source_persona_key)
            tgt_persona = self.user_enc.aggregate_persona(tgt_uid, domain=self.target_persona_key)
            src_lgn = self.user_enc.domain_embedding(src_uid, domain=self.source_domain_key)
            tgt_lgn = self.user_enc.domain_embedding(tgt_uid, domain=self.target_domain_key)
            if self.info_nce_detach_persona:
                src_persona = src_persona.detach()
                tgt_persona = tgt_persona.detach()
            src_vec = torch.cat([src_persona, src_lgn], dim=-1)
            tgt_vec = torch.cat([tgt_persona, tgt_lgn], dim=-1)
        src_norm = F.normalize(src_vec, dim=-1)
        tgt_norm = F.normalize(tgt_vec, dim=-1)
        logits = torch.matmul(tgt_norm, src_norm.transpose(0, 1)) / max(self.info_nce_tau, 1e-6)
        labels = torch.arange(tgt_norm.size(0), device=uid.device)
        return F.cross_entropy(logits, labels)

    # Shared by Top-K / full ranking
    @torch.no_grad()
    def predict(self, uid: torch.LongTensor, cand_iids: torch.LongTensor, domain: Optional[str] = None) -> torch.Tensor:
        u256   = self.user_enc(uid, domain=domain)                   # [B,256]
        it256  = self._item_encoder(domain)(cand_iids)      # [B,K,256]
        u256   = u256.unsqueeze(1)                   # [B,1,256]
        return (u256 * it256).sum(dim=-1)            # [B,K]

    # Full-ranking acceleration: precompute item embeddings (can be cached after training)
    @torch.no_grad()
    def all_item_matrix(self, num_items: int, device: torch.device, domain: Optional[str] = None) -> torch.Tensor:
        # 1..num_items-1 (0 is padding)
        idx = torch.arange(1, num_items, device=device, dtype=torch.long)
        return self._item_encoder(domain)(idx)  # [I-1, 256]
