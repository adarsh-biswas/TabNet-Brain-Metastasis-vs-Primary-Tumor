
import os, re, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import GEOparse
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, precision_recall_curve,
    average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, brier_score_loss, log_loss,
)
from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────
GEO_CACHE_DIR    = "./geo_data"
PATHWAY_FILE     = "kegg_pathways.gmt"

GSEAPY_LIBRARIES = [
    "KEGG_2021_Human",
    "Reactome_2022",
    "GO_Biological_Process_2023",
]

TOP_K_FEATURES   = None
MIN_PATHWAY_SIZE = 3
MAX_PATHWAY_SIZE = 200

# Within-pathway attention
ATTENTION_DIM     = 32
ATTENTION_DROPOUT = 0.3
ATTENTION_EPOCHS  = 120    # FIX E: 60 → 120, more time for hard negatives
ATTENTION_LR      = 5e-4
ATTENTION_WD      = 5e-4   # FIX G: 1e-3 → 5e-4, slightly relaxed L2

# Cross-pathway interaction attention (CPIA)
CPIA_HEADS        = 4
CPIA_DROPOUT      = 0.2
CPIA_GATE_HIDDEN  = 16
LAMBDA_INTERACT   = 0.15   # FIX F: 0.05 → 0.15, stronger contrastive push

TABNET_EPOCHS    = 300
TABNET_PATIENCE  = 40
RANDOM_SEED      = 42
N_BOOTSTRAP      = 1000
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Sensitivity-priority config ───────────────────────────────
POS_WEIGHT_MULTIPLIER = 4.0   # FIX A: 2× → 4×, much stronger minority penalty
TARGET_SENS           = 0.80  # FIX C: minimum sensitivity target for threshold
THRESH_FLOOR          = 0.25  # FIX D: hard ceiling on decision threshold

os.makedirs(GEO_CACHE_DIR, exist_ok=True)

print("=" * 65)
print("  BioTabNet v4.2 — Hard-Negative Encoder Fixes")
print("=" * 65)
print(f"  Device              : {DEVICE}")
print(f"  CPIA heads          : {CPIA_HEADS}")
print(f"  CPIA gate hidden    : {CPIA_GATE_HIDDEN}")
print(f"  Attention epochs    : {ATTENTION_EPOCHS}  (FIX E: was 60)")
print(f"  Attention WD        : {ATTENTION_WD}  (FIX G: was 1e-3)")
print(f"  Lambda interact     : {LAMBDA_INTERACT}  (FIX F: was 0.05)")
print(f"  pos_weight mult     : {POS_WEIGHT_MULTIPLIER}x  (FIX A: was 2x)")
print(f"  TabNet weights      : 1 (inv class freq)  (FIX B)")
print(f"  TabNet n_d/n_a      : 32  (FIX H: was 16)")
print(f"  TabNet n_steps      : 4   (FIX H: was 3)")
print(f"  Target sensitivity  : {TARGET_SENS}  (FIX C)")
print(f"  Threshold floor     : {THRESH_FLOOR}  (FIX D)")


# ══════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════

def get_probe_to_symbol(gse):
    mapping = {}
    for gpl_name, gpl in gse.gpls.items():
        table = gpl.table
        if table is None or table.empty or "ID" not in table.columns:
            continue
        symbol_col = None
        for c in ["Gene Symbol", "GENE_SYMBOL", "gene_symbol",
                  "Symbol", "SYMBOL", "gene_assignment", "mrna_assignment"]:
            if c in table.columns:
                symbol_col = c
                break
        if symbol_col is None:
            continue
        print(f"    Platform {gpl_name}: using column '{symbol_col}'")
        for _, row in table.iterrows():
            probe_id = str(row["ID"]).strip()
            raw      = str(row[symbol_col]).strip()
            if raw in ("", "nan", "---", "N/A", "NA"):
                continue
            symbol = re.split(r"\s*///\s*|,|;", raw)[0].strip()
            if symbol and symbol not in ("nan", "NA", ""):
                mapping[probe_id] = symbol
    return mapping


def load_geo(geo_id, label):
    print(f"\n  Loading {geo_id}  (label={label}) ...")
    gse  = GEOparse.get_GEO(geo=geo_id, destdir=GEO_CACHE_DIR, silent=True)
    data = gse.pivot_samples("VALUE").T
    data.columns = data.columns.astype(str)
    p2s  = get_probe_to_symbol(gse)
    if set(data.columns) & set(p2s.keys()):
        data = data[[c for c in data.columns if c in p2s]]
        data = data.rename(columns=p2s)
        data = data.T.groupby(level=0).mean().T
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    print(f"    Shape: {data.shape}  "
          f"mean={data.values.mean():.3f}  std={data.values.std():.3f}")
    return data, pd.Series(label, index=data.index, name="label")


print("\n[1/10] Loading GEO datasets...")
X1, y1 = load_geo("GSE50161",  label=0)
X2, y2 = load_geo("GSE108474", label=0)
X3, y3 = load_geo("GSE52604",  label=1)

n_primary = len(y1) + len(y2)
n_meta    = len(y3)
print(f"\n  Primary tumors : {n_primary}")
print(f"  Metastases     : {n_meta}")
print(f"  Imbalance      : {n_primary / n_meta:.1f}:1")

common = X1.columns.intersection(X2.columns).intersection(X3.columns)
print(f"  Common genes   : {len(common)}")

X_all     = pd.concat([X1[common], X2[common], X3[common]])
y_all     = pd.concat([y1, y2, y3])
study_all = pd.Series(
    ["GSE50161"] * len(X1) + ["GSE108474"] * len(X2) + ["GSE52604"] * len(X3),
    index=X_all.index, name="study")


# ══════════════════════════════════════════════════════════════
# 2. IMPUTE + STRATIFIED SPLIT
# ══════════════════════════════════════════════════════════════

print("\n[2/10] Impute + stratified split...")
imputer = SimpleImputer(strategy="mean")
X_imp   = pd.DataFrame(
    imputer.fit_transform(X_all),
    index=X_all.index, columns=X_all.columns)

X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X_imp, y_all, study_all,
    test_size=0.2, random_state=RANDOM_SEED, stratify=y_all)

print(f"  Train : {X_train.shape}  "
      f"(primary={(y_train==0).sum()}, meta={(y_train==1).sum()})")
print(f"  Test  : {X_test.shape}  "
      f"(primary={(y_test==0).sum()},  meta={(y_test==1).sum()})")


# ══════════════════════════════════════════════════════════════
# 3. SCALE  (train only)
# ══════════════════════════════════════════════════════════════

print("\n[3/10] Scaling (train only)...")
scaler     = StandardScaler()
X_tr_sc    = scaler.fit_transform(X_train)
X_te_sc    = scaler.transform(X_test)
X_tr_sc_df = pd.DataFrame(X_tr_sc, columns=X_train.columns)
X_te_sc_df = pd.DataFrame(X_te_sc, columns=X_test.columns)


# ══════════════════════════════════════════════════════════════
# 4. LOAD PATHWAYS
# ══════════════════════════════════════════════════════════════

def parse_gmt(filepath):
    pathways = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pathways[parts[0]] = [g.strip() for g in parts[2:] if g.strip()]
    return pathways


def load_gseapy_libraries(lib_names):
    try:
        import gseapy as gp
    except ImportError:
        print("  gseapy not installed. Run: pip install gseapy")
        return {}
    merged = {}
    for lib in lib_names:
        try:
            print(f"    Fetching {lib} ...")
            d = gp.get_library(name=lib, organism="Human")
            for k, v in d.items():
                merged[f"{lib}__{k}"] = v
            print(f"      → {len(d)} gene sets")
        except Exception as e:
            print(f"      WARNING: could not load {lib}: {e}")
    return merged


print("\n[4/10] Loading pathway gene sets...")
if os.path.exists(PATHWAY_FILE):
    print(f"  GMT file found: {PATHWAY_FILE}")
    raw_pathways = parse_gmt(PATHWAY_FILE)
    print(f"  Loaded {len(raw_pathways)} pathways from GMT")
else:
    print(f"  No GMT file — fetching via gseapy...")
    raw_pathways = load_gseapy_libraries(GSEAPY_LIBRARIES)
    print(f"  Total gene sets: {len(raw_pathways)}")

if not raw_pathways:
    raise RuntimeError("No pathways loaded.")


# ══════════════════════════════════════════════════════════════
# 5. PATHWAY-AWARE GENE SELECTION  (train only, no leakage)
# ══════════════════════════════════════════════════════════════

print("\n[5/10] Pathway-aware gene selection (train only)...")

all_pathway_genes = set()
for genes in raw_pathways.values():
    all_pathway_genes.update(genes)

candidate_genes = sorted(all_pathway_genes & set(X_train.columns))
print(f"  Pathway gene universe : {len(all_pathway_genes)}")
print(f"  Candidate genes       : {len(candidate_genes)}")

X_cand_tr  = X_tr_sc_df[candidate_genes].values
f_stats, _ = f_classif(X_cand_tr, y_train)
f_stats    = np.nan_to_num(f_stats, nan=0.0)
ranked_idx = np.argsort(f_stats)[::-1]

if TOP_K_FEATURES is not None:
    ranked_idx = ranked_idx[:min(TOP_K_FEATURES, len(candidate_genes))]

selected_genes = [candidate_genes[i] for i in ranked_idx]
gene_fstat     = {g: f_stats[candidate_genes.index(g)]
                  for g in selected_genes}   # F-stat lookup per gene

print(f"  Selected genes        : {len(selected_genes)}")
print(f"  Top 5 by F-stat       : {selected_genes[:5]}")

X_tr_sel = X_tr_sc_df[selected_genes].values
X_te_sel = X_te_sc_df[selected_genes].values


# ══════════════════════════════════════════════════════════════
# 6. BUILD PATHWAY INDEX  +  PATHWAY F-STAT PRIOR
# ══════════════════════════════════════════════════════════════

def build_pathway_index(sel_genes, pathways):
    g2i = {g: i for i, g in enumerate(sel_genes)}
    return [
        (name, [g2i[g] for g in genes if g in g2i])
        for name, genes in pathways.items()
        if MIN_PATHWAY_SIZE <= len([g for g in genes if g in g2i]) <= MAX_PATHWAY_SIZE
    ]


pathway_index = build_pathway_index(selected_genes, raw_pathways)
NUM_PATHWAYS  = len(pathway_index)

# Compute mean F-stat per pathway — used as biological prior in CPIA gate
pathway_fstat_prior = np.array([
    np.mean([gene_fstat[selected_genes[i]] for i in idx])
    for _, idx in pathway_index
], dtype=np.float32)

# Normalise to [0,1] for stable gating
pathway_fstat_prior = (pathway_fstat_prior - pathway_fstat_prior.min()) / \
                      (pathway_fstat_prior.max() - pathway_fstat_prior.min() + 1e-8)

print(f"\n  Usable pathways        : {NUM_PATHWAYS}")
print(f"  Pathway F-prior range  : "
      f"[{pathway_fstat_prior.min():.3f}, {pathway_fstat_prior.max():.3f}]")


# ══════════════════════════════════════════════════════════════
# 7. FIX 1 — SPLIT BEFORE OVERSAMPLING
# ══════════════════════════════════════════════════════════════

print("\n[6/10] FIX 1 — Split BEFORE oversampling...")

(X_subtrain, X_val_raw,
 y_subtrain, y_val_raw) = train_test_split(
    X_tr_sel, y_train.values,
    test_size=0.15, random_state=RANDOM_SEED, stratify=y_train.values)

print(f"  Sub-train (raw) : {len(y_subtrain)}  "
      f"(primary={(y_subtrain==0).sum()}, meta={(y_subtrain==1).sum()})")
print(f"  Val (clean)     : {len(y_val_raw)}  "
      f"(primary={(y_val_raw==0).sum()},  meta={(y_val_raw==1).sum()})")

# Oversample sub-train only
n_meta_st = int((y_subtrain == 1).sum())
n_prim_st = int((y_subtrain == 0).sum())
meta_idx  = np.where(y_subtrain == 1)[0]
prim_idx  = np.where(y_subtrain == 0)[0]
times     = n_prim_st // n_meta_st
remainder = n_prim_st %  n_meta_st
repeated  = np.concatenate([meta_idx] * times + [meta_idx[:remainder]])
all_idx   = np.concatenate([prim_idx, repeated])

rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(all_idx)

X_subtrain_bal = X_subtrain[all_idx]
y_subtrain_bal = y_subtrain[all_idx]

print(f"  Sub-train (bal) : {len(y_subtrain_bal)}  "
      f"(primary={(y_subtrain_bal==0).sum()}, meta={(y_subtrain_bal==1).sum()})")


# ══════════════════════════════════════════════════════════════
# 8. NEURAL ARCHITECTURE
#
#  Stage 1: PathwayAttention — within-pathway (same as v3)
#  Stage 2: CrossPathwayInteractionAttention (CPIA) — NEW
#
#  Mathematical definition of CPIA:
#
#  Given pathway embeddings E ∈ R^{B × P}:
#
#  (a) Cosine similarity matrix:
#      S_ij = (e_i · e_j) / (||e_i|| · ||e_j|| + ε)
#
#  (b) Gating MLP per (i,j) pair:
#      g_ij = σ( MLP( [S_ij,  S_ij²,
#                       S_ij·f_i,  S_ij·f_j] ) )
#      where f_i, f_j = normalised F-stat prior of pathways i,j
#
#  (c) Gated attention weight:
#      α_ij = softmax_j( g_ij · S_ij / √1 )
#
#  (d) Interaction-aware pathway representation:
#      h_i = Σ_j α_ij · e_j         ← context from all pathways
#
#  (e) Residual fusion:
#      out_i = LayerNorm( e_i + dropout(h_i) )
#
#  Multi-head: run (a)-(e) independently H times on
#  projected subspaces, then concatenate + linear project
#  back to P dims.
# ══════════════════════════════════════════════════════════════

class PathwayAttention(nn.Module):
    """Within-pathway soft attention (unchanged from v3)."""
    def __init__(self, n_genes: int,
                 hidden: int = ATTENTION_DIM,
                 dropout: float = ATTENTION_DROPOUT):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_genes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        weights = self.softmax(self.score(x))
        out     = (x * weights).sum(dim=1, keepdim=True)
        return out, weights


class CrossPathwayInteractionAttention(nn.Module):
    """
    CPIA: Novel cross-pathway interaction attention.

    Inputs:
        E   : (B, P)  — pathway embeddings from Stage 1
        f   : (P,)    — normalised F-stat prior per pathway

    Output:
        (B, P) — interaction-aware pathway representations
    """
    def __init__(self, n_pathways: int,
                 n_heads: int      = CPIA_HEADS,
                 gate_hidden: int  = CPIA_GATE_HIDDEN,
                 dropout: float    = CPIA_DROPOUT):
        super().__init__()
        self.P        = n_pathways
        self.H        = n_heads
        self.d_head   = max(1, n_pathways // n_heads)

        # Linear projection into H subspaces
        self.proj_in  = nn.Linear(n_pathways, self.H * self.d_head)

        # Gating MLP: maps 4 features → scalar gate g_ij
        # Features: [S_ij, S_ij², S_ij·f_i, S_ij·f_j]
        self.gate_mlp = nn.Sequential(
            nn.Linear(4, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

        # Project multi-head output back to P dims
        self.proj_out = nn.Linear(self.H * self.d_head, n_pathways)
        self.norm     = nn.LayerNorm(n_pathways)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, E, f_prior):
        """
        E       : (B, P)
        f_prior : (P,)  — biological F-stat prior
        Returns : (B, P)
        """
        B, P = E.shape
        f    = f_prior.to(E.device)                  # (P,)

        # ── Stage 2a: project to H subspaces ────────────────
        Z = self.proj_in(E)                           # (B, H*d)
        Z = Z.view(B, self.H, self.d_head)            # (B, H, d)

        head_outputs = []

        for h in range(self.H):
            z = Z[:, h, :]                            # (B, d)

            # ── Stage 2b: cosine similarity matrix ──────────
            z_norm = F.normalize(z, p=2, dim=1)       # (B, d)
            S = torch.mm(z_norm, z_norm.t())          # (B, B)
            # We want pathway-wise S: treat each row as a pathway
            # Reinterpret: S_ij over pathway dim using full-batch E
            # For within-batch pathway interactions use E directly:
            E_norm = F.normalize(E, p=2, dim=0)       # (B, P) — norm over batch
            # Pathway cosine sim: (P, P)
            Ep = E_norm.t()                           # (P, B)
            S_pw = torch.mm(Ep, Ep.t())               # (P, P)  ← S_ij

            # ── Stage 2c: gating MLP ─────────────────────────
            # Build 4-feature tensor for each (i,j) pair: (P, P, 4)
            S_sq   = S_pw ** 2                        # (P, P)
            fi     = f.unsqueeze(1).expand(P, P)      # (P, P) — row = f_i
            fj     = f.unsqueeze(0).expand(P, P)      # (P, P) — col = f_j
            S_fi   = S_pw * fi                        # (P, P)
            S_fj   = S_pw * fj                        # (P, P)

            gate_in = torch.stack(
                [S_pw, S_sq, S_fi, S_fj], dim=2      # (P, P, 4)
            ).view(P * P, 4)

            g = self.gate_mlp(gate_in).view(P, P)    # (P, P)

            # ── Stage 2d: gated scaled attention ────────────
            # α_ij = softmax_j( g_ij * S_ij / √d_head )
            score  = g * S_pw / (self.d_head ** 0.5)  # (P, P)
            alpha  = F.softmax(score, dim=1)           # (P, P)

            # ── Stage 2e: weighted aggregation ──────────────
            # h_i = Σ_j α_ij * e_j  across pathway dim
            # E: (B, P) → for each sample, aggregate pathways
            # alpha: (P, P) — shared across batch
            h = torch.mm(E, alpha.t())                 # (B, P)

            # Project h back to d_head space via slice
            # (simple: take first d_head dims after projection)
            h_proj = h[:, :self.d_head]                # (B, d)
            head_outputs.append(h_proj)

        # ── Concatenate heads + project ──────────────────────
        multi = torch.cat(head_outputs, dim=1)         # (B, H*d)
        out   = self.proj_out(multi)                   # (B, P)

        # ── Residual + LayerNorm ─────────────────────────────
        out = self.norm(E + self.dropout(out))         # (B, P)
        return out, alpha                              # also return last α


class BioTabNetEncoder(nn.Module):
    """
    Full two-stage encoder:
      Stage 1: PathwayAttention (within-pathway)
      Stage 2: CrossPathwayInteractionAttention (cross-pathway)
    """
    def __init__(self, pathway_index, f_prior,
                 hidden=ATTENTION_DIM):
        super().__init__()
        self.pathway_index = pathway_index
        self.P             = len(pathway_index)

        # Register F-stat prior as buffer (not a parameter)
        self.register_buffer(
            "f_prior", torch.FloatTensor(f_prior))

        # Stage 1: one attention head per pathway
        self.within_heads = nn.ModuleList([
            PathwayAttention(len(idx), hidden)
            for _, idx in pathway_index
        ])

        # Stage 2: cross-pathway interaction attention
        self.cpia = CrossPathwayInteractionAttention(
            n_pathways=self.P,
            n_heads=CPIA_HEADS,
            gate_hidden=CPIA_GATE_HIDDEN,
            dropout=CPIA_DROPOUT,
        )

    def forward(self, x):
        # ── Stage 1: within-pathway embeddings ──────────────
        e = torch.cat(
            [head(x[:, idx])[0]
             for (_, idx), head in
             zip(self.pathway_index, self.within_heads)],
            dim=1)                                     # (B, P)

        # ── Stage 2: cross-pathway interaction ──────────────
        out, alpha = self.cpia(e, self.f_prior)        # (B, P)
        return out, alpha

    def encode(self, x):
        out, _ = self.forward(x)
        return out

    def get_within_weights(self, x, gene_names):
        """Return top-weighted genes per pathway (Stage 1)."""
        result = {}
        for (name, idx), head in zip(self.pathway_index, self.within_heads):
            _, w = head(x[:, idx])
            avg  = w.mean(dim=0).detach().cpu().numpy()
            result[name] = sorted(
                zip([gene_names[i] for i in idx], avg),
                key=lambda t: t[1], reverse=True)
        return result

    def get_interaction_matrix(self, x):
        """Return (P, P) mean interaction attention weights."""
        _, alpha = self.forward(x)
        return alpha.detach().cpu().numpy()


# ══════════════════════════════════════════════════════════════
# 9. INTERACTION CONTRASTIVE LOSS
#
#  Encourages primary vs metastasis samples to have
#  distinguishable cross-pathway interaction patterns.
#
#  L_interact = mean_prim(||h||²) - mean_meta(||h||²)  ... no,
#
#  We use a margin-based contrastive on the pathway embedding:
#
#  L_interact = max(0, margin - D(μ_meta, μ_prim))
#
#  where D = cosine distance between class centroids in
#  the CPIA output space, and margin = 0.3.
#
#  This explicitly forces primary and metastatic pathway
#  interaction profiles to be separable.
# ══════════════════════════════════════════════════════════════

CONTRASTIVE_MARGIN = 0.3

def interaction_contrastive_loss(h, y):
    """
    h : (B, P)  — CPIA output embeddings
    y : (B,)    — labels {0,1}
    """
    mask_prim = (y == 0)
    mask_meta = (y == 1)

    if mask_prim.sum() == 0 or mask_meta.sum() == 0:
        return torch.tensor(0.0, device=h.device)

    mu_prim = h[mask_prim].mean(dim=0)   # (P,)
    mu_meta = h[mask_meta].mean(dim=0)   # (P,)

    cos_sim  = F.cosine_similarity(
        mu_prim.unsqueeze(0), mu_meta.unsqueeze(0))   # scalar
    cos_dist = 1.0 - cos_sim                          # ∈ [0, 2]

    loss = torch.clamp(CONTRASTIVE_MARGIN - cos_dist, min=0.0)
    return loss


# ══════════════════════════════════════════════════════════════
# 10. PRE-TRAIN ENCODER
# ══════════════════════════════════════════════════════════════

print(f"\n[7/10] Pre-training BioTabNetEncoder "
      f"(Stage 1 + CPIA, {ATTENTION_EPOCHS} epochs)...")

X_atn_t  = torch.FloatTensor(X_subtrain).to(DEVICE)
y_atn_t  = torch.FloatTensor(y_subtrain).to(DEVICE)
X_val_t  = torch.FloatTensor(X_val_raw).to(DEVICE)
y_val_t  = torch.FloatTensor(y_val_raw).to(DEVICE)
X_te_t   = torch.FloatTensor(X_te_sel).to(DEVICE)
X_full_t = torch.FloatTensor(X_tr_sel).to(DEVICE)

# FIX A: amplify pos_weight by POS_WEIGHT_MULTIPLIER (default 2×)
# Raw imbalance ratio × multiplier tells encoder missing a metastasis
# is twice as costly as the natural ratio already implies.
pw_val   = (float(n_prim_st) / float(max(n_meta_st, 1))) * POS_WEIGHT_MULTIPLIER
print(f"  pos_weight (primary/meta × {POS_WEIGHT_MULTIPLIER}x): {pw_val:.1f}  (FIX A)")

encoder  = BioTabNetEncoder(
    pathway_index, pathway_fstat_prior).to(DEVICE)

clf_head = nn.Sequential(
    nn.Linear(NUM_PATHWAYS, 64), nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 1)
).to(DEVICE)

pos_weight = torch.tensor([pw_val]).to(DEVICE)
bce_loss   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer  = optim.Adam(
    list(encoder.parameters()) + list(clf_head.parameters()),
    lr=ATTENTION_LR, weight_decay=ATTENTION_WD)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=ATTENTION_EPOCHS, eta_min=1e-5)

batch_size = min(64, max(8, len(y_subtrain) // 4))
loader     = DataLoader(
    TensorDataset(X_atn_t, y_atn_t),
    batch_size=batch_size, shuffle=True)

best_val_auc = 0.0
best_state   = None
patience_ctr = 0
ATN_PATIENCE = 15

for epoch in range(ATTENTION_EPOCHS):
    encoder.train(); clf_head.train()
    epoch_loss = 0.0
    epoch_l_interact = 0.0

    for xb, yb in loader:
        optimizer.zero_grad()

        h, _   = encoder(xb)                           # (B, P)
        logits = clf_head(h).squeeze()

        l_bce      = bce_loss(logits, yb)
        l_interact = interaction_contrastive_loss(h, yb)
        loss       = l_bce + LAMBDA_INTERACT * l_interact

        loss.backward()
        nn.utils.clip_grad_norm_(
            list(encoder.parameters()) +
            list(clf_head.parameters()), max_norm=1.0)
        optimizer.step()

        epoch_loss       += l_bce.item()
        epoch_l_interact += l_interact.item()

    scheduler.step()

    # Evaluate on clean val
    encoder.eval(); clf_head.eval()
    with torch.no_grad():
        h_val, _   = encoder(X_val_t)
        val_logits = clf_head(h_val).squeeze()
        val_prob   = torch.sigmoid(val_logits).cpu().numpy()
        val_auc    = roc_auc_score(y_val_raw, val_prob) \
                     if len(np.unique(y_val_raw)) == 2 else 0.5

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state   = {
            "encoder":  {k: v.clone() for k, v in
                         encoder.state_dict().items()},
            "clf_head": {k: v.clone() for k, v in
                         clf_head.state_dict().items()},
        }
        patience_ctr = 0
    else:
        patience_ctr += 1

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            h_tr, _  = encoder(X_atn_t)
            tr_preds = (torch.sigmoid(clf_head(h_tr).squeeze()) > 0.5).float()
            meta_mask = y_atn_t == 1
            meta_rec  = (tr_preds[meta_mask] == 1).float().mean().item() \
                        if meta_mask.sum() > 0 else 0.0
        print(f"    Epoch {epoch+1:3d}/{ATTENTION_EPOCHS}  "
              f"bce={epoch_loss/len(loader):.4f}  "
              f"L_interact={epoch_l_interact/len(loader):.4f}  "
              f"val_auc={val_auc:.4f}  "
              f"best={best_val_auc:.4f}  "
              f"meta_rec={meta_rec:.3f}")

    if patience_ctr >= ATN_PATIENCE:
        print(f"    Early stop at epoch {epoch+1}  "
              f"best_val_auc={best_val_auc:.4f}")
        break

print(f"  Best encoder val AUC: {best_val_auc:.4f}")
encoder.load_state_dict(best_state["encoder"])
clf_head.load_state_dict(best_state["clf_head"])

# Encode all splits
encoder.eval()
with torch.no_grad():
    X_sub_t_bal  = torch.FloatTensor(X_subtrain_bal).to(DEVICE)
    X_sub_pw_bal, _ = encoder(X_sub_t_bal)
    X_sub_pw_bal    = X_sub_pw_bal.cpu().numpy()

    X_val_pw, _  = encoder(X_val_t)
    X_val_pw     = X_val_pw.cpu().numpy()

    X_te_pw, _   = encoder(X_te_t)
    X_te_pw      = X_te_pw.cpu().numpy()

    X_full_pw, _ = encoder(X_full_t)
    X_full_pw    = X_full_pw.cpu().numpy()

print(f"  Encoder output : {X_sub_pw_bal.shape[1]} features/sample")


# ══════════════════════════════════════════════════════════════
# 11. TABNET
# ══════════════════════════════════════════════════════════════

print("\n[8/10] Training TabNet...")
print(f"  Train (bal) : {len(y_subtrain_bal)}")
print(f"  Val (clean) : {len(y_val_raw)}")

# FIX B: weights=1 passed to fit() — inverse class-frequency weighting
# FIX H: n_d/n_a 16→32, n_steps 3→4 — more capacity for hard cases
tabnet = TabNetClassifier(
    n_d=32, n_a=32,        # FIX H: was 16
    n_steps=4,             # FIX H: was 3
    gamma=1.5,
    lambda_sparse=1e-2,
    optimizer_params=dict(lr=2e-3, weight_decay=1e-4),
    mask_type="sparsemax",
    seed=RANDOM_SEED,
    verbose=1)

tabnet.fit(
    X_sub_pw_bal, y_subtrain_bal,
    eval_set=[(X_val_pw, y_val_raw)],
    eval_metric=["auc"],
    max_epochs=TABNET_EPOCHS,
    patience=TABNET_PATIENCE,
    batch_size=min(64, len(y_subtrain_bal)),
    virtual_batch_size=min(32, len(y_subtrain_bal)),
    weights=1)             # FIX B: inverse class-frequency weighting

# ── FIX C + D: Sensitivity-priority threshold tuning ─────────
# FIX C: Instead of maximising F1 (which collapses on tiny val),
#   we set a hard sensitivity floor (TARGET_SENS) and pick the
#   threshold with best precision among those that meet it.
# FIX D: Apply THRESH_FLOOR — never let the final threshold
#   exceed this value so borderline cases always call metastasis.
val_prob_tn           = tabnet.predict_proba(X_val_pw)[:, 1]
val_auc_final         = roc_auc_score(y_val_raw, val_prob_tn)
prec, rec, thresholds = precision_recall_curve(y_val_raw, val_prob_tn)

# Thresholds array has one fewer element than prec/rec
# prec[:-1], rec[:-1] align with thresholds
meets_sens  = rec[:-1] >= TARGET_SENS      # boolean mask
tuning_note = ""

if meets_sens.any():
    # Among all thresholds achieving target sensitivity,
    # pick the one with the highest precision
    best_among = np.argmax(prec[:-1][meets_sens])
    best_thresh = float(thresholds[meets_sens][best_among])
    tuning_note = f"sens≥{TARGET_SENS} + best precision"
else:
    # TARGET_SENS unreachable on val — fall back to best
    # sensitivity above 0.50, or just the lowest threshold
    above_half = rec[:-1] >= 0.50
    if above_half.any():
        best_thresh = float(thresholds[above_half][
            np.argmax(prec[:-1][above_half])])
        tuning_note = "fallback: sens≥0.50 + best precision"
    else:
        best_thresh = float(thresholds[-1])
        tuning_note = "fallback: lowest available threshold"

# FIX D: enforce hard ceiling — never go above THRESH_FLOOR
best_thresh = min(best_thresh, THRESH_FLOOR)

print(f"\n  Threshold tuning strategy : {tuning_note}")
print(f"  Tuned threshold           : {best_thresh:.3f}  "
      f"(ceiling={THRESH_FLOOR}, FIX C+D)")
print(f"  Val ROC-AUC (clean)       : {val_auc_final:.4f}")

# Report what sensitivity/specificity we get at this threshold on val
val_pred_thresh = (val_prob_tn >= best_thresh).astype(int)
if len(np.unique(y_val_raw)) == 2:
    v_tn, v_fp, v_fn, v_tp = confusion_matrix(
        y_val_raw, val_pred_thresh).ravel()
    v_sens = v_tp / (v_tp + v_fn + 1e-8)
    v_spec = v_tn / (v_tn + v_fp + 1e-8)
    print(f"  Val sensitivity @ thresh  : {v_sens:.3f}")
    print(f"  Val specificity @ thresh  : {v_spec:.3f}")


# ══════════════════════════════════════════════════════════════
# METRIC HELPERS
# ══════════════════════════════════════════════════════════════

def ece_score(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi)
        if m.sum() == 0:
            continue
        ece += m.sum() * abs(y_true[m].mean() - y_prob[m].mean())
    return ece / len(y_true)


def bootstrap_ci(y_true, y_prob, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    auc_b, prauc_b = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        auc_b.append(roc_auc_score(yt, yp))
        prauc_b.append(average_precision_score(yt, yp))
    return (np.percentile(auc_b, [2.5, 97.5]),
            np.percentile(prauc_b, [2.5, 97.5]))


# ══════════════════════════════════════════════════════════════
# 12. RESULTS
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  RESULTS  —  BioTabNet v4.2")
print("=" * 65)

y_prob     = tabnet.predict_proba(X_te_pw)[:, 1]
y_pred_05  = (y_prob >= 0.50).astype(int)
y_pred_opt = (y_prob >= best_thresh).astype(int)

auc    = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
mcc    = matthews_corrcoef(y_test, y_pred_opt)
bacc   = balanced_accuracy_score(y_test, y_pred_opt)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt).ravel()
sens  = tp / (tp + fn + 1e-8)
spec  = tn / (tn + fp + 1e-8)
prec_ = tp / (tp + fp + 1e-8)
f1_m  = 2 * prec_ * sens / (prec_ + sens + 1e-8)
lr_p  = sens / (1 - spec + 1e-8)

brier = brier_score_loss(y_test, y_prob)
ece   = ece_score(y_test.values, y_prob)
ll    = log_loss(y_test, y_prob)

ci_auc, ci_prauc = bootstrap_ci(
    y_test.values, y_prob, n=N_BOOTSTRAP, seed=RANDOM_SEED)

print(f"\n  Classification report (threshold = {best_thresh:.2f})")
print(classification_report(
    y_test, y_pred_opt,
    target_names=["Primary", "Metastasis"], zero_division=0))

print("  ── Discrimination ─────────────────────────────────────────")
print(f"  ROC-AUC  : {auc:.4f}   95% CI [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]")
print(f"  PR-AUC   : {pr_auc:.4f}   95% CI [{ci_prauc[0]:.4f}, {ci_prauc[1]:.4f}]")
print(f"  Val AUC  : {val_auc_final:.4f}  (clean, used for early stop)")

print("\n  ── Balance ────────────────────────────────────────────────")
print(f"  MCC      : {mcc:.4f}")
print(f"  Bal. Acc : {bacc:.4f}")

print("\n  ── Threshold-level ────────────────────────────────────────")
print(f"  Sensitivity : {sens:.4f}")
print(f"  Specificity : {spec:.4f}")
print(f"  Precision   : {prec_:.4f}")
print(f"  F1 (meta)   : {f1_m:.4f}")
print(f"  LR+         : {lr_p:.2f}")

print("\n  ── Confusion matrix ───────────────────────────────────────")
print(f"                  Pred Primary   Pred Meta")
print(f"  Actual Primary      {tn:4d}          {fp:4d}")
print(f"  Actual Meta         {fn:4d}          {tp:4d}")

print("\n  ── Calibration ────────────────────────────────────────────")
print(f"  Brier    : {brier:.4f}")
print(f"  ECE      : {ece:.4f}")
print(f"  Log-loss : {ll:.4f}")
if ece > 0.10:
    print(f"\n  NOTE: ECE > 0.10 — consider Platt scaling.")

print("\n  ── Per-study breakdown ────────────────────────────────────")
print(f"  {'Study':<12}  {'n':>4}  {'AUC':>6}  {'MCC':>6}  "
      f"{'Sens':>6}  {'Spec':>6}")
print("  " + "-" * 52)
for s in sorted(s_test.unique()):
    mask = (s_test.values == s)
    n    = mask.sum()
    yt   = y_test.values[mask]
    yp   = y_prob[mask]
    yd   = y_pred_opt[mask]
    if len(np.unique(yt)) < 2:
        print(f"  {s:<12}  {n:4d}  {'--':>6}  {'--':>6}  "
              f"{'--':>6}  {'--':>6}  (single class)")
        continue
    s_auc              = roc_auc_score(yt, yp)
    s_mcc              = matthews_corrcoef(yt, yd)
    s_tn, s_fp, s_fn, s_tp = confusion_matrix(yt, yd).ravel()
    s_sens             = s_tp / (s_tp + s_fn + 1e-8)
    s_spec             = s_tn / (s_tn + s_fp + 1e-8)
    print(f"  {s:<12}  {n:4d}  {s_auc:6.4f}  {s_mcc:6.4f}  "
          f"{s_sens:6.4f}  {s_spec:6.4f}")

# ── Pathway interaction heatmap (top 20 pathways) ────────────
print("\n  ── Cross-pathway interaction analysis (CPIA) ──────────────")
encoder.eval()
with torch.no_grad():
    n   = min(200, X_full_t.shape[0])
    A   = encoder.get_interaction_matrix(X_full_t[:n])     # (P, P)
    wts = encoder.get_within_weights(X_full_t[:n], selected_genes)

# Top pathways by row-sum of interaction weights
pathway_names   = [name for name, _ in pathway_index]
interaction_sum = A.sum(axis=1)
top20_idx       = np.argsort(interaction_sum)[::-1][:20]

print(f"  Top 10 most interactive pathways (by total attention received):")
for rank, i in enumerate(top20_idx[:10], 1):
    name_short = pathway_names[i][:50]
    print(f"    {rank:2d}. {name_short:<52} score={interaction_sum[i]:.4f}")

print("\n  Top genes per pathway (Stage 1 attention weights):")
for name, gw in list(wts.items())[:10]:
    top3 = gw[:3]
    print(f"  {name[:44]:<46} -> "
          f"{', '.join(f'{g}({w:.3f})' for g, w in top3)}")

print("\n" + "=" * 65)
print("  SUMMARY  —  BioTabNet v4.2")
print("=" * 65)
print(f"  ROC-AUC  : {auc:.4f}   [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]")
print(f"  PR-AUC   : {pr_auc:.4f}   [{ci_prauc[0]:.4f}, {ci_prauc[1]:.4f}]")
print(f"  Val AUC  : {val_auc_final:.4f}  (clean)")
print(f"  MCC      : {mcc:.4f}")
print(f"  Sens.    : {sens:.4f}   Spec. : {spec:.4f}")
print(f"  LR+      : {lr_p:.2f}    Brier : {brier:.4f}")
print(f"  Threshold: {best_thresh:.3f}")
print(f"  Pathways : {NUM_PATHWAYS}  |  Genes: {len(selected_genes)}")
print("=" * 65)

# ── Save outputs ──────────────────────────────────────────────
pd.DataFrame({
    "sample":       y_test.index,
    "study":        s_test.values,
    "true_label":   y_test.values,
    "pred_default": y_pred_05,
    "pred_tuned":   y_pred_opt,
    "prob_meta":    y_prob,
}).to_csv("v4_2_results.csv", index=False)

pd.DataFrame({
    "metric": [
        "ROC-AUC", "ROC-AUC CI low", "ROC-AUC CI high",
        "PR-AUC",  "PR-AUC CI low",  "PR-AUC CI high",
        "Val AUC (clean)",
        "MCC", "Balanced Accuracy",
        "Sensitivity", "Specificity", "Precision (meta)", "F1 (meta)",
        "LR+", "Brier Score", "ECE", "Log-loss",
        "Threshold", "TP", "TN", "FP", "FN",
        "Num Pathways", "Num Genes",
    ],
    "value": [
        round(auc, 4), round(float(ci_auc[0]), 4), round(float(ci_auc[1]), 4),
        round(pr_auc, 4), round(float(ci_prauc[0]), 4), round(float(ci_prauc[1]), 4),
        round(val_auc_final, 4),
        round(mcc, 4), round(bacc, 4),
        round(sens, 4), round(spec, 4), round(prec_, 4), round(f1_m, 4),
        round(lr_p, 4), round(brier, 4), round(ece, 4), round(ll, 4),
        round(best_thresh, 4), int(tp), int(tn), int(fp), int(fn),
        NUM_PATHWAYS, len(selected_genes),
    ],
}).to_csv("v4_2_metrics.csv", index=False)

# Save interaction heatmap (top 20 pathways)
top20_names = [pathway_names[i] for i in top20_idx]
pd.DataFrame(
    A[np.ix_(top20_idx, top20_idx)],
    index=top20_names, columns=top20_names
).to_csv("v4_2_interaction_heatmap.csv")

print("\n  Saved -> v4_2_results.csv")
print("  Saved -> v4_2_metrics.csv")
print("  Saved -> v4_2_interaction_heatmap.csv  (top-20 pathway crosstalk)")

# ── Built-in diagnostic: metastasis probability breakdown ─────
# No need to run a separate script — this runs automatically.
print("\n" + "=" * 65)
print("  DIAGNOSTIC — Metastasis Sample Probabilities")
print("=" * 65)

meta_mask  = y_test.values == 1
meta_probs = y_prob[meta_mask]
meta_preds = y_pred_opt[meta_mask]

print(f"\n  {'Sample':<20} {'P(meta)':>8}  {'Pred':>6}  {'Result':>10}")
print("  " + "-" * 50)
for i, (prob, pred) in enumerate(zip(meta_probs, meta_preds)):
    result = "CAUGHT ✓" if pred == 1 else "MISSED ✗"
    print(f"  Meta sample {i+1:<9}        {prob:>8.4f}  {pred:>6}  {result}")

caught_probs = meta_probs[meta_preds == 1]
missed_probs = meta_probs[meta_preds == 0]

print(f"\n  Total metastases : {len(meta_probs)}")
print(f"  Caught           : {len(caught_probs)}  " +
      (f"prob range [{caught_probs.min():.3f} – {caught_probs.max():.3f}]"
       if len(caught_probs) > 0 else "none"))
print(f"  Missed           : {len(missed_probs)}  " +
      (f"prob range [{missed_probs.min():.3f} – {missed_probs.max():.3f}]"
       if len(missed_probs) > 0 else "none — all caught!"))
print(f"\n  Current threshold : {best_thresh:.3f}")
if len(missed_probs) > 0:
    print(f"  Lowest missed prob: {missed_probs.min():.4f}  "
          f"← threshold must go below this to catch all")
else:
    print(f"  All metastases caught at threshold {best_thresh:.3f}!")

print("\n  ── Threshold sweep ──────────────────────────────────────")
print(f"  {'Threshold':>10}  {'Sensitivity':>12}  {'Specificity':>12}"
      f"  {'FP':>4}  {'FN':>4}")
print("  " + "-" * 52)
for t in np.arange(0.05, 0.55, 0.05):
    _preds = (y_prob >= t).astype(int)
    _tp    = ((_preds == 1) & (y_test.values == 1)).sum()
    _tn    = ((_preds == 0) & (y_test.values == 0)).sum()
    _fp    = ((_preds == 1) & (y_test.values == 0)).sum()
    _fn    = ((_preds == 0) & (y_test.values == 1)).sum()
    _sens  = _tp / (_tp + _fn + 1e-8)
    _spec  = _tn / (_tn + _fp + 1e-8)
    marker = "  ← current" if abs(t - best_thresh) < 0.026 else ""
    print(f"  {t:>10.2f}  {_sens:>12.3f}  {_spec:>12.3f}"
          f"  {_fp:>4}  {_fn:>4}{marker}")
