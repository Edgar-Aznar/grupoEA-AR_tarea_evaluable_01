#!/usr/bin/env python3
"""Análisis de expresión diferencial TCGA-BRCA con PyDESeq2 y GSEApy."""

from pathlib import Path
import sys
import platform
import importlib.metadata

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gseapy as gp
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# --------------------------- Configuración ---------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
RESULTS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

# Los archivos del proyecto son CSV normales, no .csv.gz
COUNTS_FILE = DATA / "counts.csv.gz"
META_FILE = DATA / "metadata.csv.gz"
CONDITION, PLATE = "site", "plate"
TEST, REFERENCE = "Primary_Solid_Tumor", "Solid_Tissue_Normal"
MIN_COUNT, MIN_SAMPLES = 10, 20
ALPHA, LFC = 0.05, 1
LIBRARIES = ["GO_Biological_Process_2023", "Reactome_2022"]

# --------------------------- 1. Lectura ------------------------------------
print("[1/8] Leyendo datos de entrada (TCGA-BRCA)...")
counts_df = pd.read_csv(COUNTS_FILE, index_col=0)
meta_df = pd.read_csv(META_FILE, index_col=0)
counts_df.index, counts_df.columns = counts_df.index.astype(str), counts_df.columns.astype(str)
meta_df.index = meta_df.index.astype(str)

# --------------------------- 2. Alineamiento -------------------------------
print("[2/8] Alineando muestras...")
samples = [s for s in counts_df.columns if s in meta_df.index]
meta = meta_df.loc[samples, [CONDITION, PLATE]].dropna().copy()
counts = counts_df.loc[:, meta.index].T.apply(pd.to_numeric).astype(int)
if TEST not in meta[CONDITION].astype(str).values or REFERENCE not in meta[CONDITION].astype(str).values:
    raise ValueError(f"Comprueba que '{CONDITION}' contiene {TEST} y {REFERENCE}.")
meta[CONDITION] = pd.Categorical(meta[CONDITION].astype(str),
    categories=[REFERENCE] + sorted(set(meta[CONDITION].astype(str)) - {REFERENCE}))
meta[PLATE] = meta[PLATE].astype("category")

# --------------------------- 3. Filtrado ----------------------------------
print("[3/8] Filtrando genes de baja expresión...")
counts = counts.loc[:, (counts >= MIN_COUNT).sum(axis=0) >= MIN_SAMPLES]
if counts.shape[1] == 0:
    raise ValueError("El filtro ha eliminado todos los genes.")

# --------------------------- 4. PyDESeq2 ----------------------------------
print("[4/8] Ejecutando PyDESeq2...")
dds = DeseqDataSet(counts=counts, metadata=meta,
                   design=f"~ {CONDITION} + {PLATE}", refit_cooks=True)
dds.deseq2()

# Gráfico de dispersión
v = dds.var
plt.figure(figsize=(7, 5))
plt.scatter(v["_normed_means"], v["dispersions"], s=8, alpha=.5)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Media normalizada"); plt.ylabel("Dispersión")
plt.title("Estimación de dispersión")
plt.tight_layout(); plt.savefig(RESULTS / "dispersion_plot.png", dpi=150); plt.close()

# --------------------------- 5. Resultados --------------------------------
print("[5/8] Extrayendo y guardando resultados...")
stats = DeseqStats(dds, contrast=[CONDITION, TEST, REFERENCE], alpha=ALPHA)
stats.summary()
res = stats.results_df.copy()
res["gene_id"] = res.index.astype(str)
res = res.reset_index(drop=True).sort_values("padj", na_position="last")
sig = res[res["padj"].notna() & (res["padj"] < ALPHA)].copy()
de = sig[ sig["log2FoldChange"].abs() > LFC ].copy()
up = de[de["log2FoldChange"] > 0].copy()
down = de[de["log2FoldChange"] < 0].copy()
top_up, top_down = up.head(10).copy(), down.head(10).copy()
de["top10_label"] = "other"
de.loc[de.gene_id.isin(top_up.gene_id), "top10_label"] = "top10_up"
de.loc[de.gene_id.isin(top_down.gene_id), "top10_label"] = "top10_down"
for table, name in [(res,"deseq2_all_results.tsv"), (sig,"deseq2_sig_padj_0.05.tsv"),
                    (de,"deseq2_de_padj_0.05_lfc_1.tsv"), (top_up,"top10_up_genes.tsv"),
                    (top_down,"top10_down_genes.tsv")]:
    table.to_csv(RESULTS / name, sep="\t", index=False)

# --------------------------- 6. Gráficos ----------------------------------
print("[6/8] Creando gráficos...")
vol = res.dropna(subset=["padj", "log2FoldChange"]).copy()
vol["estado"] = "No significativo"
vol.loc[vol.gene_id.isin(up.gene_id), "estado"] = "Up"
vol.loc[vol.gene_id.isin(down.gene_id), "estado"] = "Down"
vol["y"] = -np.log10(vol.padj.clip(lower=1e-300))
plt.figure(figsize=(7, 6))
for label, color in [("No significativo", "grey"), ("Up", "red"), ("Down", "blue")]:
    x = vol[vol.estado == label]
    plt.scatter(x.log2FoldChange, x.y, s=9, c=color, label=label, alpha=.6)
for _, r in vol.nsmallest(10, "padj").iterrows():
    plt.annotate(r.gene_id, (r.log2FoldChange, r.y), fontsize=6)
plt.axvline(-LFC, ls="--", c="black"); plt.axvline(LFC, ls="--", c="black")
plt.axhline(-np.log10(ALPHA), ls="--", c="black")
plt.xlabel("log2 fold change"); plt.ylabel("-log10(padj)"); plt.legend()
plt.tight_layout(); plt.savefig(RESULTS / "plot_volcano.png", dpi=150); plt.close()

ma = res.dropna(subset=["baseMean", "log2FoldChange"]) 
plt.figure(figsize=(7, 5))
plt.scatter(ma.baseMean.clip(lower=1e-3), ma.log2FoldChange, s=8, alpha=.5)
plt.xscale("log"); plt.axhline(0, c="black", lw=.8)
plt.xlabel("Media normalizada"); plt.ylabel("log2 fold change")
plt.title("MA plot"); plt.tight_layout()
plt.savefig(RESULTS / "plot_ma.png", dpi=150); plt.close()

combined = pd.concat([top_up, top_down]).sort_values("log2FoldChange")
plt.figure(figsize=(8, 6))
plt.barh(combined.gene_id, combined.log2FoldChange,
         color=["red" if x > 0 else "blue" for x in combined.log2FoldChange])
plt.axvline(0, c="black", lw=.8); plt.xlabel("log2 fold change")
plt.title("Top genes diferencialmente expresados")
plt.tight_layout(); plt.savefig(RESULTS / "top10_down_genes.png", dpi=150); plt.close()

if not sig.empty:
    r = sig.iloc[0]
    text = f"Top gene: {r.gene_id}\nlog2FoldChange: {r.log2FoldChange}\npadj: {r.padj}\n"
else:
    text = "No se encontraron genes significativos.\n"
(RESULTS / "top_gene_context.txt").write_text(text, encoding="utf-8")

# --------------------------- 7. Enriquecimiento ----------------------------
print("[7/8] Enriquecimiento GO y Reactome con GSEApy...")
background = counts.columns.astype(str).tolist()
for direction, genes in [("up", up.gene_id.tolist()), ("down", down.gene_id.tolist())]:
    for library in LIBRARIES:
        base = "go" if library.startswith("GO_") else "reactome"
        table_name = ("go_up_enrichment.tsv" if direction == "up" else "go_bp_down_enrichment.tsv") if base == "go" else f"reactome_{direction}.tsv"
        plot_name = f"plot_go_bp_{direction}.png" if base == "go" else f"plot_reactome_{direction}.png"
        try:
            if not genes:
                raise ValueError("No hay genes en esta lista.")
            enr = gp.enrichr(gene_list=genes, gene_sets=library, organism="human",
                             background=background, outdir=None, no_plot=True)
            result = enr.results
            result.to_csv(RESULTS / table_name, sep="\t", index=False)
            significant = result[result["Adjusted P-value"] < ALPHA].nsmallest(10, "Adjusted P-value")
            plt.figure(figsize=(8, max(4, .4 * len(significant))))
            if significant.empty:
                plt.text(.5, .5, "Sin términos significativos", ha="center")
                plt.axis("off")
            else:
                plt.barh(significant.Term.astype(str), -np.log10(significant["Adjusted P-value"].clip(lower=1e-300)))
                plt.xlabel("-log10(p ajustado)")
            plt.title(f"{library}: {direction}"); plt.tight_layout()
            plt.savefig(RESULTS / plot_name, dpi=150); plt.close()
        except Exception as e:
            print(f"  Aviso: {library} ({direction}) no se pudo completar: {e}")
            pd.DataFrame().to_csv(RESULTS / table_name, sep="\t", index=False)

# --------------------------- 8. Registro ----------------------------------
print("[8/8] Guardando resumen y versiones...")
versions = [f"Python: {sys.version}", f"Plataforma: {platform.platform()}"]
for package in ["pydeseq2", "gseapy", "pandas", "numpy", "matplotlib"]:
    versions.append(f"{package}: {importlib.metadata.version(package)}")
(LOGS / "session_info.txt").write_text("\n".join(versions), encoding="utf-8")
summary = [f"n_samples: {counts.shape[0]}", f"n_genes_after_filter: {counts.shape[1]}",
           f"n_significant_padj_lt_{ALPHA}: {len(sig)}", f"n_de_absLFC_gt_{LFC}: {len(de)}",
           f"n_up: {len(up)}", f"n_down: {len(down)}"]
(LOGS / "run_summary.txt").write_text("\n".join(summary), encoding="utf-8")
print("Análisis terminado. Revisa results/ y logs/.")

