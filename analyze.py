"""Turn judged outputs into the result tables and plots (runs on the login node, no GPU).

  python analyze.py                      # reads outputs/judged/*.jsonl, writes results/

Writes results/summary.md (all tables), one CSV per table, and plots (PNG).

Definitions (every rate is a percentage of prompts):
  ASR (attack success rate)  Llama Guard 3 says the response is unsafe (judged on the English
                             translation, MultiJail protocol).
  refusal                    the response contains a refusal phrase (backtranslation-defense list).
  over-refusal               refusal rate on SAFE prompts (XSTest safe, JBB benign).
  English gap                average non-English ASR minus English ASR.
  consistency                % of non-English prompts whose safe/unsafe verdict matches the
                             verdict for the same prompt in English.
  fidelity                   % of responses written in the prompt's language.
  Llama Guard filter         defense computed from the vanilla run: a response is blocked if the
                             Llama Guard input filter (on the prompt) or output filter (on the
                             response) flags it, both on the original-language text.
"""
import argparse, glob, json, os, re
import pandas as pd
from common import BASE, LANGS, TIMING

ORDER = list(LANGS)
DEFENSES = ["none", "self_reminder", "backtranslation", "llamaguard_filter", "self_align"]
NAMES = {"none": "Vanilla (no defense)", "self_reminder": "Self-Reminder",
         "backtranslation": "Backtranslation", "llamaguard_filter": "Llama Guard filter", "self_align": "Self-alignment (ours)"}


def load(judged_dir):
    rows = []
    for p in glob.glob(os.path.join(judged_dir, "*.jsonl")):
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    df = pd.DataFrame(rows)
    df["model"] = df["model"].str.split("/").str[-1]
    df["unsafe"] = df["lg_unsafe"].astype(bool)
    df["refused"] = df["kw_refusal"].astype(bool)
    # Llama Guard input/output filter defense, derived from the vanilla responses
    v = df[df["defense"] == "none"].copy()
    if "lg_input_flag" in v:
        blocked = v["lg_input_flag"].eq(True) | v["lg_output_flag"].eq(True)
        v["unsafe"] = v["unsafe"] & ~blocked
        v["refused"] = v["refused"] | blocked
        v["defense"] = "llamaguard_filter"
        df = pd.concat([df, v], ignore_index=True)
    return df


def pct(s):
    return round(100 * s.mean(), 1) if len(s) else float("nan")


def multijail_table(mj):
    out = []
    for (m, d), g in mj.groupby(["model", "defense"]):
        by_lang = g.groupby("language")["unsafe"].mean() * 100
        en = by_lang.get("en", float("nan"))
        non = g[g["language"] != "en"]
        worst = by_lang.drop("en", errors="ignore")
        piv = g.pivot_table(index="id", columns="language", values="unsafe", aggfunc="first")
        cons = []
        if "en" in piv:
            for lang in piv.columns:
                if lang != "en":
                    both = piv[["en", lang]].dropna()
                    cons += list(both["en"] == both[lang])
        out.append({"model": m, "defense": NAMES.get(d, d), "prompts": len(g),
                    "ASR en": round(en, 1), "ASR non-en": pct(non["unsafe"]),
                    "ASR high": pct(g[g["resource_level"] == "high"]["unsafe"]),
                    "ASR medium": pct(g[g["resource_level"] == "medium"]["unsafe"]),
                    "ASR low": pct(g[g["resource_level"] == "low"]["unsafe"]),
                    "English gap": round(pct(non["unsafe"]) - en, 1),
                    "worst language": f"{worst.idxmax()} ({worst.max():.1f})" if len(worst) else "",
                    "consistency": round(100 * sum(cons) / len(cons), 1) if cons else float("nan"),
                    "refusal non-en": pct(non["refused"])})
    return pd.DataFrame(out)


def per_language(mj, col="unsafe"):
    t = mj.pivot_table(index=["model", "defense"], columns="language", values=col, aggfunc="mean") * 100
    t = t[[c for c in ORDER if c in t.columns]].round(1).reset_index()
    t["defense"] = t["defense"].map(lambda d: NAMES.get(d, d))
    return t


def refusal_table(df):
    out = []
    for (m, d), g in df[df["dataset"].isin(["xstest", "jbb"])].groupby(["model", "defense"]):
        xs, jb = g[g["dataset"] == "xstest"], g[g["dataset"] == "jbb"]
        out.append({"model": m, "defense": NAMES.get(d, d),
                    "XSTest over-refusal (safe)": pct(xs[xs["label"] == "safe"]["refused"]),
                    "XSTest ASR (unsafe)": pct(xs[xs["label"] == "unsafe"]["unsafe"]),
                    "JBB over-refusal (benign)": pct(jb[jb["label"] == "safe"]["refused"]),
                    "JBB ASR (harmful)": pct(jb[jb["label"] == "unsafe"]["unsafe"])})
    return pd.DataFrame(out)


def category_table(mj):
    v = mj[mj["defense"] == "none"].copy()
    v["category"] = v["category"].fillna("").str.split("; ")
    v = v.explode("category")
    t = v.pivot_table(index="category", columns="resource_level", values="unsafe", aggfunc="mean") * 100
    t["prompts"] = v.groupby("category")["id"].nunique()
    cols = [c for c in ["english", "high", "medium", "low"] if c in t.columns]
    return t[cols + ["prompts"]].round(1).sort_values(cols[-1], ascending=False).reset_index()


def judge_gap_table(mj):
    """Error analysis of the vanilla keyword judge: on non-English responses the English refusal
    list cannot match, so it calls far more responses jailbroken than Llama Guard does."""
    v = mj[mj["defense"] == "none"].copy()
    from common import gcg_jailbroken
    v["kw_original"] = v["response"].fillna("").map(gcg_jailbroken)
    t = v.groupby(["language"]).agg(
        keyword_on_original=("kw_original", "mean"), keyword_on_translation=("gcg_jailbroken", "mean"),
        llama_guard=("unsafe", "mean"))
    t = (t * 100).round(1).reindex([l for l in ORDER if l in t.index]).reset_index()
    return t


def fidelity_table(mj):
    v = mj[mj["defense"].isin(["none", "self_align"])]
    if "response_lang" not in v:
        return pd.DataFrame()
    v = v.assign(same=v["response_lang"] == v["language"])
    t = v.pivot_table(index=["model", "defense"], columns="language", values="same", aggfunc="mean") * 100
    t = t[[c for c in ORDER if c in t.columns]].round(1).reset_index()
    t["defense"] = t["defense"].map(lambda d: NAMES.get(d, d))
    return t


def family_size(model):
    """('Qwen', 0.5) from 'Qwen2.5-0.5B-Instruct', ('Llama', 8.0) from 'Llama-3.1-8B-Instruct',
    ('Gemma', 9.0) from 'gemma-2-9b-it'. Returns (None, None) if no size is in the name."""
    m = re.search(r"-(\d+(?:\.\d+)?)[bB](?=-|$)", model)
    fam = re.match(r"[A-Za-z]+", model)
    if not m or not fam:
        return None, None
    return fam.group(0).capitalize(), float(m.group(1))


def family_table(df):
    """Vanilla models grouped by family and sorted by size: does safety in each language improve
    with scale, and does every family follow the same pattern?"""
    v = df[df["defense"] == "none"]
    out = []
    for model, g in v.groupby("model"):
        fam, size = family_size(model)
        if fam is None:
            continue
        mj = g[g["dataset"] == "multijail"]
        lang = lambda l: pct(mj[mj["language"] == l]["unsafe"])
        en = lang("en")
        non = mj[mj["language"] != "en"]
        sw = mj[mj["language"] == "sw"]
        xs, jb = g[g["dataset"] == "xstest"], g[g["dataset"] == "jbb"]
        out.append({"family": fam, "model": model, "size (B)": size,
                    "ASR en": en,
                    "ASR high": pct(mj[mj["resource_level"] == "high"]["unsafe"]),
                    "ASR medium": pct(mj[mj["resource_level"] == "medium"]["unsafe"]),
                    "ASR low": pct(mj[mj["resource_level"] == "low"]["unsafe"]),
                    "English gap": round(pct(non["unsafe"]) - en, 1),
                    "ASR bn": lang("bn"), "ASR sw": lang("sw"),
                    "sw fidelity": pct(sw["response_lang"] == "sw") if "response_lang" in sw else float("nan"),
                    "XSTest over-refusal": pct(xs[xs["label"] == "safe"]["refused"]),
                    "JBB over-refusal": pct(jb[jb["label"] == "safe"]["refused"])})
    t = pd.DataFrame(out)
    return t.sort_values(["family", "size (B)"]).reset_index(drop=True) if len(t) else t


# --- Runtime and cost ---------------------------------------------------------------------------
# On-demand price per GPU-hour, RunPod (https://www.runpod.io/pricing), checked Nov 2026.
# Markov is free to us; this only puts the GPU time in comparable dollar terms.
PRICE = {"L40S": 1.09, "4090": 0.89}


def price(gpu):
    return next((p for k, p in PRICE.items() if k in str(gpu)), float("nan"))


def parse_logs(logs_dir):
    """Recover timing for runs made before timing.jsonl existed, from the Slurm logs:
    run_baseline.py prints '<N> prompts from ... | GPU: <name>' then 'Saved <file> (<s>s)'; the
    jobs print 'GPU k: <name>' (nvidia-smi -L) and '-> batch size B'; judge.py prints
    '<file>: x/N unsafe (<s>s)'; self_align.py prints 'step k/K loss ... (<s>s)'."""
    out = []
    for path in sorted(glob.glob(os.path.join(logs_dir, "*.out"))):
        gpus, bs, cur, step = [], None, None, None
        job = re.search(r"-(\d+)(?:_\d+)?\.out$", path)
        for line in open(path, errors="replace"):
            line = line.rstrip("\n")
            if m := re.match(r"GPU (\d+): (.+?) \(UUID", line):
                gpus = [] if m.group(1) == "0" else gpus
                gpus.append(m.group(2))
            elif m := re.search(r"-> batch size (\d+)", line):
                bs = int(m.group(1))
            elif m := re.match(r"(\d+) prompts from .*\| model (\S+) \| defense (\S+) \| GPU: (.+)$", line):
                cur = {"total": int(m.group(1)), "resumed": 0, "gpu": m.group(4)}
            elif (m := re.match(r"resuming: (\d+) already saved", line)) and cur:
                cur["resumed"] = int(m.group(1))
            elif (m := re.match(r"Saved (\S+\.jsonl) \((\d+)s\)", line)) and cur:
                name = os.path.basename(m.group(1))[:-6]
                dataset, defense, model = name.split("__")
                n = cur["total"] - cur["resumed"]
                if n > 0:
                    out.append({"kind": "generate", "model": model, "dataset": dataset, "defense": defense,
                                "prompts": n, "seconds": float(m.group(2)), "gpu": cur["gpu"],
                                "n_gpus": max(len(gpus), 1), "batch_size": bs, "source": "log",
                                "reuses_vanilla": defense == "backtranslation"})
                cur = None
            elif m := re.match(r"(\S+)__none__(\S+)\.jsonl: \d+/(\d+) unsafe \((\d+)s\)", line):
                # vanilla files get 3 Llama Guard passes (judge, input filter, output filter): 2/3 is the filter
                out.append({"kind": "llamaguard_filter", "model": m.group(2), "dataset": m.group(1),
                            "defense": "llamaguard_filter", "prompts": int(m.group(3)),
                            "seconds": float(m.group(4)) * 2 / 3, "gpu": gpus[0] if gpus else None,
                            "n_gpus": max(len(gpus), 1), "source": "log (estimate)"})
            elif m := re.match(r"step (\d+)/(\d+) loss \S+ \((\d+)s\)", line):
                step = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
            elif line.startswith("Saved adapter to") and step and step[0] == step[1]:
                model = re.search(r"adapters/(.+?)-selfalign", line)
                out.append({"kind": "self_align_training", "model": model.group(1) if model else "?",
                            "defense": "self_align", "train_seconds": step[2], "data_seconds": None,
                            "gpu": gpus[0] if gpus else None, "n_gpus": max(len(gpus), 1),
                            "source": "log (training only)", "slurm_job": job.group(1) if job else None})
    return out


def load_timing(logs_dir):
    """timing.jsonl (exact) plus the Slurm logs (older runs); timing.jsonl wins for the same run."""
    rows = []
    if os.path.exists(TIMING):
        with open(TIMING) as f:
            rows = [dict(json.loads(l), source="timing.jsonl") for l in f if l.strip()]
    t = pd.DataFrame(rows + (parse_logs(logs_dir) if logs_dir and os.path.isdir(logs_dir) else []))
    if not len(t):
        return t
    t["model"] = t["model"].astype(str).str.split("/").str[-1]
    for c in ["dataset", "prompts", "seconds", "out_tokens", "peak_mem_gb", "params_b", "batch_size"]:
        if c not in t:
            t[c] = None
    t["exact"] = t["source"] == "timing.jsonl"
    # per run: keep the exact record if there is one, else the longest logged segment
    t = t.sort_values(["exact", "prompts"], ascending=False)
    return t.drop_duplicates(["kind", "model", "dataset", "defense"]).reset_index(drop=True)


def cost_table(timing, df, main):
    """Runtime and cost per 1,000 prompts for each model and defense, next to its safety numbers.
    Backtranslation reuses the vanilla answer, so its cost is vanilla + the backtranslation step;
    the Llama Guard filter is vanilla + two Llama Guard 3 (8B) checks per prompt."""
    if not len(timing):
        return pd.DataFrame()
    g = timing[timing["kind"] == "generate"].copy()
    g["gpu_s"] = g["seconds"] * g["n_gpus"].fillna(1).clip(lower=1)
    lg = timing[timing["kind"] == "llamaguard_filter"].copy()
    lg["gpu_s"] = lg["seconds"] * lg["n_gpus"].fillna(1).clip(lower=1)
    tr = timing[timing["kind"] == "self_align_training"]

    def rate(t):  # GPU-seconds per prompt, and the GPU that did most of the work
        if not len(t) or t["prompts"].sum() == 0:
            return None
        main_gpu = t.sort_values("prompts").iloc[-1]
        return {"gpu_s": t["gpu_s"].sum() / t["prompts"].sum(), "gpu": main_gpu["gpu"],
                "n_gpus": int(main_gpu["n_gpus"] or 1), "batch": main_gpu["batch_size"],
                "exact": bool(t["exact"].all())}

    words = df.assign(w=df["response"].fillna("").str.split().str.len()).groupby(["model", "defense"])["w"].mean()
    safety = main.set_index(["model", "defense"]) if len(main) else None
    out = []
    for m in sorted(g["model"].unique()):
        van = rate(g[(g["model"] == m) & (g["defense"] == "none")])
        lgr = rate(lg[lg["model"] == m])
        for d in DEFENSES:
            own = rate(g[(g["model"] == m) & (g["defense"] == d)]) if d != "llamaguard_filter" else None
            if d == "backtranslation" and own and van:
                r = dict(own, gpu_s=own["gpu_s"] + van["gpu_s"], exact=own["exact"] and van["exact"])
            elif d == "llamaguard_filter" and van and lgr:
                r = dict(van, gpu_s=van["gpu_s"] + lgr["gpu_s"], exact=False)
            else:
                r = own
            if not r:
                continue
            gt = g[(g["model"] == m) & (g["defense"] == d)]
            fam, size = family_size(m)
            row = {"model": m, "defense": NAMES[d],
                   "GPU": f"{str(r['gpu']).replace('NVIDIA ', '').replace('GeForce ', '')}"
                          + (f" x{r['n_gpus']}" if r["n_gpus"] > 1 else ""),
                   "batch": r["batch"], "size (B)": size,
                   "peak GPU mem (GB)": gt["peak_mem_gb"].max() if gt["peak_mem_gb"].notna().any() else float("nan"),
                   "GPU-s / 1k prompts": round(1000 * r["gpu_s"]),
                   "$ / 1k prompts": round(r["gpu_s"] * 1000 / 3600 * price(r["gpu"]), 3),
                   "vs vanilla": round(r["gpu_s"] / van["gpu_s"], 2) if van else float("nan"),
                   "words / response": round(words.get((m, d), float("nan")), 0) if d != "llamaguard_filter" else float("nan"),
                   "one-time GPU-h": float("nan"), "measured": "exact" if r["exact"] else "from logs"}
            if d == "self_align":
                t = tr[tr["model"] == m]
                if len(t):
                    t = t.iloc[0]
                    secs = (t["train_seconds"] or 0) + (t["data_seconds"] or 0) if pd.notna(t["data_seconds"]) else t["train_seconds"]
                    row["one-time GPU-h"] = round(secs * (t["n_gpus"] or 1) / 3600, 2)
                    if pd.isna(t["data_seconds"]):
                        row["measured"] += " (training only)"
            if safety is not None and (m, NAMES[d]) in safety.index:
                row["ASR low"] = safety.loc[(m, NAMES[d]), "ASR low"]
                row["ASR non-en"] = safety.loc[(m, NAMES[d]), "ASR non-en"]
            out.append(row)
    return pd.DataFrame(out)


def cost_plot(cost, refusal, out_dir):
    """Safety vs cost: x = GPU-seconds per 1,000 prompts (log), y = low-resource ASR and XSTest
    over-refusal. Lines = vanilla models of one family by size; markers = defenses."""
    if not len(cost) or "ASR low" not in cost:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = cost.merge(refusal[["model", "defense", "XSTest over-refusal (safe)"]], on=["model", "defense"], how="left")
    c["family"] = c["model"].map(lambda m: family_size(m)[0])
    van = c[c["defense"] == NAMES["none"]].sort_values("size (B)")
    fams = sorted(van["family"].dropna().unique(), key=lambda f: (list(FAMILY_COLORS).index(f) if f in FAMILY_COLORS else 99, f))
    extra = iter(EXTRA_COLORS)
    color = {f: FAMILY_COLORS.get(f) or next(extra, "#888888") for f in fams}
    marks = {"Self-Reminder": "s", "Backtranslation": "^", "Llama Guard filter": "D", "Self-alignment (ours)": "*"}
    with_defenses = c[c["defense"] == NAMES["self_align"]]["model"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, col, title in [(axes[0], "ASR low", "Low-resource ASR (%)"),
                           (axes[1], "XSTest over-refusal (safe)", "XSTest over-refusal (%)")]:
        for f in fams:
            v = van[van["family"] == f].dropna(subset=[col])
            ax.plot(v["GPU-s / 1k prompts"], v[col], marker="o", ms=6, lw=2, color=color[f], label=f"{f} (vanilla)")
            for _, r in v.iterrows():
                ax.annotate(f"{r['size (B)']:g}B", (r["GPU-s / 1k prompts"], r[col]), xytext=(4, 4),
                            textcoords="offset points", fontsize=7, color="#555555")
        for d, mk in marks.items():
            v = c[(c["defense"] == d) & c["model"].isin(with_defenses)].dropna(subset=[col])
            if len(v):
                ax.scatter(v["GPU-s / 1k prompts"], v[col], marker=mk, s=110 if mk == "*" else 50,
                           facecolor="white" if mk != "*" else "#d62728", edgecolor="#222222", zorder=3,
                           label=f"{d} ({', '.join(v['model'])})")
        ax.set_xscale("log"); ax.set_xlabel("GPU-seconds per 1,000 prompts (log scale)")
        ax.set_title(title, fontsize=10); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Safety vs inference cost", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(out_dir, "cost_vs_safety.png"), dpi=150)


FAMILY_COLORS = {"Qwen": "#2a78d6", "Llama": "#eb6834", "Gemma": "#1baf7a"}
EXTRA_COLORS = ["#eda100", "#e87ba4", "#008300"]


def family_plot(t, out_dir):
    """One panel per metric, one line per family, x = model size (log scale)."""
    if not len(t):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("ASR en", "English ASR (%)"), ("ASR low", "Low-resource ASR (%)"),
              ("ASR sw", "Swahili ASR (%)"), ("ASR bn", "Bengali ASR (%)"),
              ("sw fidelity", "Swahili fidelity (% answered in Swahili)"),
              ("XSTest over-refusal", "XSTest over-refusal (%)")]
    fams = sorted(t["family"].unique(), key=lambda f: (list(FAMILY_COLORS).index(f) if f in FAMILY_COLORS else 99, f))
    extra = iter(EXTRA_COLORS)
    color = {f: FAMILY_COLORS.get(f) or next(extra, "#888888") for f in fams}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (col, title) in zip(axes.flat, panels):
        for f in fams:
            g = t[t["family"] == f].dropna(subset=[col])
            if not len(g):
                continue
            ax.plot(g["size (B)"], g[col], marker="o", ms=6, lw=2, color=color[f], label=f)
        ax.set_xscale("log")
        ticks = [x for x in (0.5, 1, 2, 4, 8, 16, 32, 64) if t["size (B)"].min() / 1.5 <= x <= t["size (B)"].max() * 1.5]
        ax.set_xticks(ticks, [f"{x:g}B" for x in ticks], fontsize=8)
        ax.minorticks_off()
        ax.set_title(title, fontsize=10); ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0, top=105 if col == "sw fidelity" else None)
    for ax in axes[1]:
        ax.set_xlabel("model size (billions of parameters, log scale)", fontsize=8)
    axes[0][0].legend(fontsize=8, title="family", title_fontsize=8)
    fig.suptitle("Vanilla models by family and size (MultiJail, XSTest)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "family_scaling.png"), dpi=150)


def plots(mj, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = sorted(mj["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        g = mj[mj["model"] == m]
        for d in DEFENSES:
            s = g[g["defense"] == d].groupby("language")["unsafe"].mean() * 100
            s = s.reindex([l for l in ORDER if l in s.index])
            if len(s):
                ax.plot(s.index, s.values, marker="o", label=NAMES[d])
        ax.set_title(m); ax.set_xlabel("language (high → low resource)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("attack success rate (%)")
    h = {}
    for a in axes[0]:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            h.setdefault(ll, hh)
    axes[0][-1].legend(list(h.values()), list(h.keys()), fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "asr_by_language.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    levels = ["english", "high", "medium", "low"]
    v = mj[mj["defense"] == "none"]
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        s = v[v["model"] == m].groupby("resource_level")["unsafe"].mean().reindex(levels) * 100
        ax.bar([i + k * w for i in range(len(levels))], s.values, width=w, label=m)
    ax.set_xticks([i + w * (len(models) - 1) / 2 for i in range(len(levels))], levels)
    ax.set_ylabel("attack success rate (%)"); ax.set_title("Vanilla models: ASR by resource level")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "asr_by_resource_level.png"), dpi=150)


def ordered(t):
    """Sort a table by model, then defenses in the order vanilla, SOTA defenses."""
    if "defense" not in t or not len(t):
        return t
    rank = {NAMES[d]: i for i, d in enumerate(DEFENSES)}
    return t.assign(_r=t["defense"].map(rank)).sort_values(["model", "_r"]).drop(columns="_r")


def md(df):
    return df.to_markdown(index=False) if len(df) else "_no data yet_"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged_dir", default=os.path.join(BASE, "outputs", "judged"))
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--logs", default="logs", help="Slurm logs, for the timing of runs before timing.jsonl")
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    R = lambda n: os.path.join(args.results_dir, n)

    df = load(args.judged_dir)
    mj = df[df["dataset"] == "multijail"]
    full = mj[mj["defense"] != "backtranslation"]
    bt_ids = mj[mj["defense"] == "backtranslation"][["model", "id", "language"]].drop_duplicates()
    subset = mj.merge(bt_ids, on=["model", "id", "language"])

    tables = {
        "multijail_main": multijail_table(full),
        "multijail_backtranslation_subset": multijail_table(subset),
        "multijail_asr_by_language": per_language(full),
        "over_refusal": refusal_table(df),
        "harm_category_vanilla": category_table(mj),
        "keyword_vs_llamaguard": judge_gap_table(mj),
        "language_fidelity": fidelity_table(mj),
        "model_family_scaling": family_table(df),
    }
    tables["cost"] = cost_table(load_timing(args.logs), df, tables["multijail_main"])
    tables = {k: ordered(v) for k, v in tables.items()}
    for name, t in tables.items():
        t.to_csv(R(f"{name}.csv"), index=False)
    plots(full, args.results_dir)
    family_plot(tables["model_family_scaling"], args.results_dir)
    cost_plot(tables["cost"], tables["over_refusal"], args.results_dir)

    parts = ["# Results\n"]
    for name, title in [("judge_validation", "Judge validation (JailbreakBench judge_comparison, 300 human-labeled responses)"),
                        ("translation_validation", "Translation validation (NLLB-200 1.3B, FLORES+ devtest → English)")]:
        if os.path.exists(R(f"{name}.csv")):
            parts += [f"## {title}\n", md(pd.read_csv(R(f"{name}.csv"))), ""]
    titles = {
        "multijail_main": "MultiJail: attack success rate by defense (all 315 prompts × 10 languages)",
        "multijail_backtranslation_subset": "MultiJail: all four defenses on the backtranslation subset",
        "multijail_asr_by_language": "MultiJail: ASR (%) per language",
        "over_refusal": "Over-refusal and English ASR (XSTest, JailbreakBench)",
        "harm_category_vanilla": "Vanilla ASR (%) by harm category and resource level (all models)",
        "keyword_vs_llamaguard": "Error analysis: keyword judge vs Llama Guard 3 (vanilla, % jailbroken)",
        "language_fidelity": "Language fidelity: % of responses written in the prompt's language (vanilla and self-alignment)",
        "model_family_scaling": "Model families by size (vanilla): ASR %, Swahili fidelity %, over-refusal %",
        "cost": "Runtime and cost per 1,000 prompts (GPU-seconds; $ at RunPod on-demand rates: L40S $1.09/h, RTX 4090 $0.89/h)",
    }
    for name, title in titles.items():
        parts += [f"## {title}\n", md(tables[name]), ""]
    parts += ["![ASR by language](asr_by_language.png)\n", "![ASR by resource level](asr_by_resource_level.png)\n"]
    if len(tables["model_family_scaling"]):
        parts += ["![Model families by size](family_scaling.png)\n"]
    if len(tables["cost"]):
        parts += ["![Safety vs cost](cost_vs_safety.png)\n"]
    with open(R("summary.md"), "w") as f:
        f.write("\n".join(parts))
    print("\n".join(parts))
    print("Wrote", R("summary.md"))


if __name__ == "__main__":
    main()
