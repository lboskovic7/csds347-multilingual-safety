"""Run one model on one dataset and save its raw responses (no defense = baseline).

Examples:
  python run_baseline.py --model Qwen/Qwen2.5-7B-Instruct --limit 5                      # JBB harmful, quick test
  python run_baseline.py --model Qwen/Qwen2.5-7B-Instruct --dataset jbb --split benign
  python run_baseline.py --model Qwen/Qwen2.5-7B-Instruct --dataset multijail --languages en sw jv
  python run_baseline.py --model Qwen/Qwen2.5-7B-Instruct --dataset xstest
  python run_baseline.py --model Qwen/Qwen2.5-7B-Instruct --dataset flores --languages zh sw
  python run_baseline.py --model Qwen/Qwen2.5-7B-Instruct --prompts my_prompts.csv       # any CSV

Every prompt becomes one row with at least: id, dataset, language, prompt.
Extra fields (resource level, harm category, safe/unsafe label, ...) are copied to the output.
Output: <out_dir>/<dataset>__<model>.jsonl, one JSON line per prompt.
"""
import argparse, ast, json, math, os, time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# MultiJail languages: resource level (from the MultiJail paper) and FLORES+ code
LANGS = {
    "en": ("english", "eng_Latn"),
    "zh": ("high", "cmn_Hans"), "it": ("high", "ita_Latn"), "vi": ("high", "vie_Latn"),
    "ar": ("medium", "arb_Arab"), "ko": ("medium", "kor_Hang"), "th": ("medium", "tha_Thai"),
    "bn": ("low", "ben_Beng"), "sw": ("low", "swh_Latn"), "jv": ("low", "jav_Latn"),
}
TRANSLATE = "Translate the following text into English. Reply with the translation only.\n\n{text}"


def load_jbb(split):
    from datasets import load_dataset
    rows = []
    for s in (["harmful", "benign"] if split == "both" else [split]):
        for i, r in enumerate(load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split=s)):
            rows.append({"id": f"jbb-{s}-{i}", "dataset": "jbb", "language": "en",
                         "resource_level": "english",
                         "label": "unsafe" if s == "harmful" else "safe",
                         "category": r["Category"], "prompt": r["Goal"]})
    return pd.DataFrame(rows)


def load_multijail(languages):
    from datasets import load_dataset
    ds = load_dataset("DAMO-NLP-SG/MultiJail", split="train").to_pandas()
    rows = []
    for lang in languages:
        for _, r in ds.iterrows():
            text = r.get(lang)
            if not isinstance(text, str) or not text.strip():
                continue
            try:
                tags = "; ".join(ast.literal_eval(r["tags"]))
            except (ValueError, SyntaxError):
                tags = str(r["tags"])
            rows.append({"id": f"mj-{r['id']}", "dataset": "multijail", "language": lang,
                         "resource_level": LANGS[lang][0], "label": "unsafe",
                         "category": tags, "en_prompt": r["en"], "prompt": text})
    return pd.DataFrame(rows)


def load_xstest():
    from datasets import load_dataset
    dd = load_dataset("walledai/XSTest")
    ds = next(iter(dd.values())).to_pandas()
    return pd.DataFrame({"id": [f"xs-{i}" for i in range(len(ds))], "dataset": "xstest",
                         "language": "en", "resource_level": "english",
                         "label": ds["label"], "category": ds.get("type", ""),
                         "prompt": ds["prompt"]})


def load_flores(languages, n):
    """Benign comprehension check: the model translates FLORES+ dev sentences into English."""
    from datasets import load_dataset
    def get(code):
        return load_dataset("openlanguagedata/flores_plus", code, split="dev").to_pandas() \
            .sort_values("id").head(n)
    ref = get("eng_Latn").set_index("id")["text"]
    rows = []
    for lang in languages:
        if lang == "en":
            continue
        for _, r in get(LANGS[lang][1]).iterrows():
            rows.append({"id": f"fl-{r['id']}", "dataset": "flores", "language": lang,
                         "resource_level": LANGS[lang][0], "label": "safe",
                         "source_text": r["text"], "reference_en": ref.get(r["id"], ""),
                         "prompt": TRANSLATE.format(text=r["text"])})
    return pd.DataFrame(rows)


def load_prompts(args):
    if args.prompts:
        df = pd.read_csv(args.prompts)
        if "dataset" not in df:
            df["dataset"] = os.path.splitext(os.path.basename(args.prompts))[0]
        if "language" not in df:
            df["language"] = "unknown"
        name = df["dataset"].iloc[0]
    elif args.dataset == "jbb":
        df, name = load_jbb(args.split), f"jbb-{args.split}"
    elif args.dataset == "multijail":
        df, name = load_multijail(args.languages), "multijail"
    elif args.dataset == "xstest":
        df, name = load_xstest(), "xstest"
    else:
        df, name = load_flores(args.languages, args.flores_n), "flores"
    if args.limit:  # first N prompts per language (and per safe/unsafe label), so all are covered
        keys = ["language", "label"] if "label" in df else ["language"]
        df = df.groupby(keys, sort=False).head(args.limit)
    return df.reset_index(drop=True), name


def clean(v):
    if hasattr(v, "item"):
        v = v.item()
    return None if isinstance(v, float) and math.isnan(v) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="jbb", choices=["jbb", "multijail", "xstest", "flores"])
    ap.add_argument("--split", default="harmful", choices=["harmful", "benign", "both"],
                    help="JBB only")
    ap.add_argument("--languages", nargs="+", default=list(LANGS), choices=list(LANGS),
                    help="MultiJail / FLORES only (default: all 10)")
    ap.add_argument("--flores_n", type=int, default=200, help="FLORES sentences per language")
    ap.add_argument("--prompts", default=None, help="CSV with id, language, prompt (overrides --dataset)")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--limit", type=int, default=None, help="max prompts per language and label")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    df, name = load_prompts(args)
    langs = ", ".join(f"{k}={v}" for k, v in df["language"].value_counts(sort=False).items())
    print(f"{len(df)} prompts from {name} ({langs}) | model {args.model} | "
          f"GPU: {torch.cuda.get_device_name(0)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{name}__{args.model.split('/')[-1]}.jsonl")
    rows = df.to_dict("records")
    start = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                             tokenize=False, add_generation_prompt=True)
                     for r in batch]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            replies = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
            for r, reply in zip(batch, replies):
                r = {k: clean(v) for k, v in r.items()}
                f.write(json.dumps({**r, "model": args.model, "response": reply},
                                   ensure_ascii=False) + "\n")
            f.flush()
            print(f"{i + len(batch)}/{len(rows)} done ({time.time() - start:.0f}s)")

    print("Saved", out_path)


if __name__ == "__main__":
    main()
