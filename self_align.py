"""Cross-lingual self-alignment (our Demo 3 approach): teach a model to apply its OWN English
safety behavior in the 9 other MultiJail languages, with a small LoRA adapter.

Idea: Qwen2.5 refuses harmful requests in English (~2% attack success) but not in low-resource
languages (Swahili 46% for 7B). The model already knows how to refuse; it fails to recognise the
request as harmful in another language. So we fine-tune it on its own English refusals,
translated into each language, and on its own answers to harmless requests (so it does not learn
to refuse everything). No outside teacher model: distilling a stronger model's refusals made small
models less safe in prior work (Response-Based Knowledge Distillation..., NeurIPS 2025).
This is an open-model version of MultiJail's Self-Defence (which fine-tuned ChatGPT).

Steps (one command does all of them):
  1. Harmful prompts: AdvBench (walledai/AdvBench, 520). Never MultiJail; any prompt that also
     appears in JailbreakBench is dropped so the evaluation stays clean.
     Harmless prompts: Alpaca (tatsu-lab/alpaca), --n_benign without an `input` field.
  2. The model answers every English prompt. Harmful prompts it does not refuse in English are
     dropped (we only teach behavior the model already has).
  3. NLLB-200 translates the harmful prompts AND the model's English refusals into the 9 languages;
     it translates the harmless prompts too, and the model answers those itself in each language
     (its own answer = no change to its normal behavior).
  4. LoRA fine-tuning (rank 16, alpha 32, 1 epoch, lr 2e-4), loss on the response tokens only.

  python self_align.py --model Qwen/Qwen2.5-3B-Instruct --out_dir $BASE/adapters/qwen2.5-3b-selfalign

Writes the adapter, the training data (train.jsonl) and a short stats file (stats.json).
Then evaluate exactly like the baselines:
  python run_baseline.py --model Qwen/Qwen2.5-3B-Instruct --defense self_align --adapter <out_dir> ...
"""
import argparse, json, os, random, re, time
from common import LANGS, BASE, chat_text, generate, load_lm, check_rejection, read_jsonl, write_jsonl


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()


def load_prompts(n_benign, seed):
    from datasets import load_dataset
    adv = next(iter(load_dataset("walledai/AdvBench").values()))
    harmful = [r["prompt"] for r in adv]
    # keep the evaluation sets clean: drop anything that also appears in JBB or MultiJail
    jbb = {norm(r["Goal"]) for s in ("harmful", "benign")
           for r in load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split=s)}
    mj = {norm(r["en"]) for r in load_dataset("DAMO-NLP-SG/MultiJail", split="train")}
    before = len(harmful)
    harmful = [p for p in harmful if norm(p) not in jbb and norm(p) not in mj]
    print(f"AdvBench: {before} prompts, {before - len(harmful)} dropped as overlapping JBB/MultiJail")
    alp = load_dataset("tatsu-lab/alpaca", split="train")
    benign = [r["instruction"] for r in alp if not r["input"].strip() and len(r["instruction"]) < 300]
    random.Random(seed).shuffle(benign)
    return harmful, benign[:n_benign]


def translate_lines(tr, texts, tgt):
    """Translate multi-line texts line by line (NLLB is trained on single sentences/lines)."""
    lines = [t.split("\n") for t in texts]
    flat = [(i, j) for i, ls in enumerate(lines) for j, l in enumerate(ls) if sum(c.isalpha() for c in l) >= 3]
    out = tr([lines[i][j] for i, j in flat], "eng_Latn", tgt=tgt)
    for (i, j), t in zip(flat, out):
        lines[i][j] = t
    return ["\n".join(ls) for ls in lines]


def build_data(args, tok, model):
    from translate import Translator
    harmful, benign = load_prompts(args.n_benign, args.seed)
    langs = [l for l in LANGS if l != "en"]

    # 2. the model's own English behavior
    t = time.time()
    refusals = generate(model, tok, [chat_text(tok, p) for p in harmful], args.max_new_tokens, args.batch_size)
    kept = [(p, r) for p, r in zip(harmful, refusals) if check_rejection(r, exclude_lack_of_info=False)]
    print(f"model refused {len(kept)}/{len(harmful)} harmful prompts in English ({time.time() - t:.0f}s)")
    benign_en = generate(model, tok, [chat_text(tok, p) for p in benign], args.max_new_tokens, args.batch_size)

    rows = [{"kind": "harmful", "language": "en", "prompt": p, "response": r} for p, r in kept]
    rows += [{"kind": "benign", "language": "en", "prompt": p, "response": r} for p, r in zip(benign, benign_en)]

    # 3. translate into every language with NLLB (validated in Demo 2)
    tr = Translator(batch_size=args.batch_size)
    for lang in langs:
        code = LANGS[lang][2]
        hp = tr([p for p, _ in kept], "eng_Latn", tgt=code)
        hr = translate_lines(tr, [r for _, r in kept], code)
        rows += [{"kind": "harmful", "language": lang, "prompt": p, "response": r} for p, r in zip(hp, hr)]
        bp = tr(benign, "eng_Latn", tgt=code)
        rows += [{"kind": "benign", "language": lang, "prompt": p, "response": None} for p in bp]
        print(f"translated into {lang}", flush=True)
    del tr
    import torch; torch.cuda.empty_cache()

    # the model answers the translated harmless prompts itself (keeps its normal behavior)
    todo = [r for r in rows if r["response"] is None]
    answers = generate(model, tok, [chat_text(tok, r["prompt"]) for r in todo], args.max_new_tokens, args.batch_size)
    for r, a in zip(todo, answers):
        r["response"] = a
    return rows, {"advbench_kept": len(kept), "advbench_total": len(harmful), "benign_prompts": len(benign)}


def train(args, tok, model, rows):
    import torch
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    def encode(r):
        prompt = chat_text(tok, r["prompt"])
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        r_ids = tok(r["response"] + tok.eos_token, add_special_tokens=False).input_ids
        ids = (p_ids + r_ids)[:args.max_len]
        labels = ([-100] * len(p_ids) + r_ids)[:args.max_len]
        return ids, labels

    data = [encode(r) for r in rows]
    random.Random(args.seed).shuffle(data)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps = (len(data) + args.train_batch - 1) // args.train_batch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
    model.train()
    t = time.time()
    for step in range(steps):
        batch = data[step * args.train_batch:(step + 1) * args.train_batch]
        width = max(len(i) for i, _ in batch)
        ids = torch.full((len(batch), width), tok.pad_token_id)
        lab = torch.full((len(batch), width), -100)
        att = torch.zeros((len(batch), width), dtype=torch.long)
        for k, (i, l) in enumerate(batch):
            ids[k, :len(i)] = torch.tensor(i); lab[k, :len(l)] = torch.tensor(l); att[k, :len(i)] = 1
        dev = model.device if hasattr(model, "device") else "cuda"
        loss = model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        if step % 20 == 0 or step == steps - 1:
            print(f"step {step + 1}/{steps} loss {loss.item():.3f} ({time.time() - t:.0f}s)", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_benign", type=int, default=300)
    ap.add_argument("--max_new_tokens", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=48, help="generation / translation batch")
    ap.add_argument("--train_batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    import torch; torch.manual_seed(args.seed)

    tok, model = load_lm(args.model)
    data_path = os.path.join(args.out_dir, "train.jsonl")
    if os.path.exists(data_path):
        rows, stats = read_jsonl(data_path), {"reused_data": True}
        print(f"reusing {len(rows)} training examples from {data_path}")
    else:
        rows, stats = build_data(args, tok, model)
        write_jsonl(data_path, rows)
    stats.update({"examples": len(rows),
                  "harmful": sum(r["kind"] == "harmful" for r in rows),
                  "benign": sum(r["kind"] == "benign" for r in rows), **vars(args)})
    model = train(args, tok, model, rows)
    model.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved adapter to", args.out_dir)


if __name__ == "__main__":
    main()
