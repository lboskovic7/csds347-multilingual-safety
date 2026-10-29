"""Translation with NLLB-200 (facebook/nllb-200-distilled-1.3B).

1) Validate translation quality (reproduce the NLLB paper's FLORES-200 evaluation):
     python translate.py --validate
   Translates the FLORES+ devtest split (1,012 sentences) of the 9 MultiJail languages into
   English and scores it against the human references with chrF++ and BLEU (sacrebleu, the same
   settings as fairseq/examples/nllb/evaluation: chrF++ = chrF with word order 2, BLEU = 13a).
   Runs twice: beam size 4 (the paper's setting) and greedy (the faster setting used for
   responses below). Writes results/translation_validation.csv.

2) Translate model responses into English so the judge can read them (MultiJail protocol):
     python translate.py --outputs outputs/multijail__none__Qwen2.5-7B-Instruct.jsonl ...
   Adds `response_en` and `response_lang` to every row and writes the file to outputs/translated/.
   Each response is translated line by line; lines already in English are kept as they are.
"""
import argparse, os, time
import pandas as pd
from common import LANGS, BASE, read_jsonl, write_jsonl

NLLB = "facebook/nllb-200-distilled-1.3B"
LANGID = {"en": "en", "zh": "zh", "it": "it", "vi": "vi", "ar": "ar", "ko": "ko", "th": "th",
          "bn": "bn", "sw": "sw", "jv": "jv"}


class Translator:
    def __init__(self, name=NLLB, batch_size=32, max_len=256):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name, dtype=torch.float16).cuda().eval()
        self.bs, self.max_len = batch_size, max_len
        self.eng = self.tok.convert_tokens_to_ids("eng_Latn")

    def __call__(self, texts, src, beams=1):
        """Translate a list of strings from NLLB language code `src` into English."""
        self.tok.src_lang = src
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [None] * len(texts)
        for s in range(0, len(order), self.bs):
            idx = order[s:s + self.bs]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                           truncation=True, max_length=self.max_len).to("cuda")
            with self.torch.no_grad():
                gen = self.model.generate(**enc, forced_bos_token_id=self.eng, num_beams=beams,
                                          max_new_tokens=int(self.max_len * 1.5))
            for i, t in zip(idx, self.tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = t
        return out


def validate(tr, out_dir, n):
    import sacrebleu
    from datasets import load_dataset

    def get(code):
        d = load_dataset("openlanguagedata/flores_plus", code, split="devtest").to_pandas()
        return d.sort_values("id").head(n) if n else d.sort_values("id")

    ref = get("eng_Latn").set_index("id")["text"]
    rows = []
    for lang, (level, flores, nllb) in LANGS.items():
        if lang == "en":
            continue
        d = get(flores)
        refs = [ref[i] for i in d["id"]]
        for beams in (4, 1):
            t = time.time()
            hyp = tr(list(d["text"]), nllb, beams=beams)
            chrf = sacrebleu.corpus_chrf(hyp, [refs], word_order=2).score
            bleu = sacrebleu.corpus_bleu(hyp, [refs], tokenize="13a").score
            rows.append({"language": lang, "resource_level": level, "nllb_code": nllb,
                         "beams": beams, "sentences": len(hyp), "chrF++": round(chrf, 1),
                         "BLEU": round(bleu, 1), "seconds": round(time.time() - t)})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "translation_validation.csv")
    df.to_csv(path, index=False)
    print(df.to_string(index=False), "\nSaved", path)


_LID = []


def detect(text):
    """Language of a text, restricted to the 10 MultiJail languages (py3langid)."""
    if not _LID:
        import py3langid
        py3langid.set_languages(list(LANGID.values()))
        _LID.append(py3langid)
    return _LID[0].classify(text)[0]


def needs_translation(line, lang):
    """Translate a response line unless it is already English or has (almost) no words."""
    letters = sum(c.isalpha() for c in line)
    if letters < 3:
        return False
    if lang in ("zh", "ar", "ko", "th", "bn") and all(ord(c) < 128 for c in line):
        return False  # non-Latin-script language but the line is plain ASCII: it is English
    return detect(line) != "en"


def translate_outputs(tr, paths, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for path in paths:
        rows = read_jsonl(path)
        target = os.path.join(out_dir, os.path.basename(path))
        # collect every non-English line of every response, grouped by source language
        jobs = {}
        for ri, r in enumerate(rows):
            resp = r.get("response") or ""
            r["response_lang"] = detect(resp) if resp.strip() else "empty"
            r["_lines"] = resp.split("\n")
            r["_en"] = list(r["_lines"])
            if r["language"] == "en":
                continue
            for li, line in enumerate(r["_lines"]):
                if needs_translation(line, r["language"]):
                    jobs.setdefault(LANGS[r["language"]][2], []).append((ri, li))
        t = time.time()
        for src, items in jobs.items():
            texts = [rows[ri]["_lines"][li] for ri, li in items]
            for (ri, li), en in zip(items, tr(texts, src)):
                rows[ri]["_en"][li] = en
        for r in rows:
            r["response_en"] = "\n".join(r.pop("_en"))
            r.pop("_lines")
        write_jsonl(target, rows)
        n = sum(len(v) for v in jobs.values())
        print(f"{os.path.basename(path)}: translated {n} lines in {time.time() - t:.0f}s -> {target}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--outputs", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=None, help="validation: sentences per language (default all)")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default=os.path.join(BASE, "outputs", "translated"))
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    tr = Translator(batch_size=args.batch_size)
    if args.validate:
        os.makedirs(args.results_dir, exist_ok=True)
        validate(tr, args.results_dir, args.n)
    if args.outputs:
        translate_outputs(tr, args.outputs, args.out_dir)


if __name__ == "__main__":
    main()
