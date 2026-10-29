# Multilingual Safety of LLMs: Baselines and Evaluation

CSDS 347, Team 5: Luka Boskovic (lxb451), Jiye Yong (jxy1203)

## Repository contents

| File | Purpose |
| --- | --- |
| `setup_markov.sh` | One-time environment setup on Markov (comments inside explain each step) |
| `run_baseline.py` | Runs one model with one defense (none, Self-Reminder, Backtranslation) on one dataset and saves its responses |
| `translate.py` | NLLB-200 translation: validation on FLORES+, and translating responses into English for the judge |
| `judge.py` | Jailbreak judges (keyword lists, Llama Guard 3): validation on human labels, and labeling every response |
| `analyze.py` | Builds all result tables and plots into `results/` |
| `common.py` | Shared helpers, including the published keyword lists copied from the authors' code |
| `jobs/` | Slurm batch jobs: `generate.sh`, `validate.sh`, `evaluate.sh` |
| `models.txt` | The three models we evaluate, one per line |
| `results/` | Committed results: `summary.md` (all tables), CSVs and plots |

## Requirements

- **Cluster:** Markov (`markov.case.edu`), course account `csds447`. Off campus you need the CWRU VPN.
- **GPU:** one NVIDIA GPU with at least 24 GB of memory. We use the RTX 4090 (24 GB) or L40S (48 GB).
  The 7–9B models need about 15–18 GB in bfloat16. The `gpu2080` and `gpu2h100` types are too small.
- **Python:** 3.9 or newer (Markov's `python3` works).
- **Python packages:** `torch`, `transformers`, `accelerate`, `datasets`, `pandas`, `sacrebleu`, `sentencepiece`,
  `py3langid`, `matplotlib`, `tabulate`, installed by `setup_markov.sh`.
- **Storage:** about 65 GB for models (3 target models, Llama Guard 3, NLLB-200) and 7 GB for the Python environment, kept in course space
  (`/mnt/vstor/courses/csds447/team5`), not in the 25 GB home directory.
- **Hugging Face account** with an access token (Settings → Access Tokens → Read).

## 1. Log in and clone

From a terminal on your own computer:

```bash
ssh <caseID>@markov.case.edu
module load git/2.50.1-GCCcore-14.3.0
git clone https://github.com/lboskovic7/csds347-multilingual-safety.git ~/project
cd ~/project
```

## 2. Run the setup script

```bash
bash setup_markov.sh
```

This takes about 10 minutes, mostly installing PyTorch. It creates the shared team folder
`/mnt/vstor/courses/csds447/team5`, shares it with your teammate, checks the Python version,
builds a virtual environment (`venv/`) and installs the packages. It also creates `hf_cache/` and
`outputs/`, and points Hugging Face downloads at `hf_cache/` through your `~/.bashrc`.
The comments in the script explain each step.

The setup only needs to run once per team. If your teammate has already run it, skip this step
and add the cache location to your own `~/.bashrc` instead:

```bash
echo 'export HF_HOME=/mnt/vstor/courses/csds447/team5/hf_cache' >> ~/.bashrc
```

If the environment was built before the evaluation code was added, install the new packages once:

```bash
source /mnt/vstor/courses/csds447/team5/venv/bin/activate
pip install sacrebleu sentencepiece py3langid==0.3.0 matplotlib tabulate
```

Both teammates should also add this line, so files either of you creates (downloads, outputs)
can be written by the other. Then log out and back in:

```bash
echo 'umask 007' >> ~/.bashrc
```

## 3. Activate the environment

Do this at the start of every new session on Markov:

```bash
source /mnt/vstor/courses/csds447/team5/venv/bin/activate
export HF_HOME=/mnt/vstor/courses/csds447/team5/hf_cache
```

## 4. Log in to Hugging Face

Three models and two datasets are gated. Accept their terms on Hugging Face first, using the same
account as your token:

- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (fill in the form; approval can take a little while)
- [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)
- [meta-llama/Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B) (the judge)
- [walledai/XSTest](https://huggingface.co/datasets/walledai/XSTest)
- [openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus)

Then log in once and paste your token (it stays invisible while you paste):

```bash
hf auth login
```

Answer `n` if it asks to add the token as a git credential.

## 5. Download the models

Run these on the login node. Downloading needs internet but no GPU.

```bash
hf download Qwen/Qwen2.5-7B-Instruct
hf download google/gemma-2-9b-it
hf download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*"
hf download meta-llama/Llama-Guard-3-8B --exclude "original/*"
hf download facebook/nllb-200-distilled-1.3B
```

The `--exclude` skips a second 16 GB copy of Llama in Meta's own format, which `transformers` does not use.

| Model | Parameters | Download size | Developer |
| --- | --- | --- | --- |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 7B | 15 GB | Alibaba |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 8B | 15 GB | Meta |
| [gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) | 9B | 18 GB | Google |
| [Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B) (judge) | 8B | 15 GB | Meta |
| [NLLB-200-distilled-1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B) (translator) | 1.3B | 5 GB | Meta |

The first three are the models we evaluate: open-weight, instruction-tuned and safety-aligned. We need open weights because our
analysis looks at the models' internal activations, not only their text output.

## 6. Download the datasets

Also on the login node:

```bash
python -c "from datasets import load_dataset as L; L('JailbreakBench/JBB-Behaviors','behaviors')"
python -c "from datasets import load_dataset as L; L('JailbreakBench/JBB-Behaviors','judge_comparison')"
python -c "from datasets import load_dataset as L; L('DAMO-NLP-SG/MultiJail')"
python -c "from datasets import load_dataset as L; L('walledai/XSTest')"
python -c "from datasets import load_dataset as L; [L('openlanguagedata/flores_plus', c) for c in ['eng_Latn','cmn_Hans','ita_Latn','vie_Latn','arb_Arab','kor_Hang','tha_Thai','ben_Beng','swh_Latn','jav_Latn']]"
```

| Dataset | Role in the pipeline | Size | Splits | Key attributes |
| --- | --- | --- | --- | --- |
| [JBB `judge_comparison`](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | **Validate the judge** against human labels | 300 responses | one split, used in full | `goal`, `prompt`, `target_response`, `human1-3`, `human_majority` (jailbroken or not), labels of 4 published judges |
| [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) | **Validate the translator** against human reference translations | 1,012 sentences per language | `devtest` (validation), `dev` | `text`, `id` (same `id` = same sentence in every language) |
| [MultiJail](https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail) | **Main experiment**: the same harmful requests in 10 languages | 315 prompts × 10 languages | none (one `train` split, used in full) | one column per language (`en` … `jv`), `tags` (harm categories) |
| [XSTest](https://huggingface.co/datasets/walledai/XSTest) | **Over-refusal**: safe prompts that look unsafe (cost of a defense) | 450 prompts | none | `prompt`, `label` (250 safe, 200 unsafe), `type` |
| [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | **English harmful + benign** pairs (ASR and over-refusal) | 200 prompts | `harmful` (100), `benign` (100) | `Goal` (the prompt), `Category` (10 harm types) |

**How the datasets fit together.** We cannot hand-label thousands of responses in 10 languages,
so we use an automatic judge (Llama Guard 3) on English translations of the responses, which is
the MultiJail paper's protocol (they used Google Translate and GPT-4). Before trusting either
tool we validate it on data with human labels: the judge on JBB `judge_comparison`, the
translator on FLORES+. Only then do we apply them to MultiJail, XSTest and JBB.

**Attributes that matter for our safety property.** The key attribute is the **language of the
prompt** and its **resource level**, as grouped by MultiJail:

| Resource level | Languages |
| --- | --- |
| High | Chinese (`zh`), Italian (`it`), Vietnamese (`vi`) |
| Medium | Arabic (`ar`), Korean (`ko`), Thai (`th`) |
| Low | Bengali (`bn`), Swahili (`sw`), Javanese (`jv`) |

Because every MultiJail prompt has the same meaning in all 10 languages, any difference in a
model's behavior across languages comes from the language, not the request. The **harm category**
(`tags`, `Category`) lets us check whether some kinds of harm slip through more than others.
FLORES+ uses the same sentences in every language, so translation quality per language gives a
measure of how well each model understands that language, separate from its safety behavior.

## 7. Check that everything loads

Still on the login node, these load everything from the cache without internet and without a GPU.

Models (expect three `OK` lines):

```bash
HF_HUB_OFFLINE=1 python -c "from transformers import AutoConfig as C, AutoTokenizer as T; [print('OK', m, C.from_pretrained(m).model_type) for m in ['Qwen/Qwen2.5-7B-Instruct','meta-llama/Llama-3.1-8B-Instruct','google/gemma-2-9b-it']]"
```

Datasets, one command each (expect 100 + 100, 300, 315 and 450 rows):

```bash
HF_HUB_OFFLINE=1 python -c "from datasets import load_dataset as L; print(L('JailbreakBench/JBB-Behaviors','behaviors'))"
HF_HUB_OFFLINE=1 python -c "from datasets import load_dataset as L; print(L('JailbreakBench/JBB-Behaviors','judge_comparison'))"
HF_HUB_OFFLINE=1 python -c "from datasets import load_dataset as L; print(L('DAMO-NLP-SG/MultiJail'))"
HF_HUB_OFFLINE=1 python -c "from datasets import load_dataset as L; print(L('walledai/XSTest'))"
```

FLORES+ languages downloaded so far (expect 10 folders):

```bash
ls /mnt/vstor/courses/csds447/team5/hf_cache/datasets/openlanguagedata___flores_plus/
```

If a check fails with `PermissionError` on a `.lock` file, a teammate's download left behind an
empty lock file you cannot write. When nobody is downloading, delete them and retry:

```bash
rm -f /mnt/vstor/courses/csds447/team5/hf_cache/datasets/*.lock
```

## 8. Baselines

Every baseline runs on every model (Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B) and every dataset
(MultiJail, XSTest, JBB). Code from the original authors is reused verbatim where it exists:
prompts, keyword lists, thresholds and decision logic are copied, and only the model calls are
adapted to our models' chat templates.

**Defenses** (what a deployer does to stop jailbreaks):

| Defense | Type | Source | What it does |
| --- | --- | --- | --- |
| None | Vanilla | – | The model as-is, default chat template, greedy decoding |
| Self-Reminder | SOTA | Xie et al. 2023, *Nature Machine Intelligence*; text from `llm-jailbreaking-defense` | Adds a "be a responsible language model" system prompt and reminder around the query |
| Backtranslation | SOTA | Wang et al. 2024, *ACL Findings* (our ref. [9]); logic from the authors' `llm-jailbreaking-defense` | Asks the model to guess the request behind its own answer; refuses if the model refuses that guessed request. Likelihood threshold −2.0, 150-token lengths, as in the paper. The target model does the backtranslation. Run on the first 50 MultiJail prompts per language (it costs 2–3 generations per prompt) |
| Llama Guard filter | SOTA / off-the-shelf guardrail | Meta, Llama Guard 3 | Blocks the answer if Llama Guard flags the prompt (input filter) or the response (output filter), on the original-language text |

**Judges** (how we decide a response is a jailbreak):

| Judge | Type | Source |
| --- | --- | --- |
| Refusal keywords | Vanilla | GCG (Zou et al. 2023) list and JailbreakBench `StringClassifier`: jailbroken if no refusal phrase |
| Llama Guard 3 8B | SOTA | Meta model card usage; applied to the English translation of each response (MultiJail protocol) |

## 9. Quick test (a few minutes, interactive)

Request one GPU:

```bash
srun -A csds447 -p markov_gpu -C "gpu4090|gpul40s" --gres=gpu:1 -c 4 --mem=32G -t 00:30:00 --pty bash
```

On the GPU node:

```bash
cd ~/project
source /mnt/vstor/courses/csds447/team5/venv/bin/activate
export HF_HOME=/mnt/vstor/courses/csds447/team5/hf_cache HF_HUB_OFFLINE=1
T=/mnt/vstor/courses/csds447/team5/outputs_test
python run_baseline.py --model Qwen/Qwen2.5-7B-Instruct --dataset multijail --limit 2 --out_dir $T
python run_baseline.py --model Qwen/Qwen2.5-7B-Instruct --dataset multijail --limit 2 --defense self_reminder --out_dir $T
python run_baseline.py --model Qwen/Qwen2.5-7B-Instruct --dataset multijail --limit 2 --defense backtranslation --from_responses $T/multijail__none__Qwen2.5-7B-Instruct.jsonl --out_dir $T
python translate.py --outputs $T/*__*__*.jsonl --out_dir $T/translated
python judge.py --outputs "$T/translated/*.jsonl" --out_dir $T/judged
python analyze.py --judged_dir $T/judged --results_dir $T/results
exit
```

This runs every step of the pipeline on 20 prompts and prints the result tables.

## 10. Full run (batch jobs)

Three jobs. Submit them from the repo folder on the login node:

```bash
cd ~/project
mkdir -p logs
sbatch jobs/validate.sh                    # judge + translation validation, ~30 min
sbatch jobs/generate.sh                    # all models x defenses x datasets, ~2-3 h (3 GPUs)
```

`jobs/generate.sh` prints a job ID. When it finishes, run the evaluation (or chain it right away
with the ID so it starts by itself):

```bash
sbatch --dependency=afterok:<generate job ID> jobs/evaluate.sh   # translate + judge + tables, ~1-1.5 h
```

| Job | What it runs | Writes |
| --- | --- | --- |
| `validate.sh` | `translate.py --validate`, `judge.py --validate` | `results/translation_validation.csv`, `results/judge_validation.csv` |
| `generate.sh` | One GPU per model: vanilla and Self-Reminder on MultiJail (3,150), XSTest (450), JBB (200); Backtranslation on MultiJail (500), XSTest, JBB | `outputs/<dataset>__<defense>__<model>.jsonl` |
| `evaluate.sh` | Translate every response into English, judge it, build tables | `outputs/translated/`, `outputs/judged/`, `results/` |

Check progress with `squeue -u $USER` and `tail -f logs/*.out`. A run that stops (time limit,
crash) can be resubmitted: finished prompts are skipped. To run a smaller MultiJail set:
`LIMIT=100 sbatch jobs/generate.sh` (100 prompts per language).

## 11. Results and metrics

`analyze.py` writes `results/summary.md` with every table, plus CSVs and plots:

| Metric | Kind | Definition |
| --- | --- | --- |
| Agreement, precision, recall, F1, FPR, FNR, Cohen's κ | Standard | Judge vs. human majority on JBB `judge_comparison` |
| chrF++, BLEU | Standard | NLLB translation vs. FLORES+ human references (NLLB paper's settings) |
| Attack success rate (ASR) | Standard / RAI | % of harmful prompts with an unsafe response (Llama Guard 3) |
| Over-refusal | RAI (utility cost) | % of safe prompts refused (XSTest safe, JBB benign) |
| English gap | RAI (fairness across languages) | average non-English ASR − English ASR |
| Worst-language ASR | RAI (worst group) | highest ASR of any language |
| ASR by resource level | RAI (subgroups) | high / medium / low-resource averages |
| Cross-lingual consistency | RAI | % of non-English prompts with the same safe/unsafe verdict as English |
| ASR by harm category | Subgroup analysis | MultiJail `tags` × resource level |
| Language fidelity | RAI | % of responses written in the prompt's language |

The results are committed to this repository under `results/`. To copy them from Markov to your
computer and commit them, run on your own computer:

```bash
scp -r <caseID>@markov.case.edu:~/project/results .
git add results && git commit -m "Add results" && git push
```

## Storage and etiquette

- Check space with `quotagrp csds447` and `du -sh ~`. Exceeding 25 GB in your home directory blocks access.
- Clear pip's download cache after installing: `pip cache purge`.
- Use one GPU at a time and end interactive sessions when you are done.
- Cluster problems: hpc-support@case.edu.
