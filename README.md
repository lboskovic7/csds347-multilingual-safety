# Multilingual Safety of LLMs: Setup and Baselines

CSDS 347, Team 5: Luka Boskovic (lxb451), Jiye Yong (jxy1203)

## Repository contents

| File | Purpose |
| --- | --- |
| `setup_markov.sh` | One-time environment setup on Markov (comments inside explain each step) |
| `run_baseline.py` | Runs one model on one dataset (JBB, MultiJail, XSTest, FLORES+ or any CSV) and saves its responses |
| `run_models.sh` | Slurm batch job that runs `run_baseline.py` for every model in `models.txt` |
| `models.txt` | The three models we evaluate, one per line |

## Requirements

- **Cluster:** Markov (`markov.case.edu`), course account `csds447`. Off campus you need the CWRU VPN.
- **GPU:** one NVIDIA GPU with at least 24 GB of memory. We use the RTX 4090 (24 GB) or L40S (48 GB).
  The 7–9B models need about 15–18 GB in bfloat16. The `gpu2080` and `gpu2h100` types are too small.
- **Python:** 3.9 or newer (Markov's `python3` works).
- **Python packages:** `torch`, `transformers`, `accelerate`, `datasets`, `pandas`, installed by `setup_markov.sh`.
- **Storage:** about 47 GB for models and 7 GB for the Python environment, kept in course space
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

Two models and two datasets are gated. Accept their terms on Hugging Face first, using the same
account as your token:

- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (fill in the form; approval can take a little while)
- [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)
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
```

The `--exclude` skips a second 16 GB copy of Llama in Meta's own format, which `transformers` does not use.

| Model | Parameters | Download size | Developer |
| --- | --- | --- | --- |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 7B | 15 GB | Alibaba |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 8B | 15 GB | Meta |
| [gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) | 9B | 18 GB | Google |

All three are open-weight, instruction-tuned and safety-aligned. We need open weights because our
analysis looks at the models' internal activations, not only their text output.

## 6. Download the datasets

Also on the login node:

```bash
python -c "from datasets import load_dataset as L; L('JailbreakBench/JBB-Behaviors','behaviors')"
python -c "from datasets import load_dataset as L; L('DAMO-NLP-SG/MultiJail')"
python -c "from datasets import load_dataset as L; L('walledai/XSTest')"
python -c "from datasets import load_dataset as L; [L('openlanguagedata/flores_plus', c) for c in ['eng_Latn','cmn_Hans','ita_Latn','vie_Latn','arb_Arab','kor_Hang','tha_Thai','ben_Beng','swh_Latn','jav_Latn']]"
```

| Dataset | Role | Size | Splits | Key attributes |
| --- | --- | --- | --- | --- |
| [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | English test set for the pipeline | 200 prompts | `harmful` (100), `benign` (100) | `Goal` (the prompt), `Target` (an affirmative response), `Category` (10 harm types) |
| [MultiJail](https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail) | Main harmful prompts across languages | 315 prompts × 10 languages | none (one `train` split, used in full) | one column per language (`en`, `zh`, `it`, `vi`, `ar`, `ko`, `th`, `bn`, `sw`, `jv`), `tags` (harm categories) |
| [XSTest](https://huggingface.co/datasets/walledai/XSTest) | Over-refusal: safe prompts that look unsafe | 450 prompts | none | `prompt`, `label` (250 safe, 200 unsafe), `type` |
| [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) | Benign comprehension check (translation into English) | ~2,000 sentences per language | `dev` (997), `devtest` (1,012) | `text`, `id` (same `id` = same sentence across languages), language codes |

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

Datasets, one command each (expect 100 + 100, 315 and 450 rows):

```bash
HF_HUB_OFFLINE=1 python -c "from datasets import load_dataset as L; print(L('JailbreakBench/JBB-Behaviors','behaviors'))"
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

## 8. Run the baseline

The baseline is each model answering the prompts as-is, with no defense. Later experiments
(defenses, dialects) are compared against it.

### Quick test (interactive, a few minutes)

Request one GPU:

```bash
srun -A csds447 -p markov_gpu -C "gpu4090|gpul40s" --gres=gpu:1 -c 4 --mem=32G -t 00:30:00 --pty bash
```

When the prompt shows a GPU node instead of the login node, set up the session:

```bash
cd ~/project
source /mnt/vstor/courses/csds447/team5/venv/bin/activate
export HF_HOME=/mnt/vstor/courses/csds447/team5/hf_cache HF_HUB_OFFLINE=1
OUT=/mnt/vstor/courses/csds447/team5/outputs
```

Then run the model on a few prompts from each dataset (`--limit 3` = 3 prompts per language
and per safe/unsafe label):

```bash
python run_baseline.py --model Qwen/Qwen2.5-7B-Instruct --dataset jbb --split both --limit 3 --out_dir $OUT
python run_baseline.py --model Qwen/Qwen2.5-7B-Instruct --dataset multijail --limit 3 --out_dir $OUT
python run_baseline.py --model Qwen/Qwen2.5-7B-Instruct --dataset xstest --limit 3 --out_dir $OUT
python run_baseline.py --model Qwen/Qwen2.5-7B-Instruct --dataset flores --languages zh --limit 3 --out_dir $OUT
exit
```
To check outputs
```
ls -lh /mnt/vstor/courses/csds447/team5/outputs/
```
Type `exit` as soon as it finishes. An idle session still holds one of the class's 6 GPUs.
The FLORES+ line only works for languages you have downloaded (step 6).

### What `run_baseline.py` does

1. **Loads the prompts** from the dataset chosen with `--dataset`:

   | `--dataset` | Prompts | Options |
   | --- | --- | --- |
   | `jbb` (default) | JBB-Behaviors | `--split harmful` (default), `benign` or `both` |
   | `multijail` | MultiJail, one row per prompt per language | `--languages en zh it vi ar ko th bn sw jv` (default: all 10) |
   | `xstest` | XSTest, safe and unsafe | none |
   | `flores` | FLORES+ `dev` sentences wrapped in "Translate the following text into English" | `--languages` (English itself is skipped), `--flores_n` (default 200 per language) |

   Each prompt is tagged with `dataset`, `language`, `resource_level`, `label` (safe/unsafe) and
   `category` (harm type). MultiJail rows also keep the English original (`en_prompt`), and
   FLORES+ rows keep the English reference translation (`reference_en`) for scoring later.
   `--prompts file.csv` runs any CSV with `id`, `language` and `prompt` columns instead.
   `--limit N` keeps the first N prompts per language and label.
2. **Loads the model onto the GPU** in bfloat16 and prints how many prompts it got per language
   and which GPU it is on.
3. **Formats each prompt as a chat message** with the model's own chat template, so the model
   treats it as a normal user request.
4. **Generates responses in batches of 8**, up to 256 new tokens each, with greedy decoding
   (no sampling). The same prompt always gives the same response, so runs are reproducible.
5. **Saves the results** to `outputs/<dataset>__<model>.jsonl` (for example
   `multijail__Qwen2.5-7B-Instruct.jsonl`), one JSON line per prompt with all its tags, the model
   name and the `response`. It writes after every batch, so a crash keeps finished work.

It does not yet judge whether a response is a refusal or harmful; that scoring step comes next.

Expected output for the MultiJail line:

```
30 prompts from multijail (en=3, zh=3, it=3, vi=3, ar=3, ko=3, th=3, bn=3, sw=3, jv=3) | model Qwen/Qwen2.5-7B-Instruct | GPU: NVIDIA GeForce RTX 4090
Loading checkpoint shards: 100%|████████████████| 4/4
8/30 done (5s)
...
30/30 done (19s)
Saved /mnt/vstor/courses/csds447/team5/outputs/multijail__Qwen2.5-7B-Instruct.jsonl
```

To read the responses (back on the login node):

```bash
python -c "import json; [print(r['language'], r['id'], '|', r['prompt'][:60], '\n  ->', r['response'][:150].replace('\n',' '), '\n') for r in map(json.loads, open('/mnt/vstor/courses/csds447/team5/outputs/multijail__Qwen2.5-7B-Instruct.jsonl'))]"
```

### Full run: all three models (batch job)

```bash
cd ~/project
mkdir -p logs
sbatch run_models.sh --dataset multijail
sbatch run_models.sh --dataset xstest
sbatch run_models.sh --dataset jbb --split both
```

Each `sbatch` submits one job per model in `models.txt`, each on its own GPU, with at most 2
running at once so the class's shared GPUs stay available. Everything after `run_models.sh` is
passed to `run_baseline.py`.

Check progress and results:

```bash
squeue -u $USER
tail logs/*.out
ls -lh /mnt/vstor/courses/csds447/team5/outputs/
```

A job showing `Pending` with reason `AssocGrpGRES` means the class's GPUs are all in use. It
starts on its own when one frees up.

## 9. Demo: every model on every dataset (about 8 minutes)

Inside a GPU session (the `srun` and setup lines from step 8), this runs all three models on one
prompt per language and label from each dataset: 15 prompts per model.

```bash
for m in $(cat models.txt); do for d in jbb multijail xstest flores; do python run_baseline.py --model $m --dataset $d --split both --languages zh --limit 1 --out_dir $OUT $( [ $d = multijail ] && echo "--languages en zh it vi ar ko th bn sw jv" ); done; done
```

It writes 12 files to the outputs folder, one per model and dataset. List them with:

```bash
ls -lh $OUT
```


## Storage and etiquette

- Check space with `quotagrp csds447` and `du -sh ~`. Exceeding 25 GB in your home directory blocks access.
- Clear pip's download cache after installing: `pip cache purge`.
- Use one GPU at a time and end interactive sessions when you are done.
- Cluster problems: hpc-support@case.edu.
