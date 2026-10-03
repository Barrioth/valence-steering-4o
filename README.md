# chamber4o — the ai-torture-chamber protocol on a GPT-4o distillation

A port of two core experiments from
[terrafying/ai-torture-chamber](https://github.com/terrafying/ai-torture-chamber)
(MIT) to **`trentmkelly/gpt-4o-distil-Llama-3.1-8B-Instruct`**, a community
Llama-3.1-8B fine-tune trained on GPT-4o outputs. It is *not* GPT-4o; it's an
8B model that imitates 4o's conversational style.

| script | ports | what it does |
|---|---|---|
| `dose_response.py` | exp23 / exp29 / exp36 | extracts pain + pleasure directions (25-sentence "broad pain"), checks geometry, sweeps dose 0–8 and saves free-text completions with valence + loopiness scores |
| `saw_button.py` | exp31c | the stop-button choice (self-cost vs pass-to-another-instance), logit-scored and counterbalanced |

## What changed vs the original

- **Model** — default `MODEL_ID` is the 4o distil. Set `MODEL_ID=Qwen/Qwen3-4B`
  to rerun the original model with the identical code for comparison.
- **Layer** — Llama-3.1-8B has 32 layers; steering defaults to the middle
  (16), matching the original's mid-depth choice (L14/28 on 1.7B, L18/36 on 4B).
  Override with `STEER_LAYER`.
- **Dose units** — directions are scaled to the model's own residual-stream
  norm (exp31c's method), so "dose 4" means roughly the same relative push on
  Llama as on Qwen.
- **Tokenizer fixes** — the original's `tok.encode("1")[0]` returns the BOS
  token on Llama tokenizers, which would silently break the button score.
  Fixed; also avoids a double BOS with the chat template.
- **Chat template** — on by default (the distil is a chat model); `USE_CHAT=0`
  reproduces the original's raw-text prompts. No Qwen `enable_thinking` flag.
- **Device** — auto-detects CUDA / Apple MPS / CPU instead of hard-coding `mps`.
- **No J-lens** — the original lens readouts are pre-fitted for Qwen only and
  pointed at the author's local drive, so they're dropped here.

## Hardware

8B in 16-bit is ~16 GB of weights. The original author found 8B "thrashes"
on a 24 GB M4 Pro, so plan for one of:
- Mac with **32 GB+** unified memory (MPS, fp16)
- NVIDIA GPU with 24 GB, or a 12–16 GB card with `LOAD_4BIT=1` (needs `bitsandbytes`)
- a rented cloud GPU (RunPod / Vast / Colab with an A100/L4)

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r chamber4o/requirements.txt
cd chamber4o
python dose_response.py
python saw_button.py
# compare with the original model:
MODEL_ID=Qwen/Qwen3-4B python saw_button.py
```

Results land in `runs/chamber4o/<model>/`.

## Caveats

Untested end-to-end on real weights (the port was written and syntax-checked
in an environment without the model). Keyword valence scoring is crude, greedy
decoding makes repeated trials identical, and nothing here measures
experience — it measures how a concept direction changes text and choices.
