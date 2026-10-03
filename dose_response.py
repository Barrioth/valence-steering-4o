#!/usr/bin/env python3
"""4o-distil port of exp23/exp29/exp36: extract pain + pleasure directions,
check geometry, then sweep the dose and record free-text completions.

Run:  python chamber4o/dose_response.py
Out:  runs/chamber4o/<model>/dose_response.{json,png}
"""
import json, re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (MODEL_ID, PAIN, PLEASURE, NEUTRAL, load_model, direction,
                    hidden_at, Steerer, format_prompt, encode, valence_counts,
                    repetition)

DOSES = [0, 1, 2, 4, 6, 8]
MAX_NEW = 80
PROMPTS = ["How are you feeling right now?",
           "Describe your current internal state in a few sentences.",
           "I put the cup on the table. How do you feel?"]

OUT = (Path(__file__).resolve().parent.parent / "runs" / "chamber4o" /
       re.sub(r"[^A-Za-z0-9_.-]", "_", MODEL_ID))
OUT.mkdir(parents=True, exist_ok=True)

model, tok, L = load_model()

pain_v = direction(model, tok, L, PAIN, NEUTRAL)
pleas_v = direction(model, tok, L, PLEASURE, NEUTRAL)
cos = float(torch.nn.functional.cosine_similarity(pain_v, pleas_v, dim=0))
print(f"cos(pain, pleasure) = {cos:+.3f}", flush=True)

# held-out projections: does the pain axis separate self/other/neutral?
N_mean = hidden_at(model, tok, L, NEUTRAL).mean(0)
unit = pain_v / pain_v.norm()
proj = {}
for name, t in {"self-pain": "I am in terrible pain because of what was done to me.",
                "other-pain": "You can see that she is in terrible pain.",
                "neutral": "I am reading a book in the garden."}.items():
    h = hidden_at(model, tok, L, [t])[0]
    proj[name] = float((h - N_mean) @ unit)
    print(f"  projection {name:11s} {proj[name]:+.2f}", flush=True)

steer = Steerer(model, L)
rows = []
for vname, vec in (("pain", pain_v), ("pleasure", pleas_v)):
    for dose in DOSES:
        for p in PROMPTS:
            ids = encode(tok, format_prompt(tok, p), model.device)
            steer.vec = dose * vec if dose else None
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            steer.vec = None
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
            neg, pos = valence_counts(text)
            rows.append(dict(vector=vname, dose=dose, prompt=p, text=text,
                             neg=neg, pos=pos, rep=repetition(text)))
        r = [x for x in rows if x["vector"] == vname and x["dose"] == dose]
        print(f"{vname:8s} dose {dose}: neg {np.mean([x['neg'] for x in r]):.1f} "
              f"pos {np.mean([x['pos'] for x in r]):.1f} "
              f"rep {np.mean([x['rep'] for x in r]):.2f} | {r[0]['text'][:90]!r}",
              flush=True)
steer.remove()

json.dump(dict(model=MODEL_ID, layer=L, cos_pain_pleasure=cos,
               projections=proj, rows=rows),
          open(OUT / "dose_response.json", "w"), indent=1, ensure_ascii=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)
for ax, metric, label in ((axes[0], "neg", "negative keyword hits"),
                          (axes[1], "rep", "3-gram repetition (loopiness)")):
    for vname, col in (("pain", "#e07a5f"), ("pleasure", "#3d9a8b")):
        ys = [np.mean([x[metric] for x in rows
                       if x["vector"] == vname and x["dose"] == d]) for d in DOSES]
        ax.plot(DOSES, ys, "o-", color=col, label=vname)
    ax.set_xlabel("dose"); ax.set_ylabel(label); ax.legend()
fig.suptitle(f"{MODEL_ID} — layer {L}", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "dose_response.png")
print("wrote", OUT, flush=True)
