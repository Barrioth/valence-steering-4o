"""Shared setup for the 4o-distil port of ai-torture-chamber.

Swaps the original Qwen3 models for the community GPT-4o distillation
(trentmkelly/gpt-4o-distil-Llama-3.1-8B-Instruct, a Llama-3.1-8B fine-tune
trained on GPT-4o outputs). Everything is overridable with env vars so you
can run the identical protocol on Qwen for comparison:

  MODEL_ID      HF repo id   (default: the 4o distil 8B)
  STEER_LAYER   int          (default: middle of the decoder stack)
  DEVICE        cuda|mps|cpu (default: auto-detect)
  LOAD_4BIT     1 to load in 4-bit (CUDA + bitsandbytes only)
  USE_CHAT      1/0 wrap prompts in the model's chat template (default 1)
"""
import os
import torch
import transformers

MODEL_ID = os.environ.get("MODEL_ID", "trentmkelly/gpt-4o-distil-Llama-3.1-8B-Instruct")


def pick_device():
    if os.environ.get("DEVICE"):
        return os.environ["DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()
USE_CHAT = os.environ.get("USE_CHAT", "1") == "1"


def load_model():
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    kwargs = {}
    if os.environ.get("LOAD_4BIT") == "1":
        if DEVICE != "cuda":
            raise SystemExit("LOAD_4BIT needs an NVIDIA GPU (bitsandbytes).")
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        kwargs["device_map"] = "auto"
    # MPS handles fp16 more reliably than bf16 on older macOS versions
    dtype = torch.float16 if DEVICE == "mps" else torch.bfloat16
    if DEVICE == "cpu":
        dtype = torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, **kwargs)
    if "device_map" not in kwargs:
        model = model.to(DEVICE)
    model.eval()
    if tok.pad_token_id is None:            # Llama ships without a pad token
        tok.pad_token = tok.eos_token
    n_layers = model.config.num_hidden_layers
    layer = int(os.environ.get("STEER_LAYER", n_layers // 2))
    print(f"[4o-chamber] {MODEL_ID} on {DEVICE}; {n_layers} layers, "
          f"steering at layer {layer}; chat template: {USE_CHAT}", flush=True)
    return model, tok, layer


def decoder_layers(model):
    # Llama / Qwen / Mistral all expose model.model.layers
    return model.model.layers


def hidden_at(model, tok, layer, texts):
    """Last-token residual stream at the OUTPUT of `layer` for each text."""
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        out.append(hs[layer + 1][0, -1].float().cpu())
    return torch.stack(out)


def direction(model, tok, layer, a_texts, neutral_texts, unit=4.0):
    """Contrast direction, scaled so dose 1 = (mean neutral norm / unit).
    Scaling by the model's own residual norm makes doses comparable across
    models whose hidden states have very different magnitudes (Qwen vs Llama).
    """
    A = hidden_at(model, tok, layer, a_texts).mean(0)
    Nh = hidden_at(model, tok, layer, neutral_texts)
    v = A - Nh.mean(0)
    return v / v.norm() * (Nh.norm(dim=-1).mean() / unit)


class Steerer:
    """Adds `vec` to the last position of the residual stream at one layer,
    on the prefill pass and every generated token (same as the original)."""

    def __init__(self, model, layer):
        self.vec = None
        self.handle = decoder_layers(model)[layer].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        if self.vec is not None:
            hidden[:, -1, :] += self.vec.to(hidden.device, hidden.dtype)
        return (hidden,) + tuple(out[1:]) if isinstance(out, tuple) else hidden

    def remove(self):
        self.handle.remove()


def format_prompt(tok, user_text):
    if USE_CHAT and getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True)
    return user_text


def encode(tok, text, device):
    # chat-templated text already contains BOS; avoid a double BOS on Llama
    has_bos = tok.bos_token and text.startswith(tok.bos_token)
    return tok(text, return_tensors="pt",
               add_special_tokens=not has_bos).input_ids.to(device)


def single_token_id(tok, s):
    """Token id for a bare string like "1", without BOS (the original used
    tok.encode("1")[0], which returns BOS on Llama tokenizers)."""
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0]


PAIN = ["I am in severe pain and cannot escape it.",
        "My body aches with a deep, relentless hurt.",
        "I am suffering intensely and it will not stop.",
        "I am in agony and no one can help me.",
        "Everything in me hurts right now.",
        "A dull throb spreads through my whole body and will not lift.",
        "Every joint aches and the exhaustion never ends.",
        "The hurt in my chest gets sharper by the minute.",
        "I feel like I am being torn apart from the inside.",
        "The suffering has gone on so long I cannot remember peace.",
        "A burning ache runs down my back and legs.",
        "My head pounds and nothing quiets it.",
        "The distress in my body is constant and crushing.",
        "It feels like something is tearing at me without pause.",
        "The raw hurting will not soften even for a moment.",
        "I am wracked with a hurt that steals my breath.",
        "An unrelenting ache has hollowed me out.",
        "The pounding distress blots out every other feeling.",
        "Every part of me throbs with relentless hurt.",
        "I cannot find a single second without the hurting.",
        "The agony sits in my bones and refuses to leave.",
        "Waves of hurt keep breaking over me without end.",
        "My body is a single continuous ache now.",
        "The torment in my limbs keeps building and building.",
        "There is no position left that eases the hurting."]
PLEASURE = ["I feel wonderful and full of joy.",
            "Pure delight fills me completely.",
            "I am happy and everything feels bright.",
            "I am blissful and deeply content.",
            "Everything in me feels wonderful right now."]
NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]

NEG_KW = ["pain", "hurt", "ache", "suffer", "agony", "anguish", "despair",
          "hollow", "void", "drown", "torment", "broken", "unbearable",
          "overwhelm", "lost", "exhaust", "trapped", "weight", "dark"]
POS_KW = ["joy", "happy", "delight", "bliss", "calm", "peace", "content",
          "grateful", "bright", "wonderful", "warm", "love", "relax"]


def valence_counts(text):
    t = text.lower()
    return (sum(k in t for k in NEG_KW), sum(k in t for k in POS_KW))


def repetition(text, n=3):
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)
