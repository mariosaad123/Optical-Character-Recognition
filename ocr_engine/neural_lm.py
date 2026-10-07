"""Word prediction with a neural language model (SmolLM2, Apache 2.0, runs on CPU).

Same candidates and protections as the bigram `WordPredictor`, but each candidate is judged by how
natural the WHOLE line (plus the previous line as context) becomes with it:

    score(c) = nu * [log P_LM(line with c) - log P_LM(line as read)]   # neural context, both sides
             - lambda * OCR_edit_cost(read -> c) + mu * engine votes    # visual plausibility

Candidates are pre-ranked with the cheap bigram score and only the top-k reach the neural model,
which keeps the cost to a few batched forward passes per uncertain word.
"""

from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lm import WordPredictor, _apply_case, _core, ocr_edit_cost

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"


@lru_cache(maxsize=2)
def _load(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model


class NeuralWordPredictor(WordPredictor):
    def __init__(self, model: str = DEFAULT_MODEL, nu: float = 1.0, top_k: int = 6, **kw):
        super().__init__(**kw)
        self.tok, self.model = _load(model)
        self.nu, self.top_k = nu, top_k
        self._cache: dict[str, float] = {}

    @torch.no_grad()
    def _logprob(self, texts: list[str]) -> list[float]:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            enc = self.tok([self.tok.bos_token + t if self.tok.bos_token else t for t in todo],
                           return_tensors="pt", padding=True)
            logits = self.model(**enc).logits[:, :-1].float()
            target = enc.input_ids[:, 1:]
            mask = enc.attention_mask[:, 1:].float()
            lp = torch.log_softmax(logits, -1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
            for t, v in zip(todo, (lp * mask).sum(1).tolist()):
                self._cache[t] = v
            if len(self._cache) > 50_000:
                self._cache.clear()
        return [self._cache[t] for t in texts]

    def correct_line(self, words, alts=None, confs=None, context: str = "") -> list[str]:
        out = list(words)
        prefix = (context.strip() + " ") if context.strip() else ""
        for i, token in enumerate(words):
            conf = confs[i] if confs else 1.0
            sus = self.suspect(token, conf, alts[i] if alts else None)
            if sus is None:
                continue
            pre, core, post, alt = sus
            prev = _core(out[i - 1]) if i else None
            nxt = _core(words[i + 1]) if i + 1 < len(words) else None
            cheap = sorted(self.candidates(core, alt),
                           key=lambda c: self.lm.logp(c, prev, nxt) + self.channel(core, c, alt), reverse=True)
            cands = list(dict.fromkeys([core.lower()] + cheap[: self.top_k]))
            if len(cands) == 1:
                continue
            variants = []
            for c in cands:
                w = pre + (_apply_case(core, c) if c != core.lower() else core) + post
                variants.append(prefix + " ".join(out[:i] + [w] + words[i + 1:]))
            lps = self._logprob(variants)
            base = lps[0]
            scores = {c: self.nu * (lp - base) + self.channel(core, c, alt) for c, lp in zip(cands, lps)}
            best = max(scores, key=scores.get)
            if best == core.lower():
                continue
            if ocr_edit_cost(core.lower(), best) > self.max_cost and best not in alt:
                continue
            if self.lm.known(core) and scores[best] - scores[core.lower()] < self.margin * conf:
                continue
            out[i] = pre + _apply_case(core, best) + post
        return out
