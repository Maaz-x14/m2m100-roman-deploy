
**On KV cache — direct answer:**

Your model **does** use KV caching by default — this is standard, automatic behavior in HuggingFace's `generate()` for any autoregressive decoder (which M2M100's decoder is). It's not something you enabled or need to enable; `transformers` does it unless explicitly disabled (`use_cache=False`, which nothing in your code sets).

**What it does:** during beam search decoding, each new token normally requires recomputing attention over all previous tokens. KV caching stores the previously computed key/value tensors so each new token only computes attention for itself, not the whole sequence again. Without it, generation would be dramatically slower — likely the single biggest reason your `generate()` calls are as fast as they are.

**Does it affect your current results?** Not as an open question — it's already baked into every number you've collected. There's no "should we turn it on" decision here.

**Where KV cache *does* become relevant going forward:** it's the main reason batch memory scales roughly the way we saw (linear-ish growth with batch size × sequence length) — each item in a batch needs its own KV cache. If you ever significantly increase `MAX_NEW_TOKENS` or see longer generations, KV cache memory becomes the dominant VRAM cost, more than beam width. Not a current problem given your headroom, but worth knowing why VRAM grows when it does.
