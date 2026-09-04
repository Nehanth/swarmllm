# Security

This page is the honest version of what SwarmLLM does and does not protect. Read it before running a room with anyone you would not hand a shared document link to.

## Threat model

A SwarmLLM room is a set of browsers that split one model's layers and pass the model's intermediate activations (the "hidden state") between them over direct WebRTC connections.

**What the design gives you**

- **No counterparty.** There is no account, no API key, and no server that sees your conversation. The signaling broker only introduces browsers to each other and carries no model traffic.
- **Text stays on the host.** The device that asks the question holds the tokenizer, the embedding table, the LM head, and the sampler. Other devices never receive your prompt text, the model's word scores, or the chosen tokens.
- **Weights never move between peers.** Each device fetches only its own layer range from the public model repository.
- **Transport encryption.** WebRTC data channels are encrypted with DTLS between the two browsers on each hop.

**What it cannot promise**

- **Activations are not encryption.** The hidden state that crosses each hop is a lossy transformation of your text. Published attacks reconstruct a large fraction of tokens from mid-model activations (see e.g. [arXiv 2503.09022](https://arxiv.org/abs/2503.09022)). **Assume anyone in your room can read your prompts.** The trust model is "people you would share a document link with", not "strangers".
- **No verification of remote compute.** A peer could return wrong or manipulated activations. Nothing in the current design detects this. Spot-check auditing is on the roadmap and will be documented here when it ships.
- **Noise or permutation "privacy" tricks are not used, deliberately.** They are known to be breakable and would give a false sense of safety.
- **Peers learn metadata:** that a generation happened, its length, and timing.

Consequently SwarmLLM does not, and will not, run open swarms of strangers by default, and does not claim to be "private", "encrypted end-to-end", or "verified".

## Model weights and supply chain

Weights are downloaded by each browser directly from public Hugging Face repositories over HTTPS and cached in the browser's Cache API. SwarmLLM ships no weights. The exact file each room runs is identified by its URL and size stamp; verifying a content hash against the upstream repository is planned.

## Reporting a vulnerability

Please report security issues privately to **nehanthnarendrula@gmail.com** with "SwarmLLM security" in the subject. Include steps to reproduce and the browser/OS involved. You will get an acknowledgement within 72 hours. Please do not open a public issue for security reports until a fix is available.

Issues in the threat-model sense above (activation inversion, unverified peers) are known limitations rather than vulnerabilities; discussion of them is welcome in public issues.
