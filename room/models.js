// Model catalogue for the room: URLs, layer counts, memory needs, context length.

export const NEED_GB = { "qwen3-0.6b": 0.8, "qwen3-1.7b": 2.0, "qwen3-4b": 4.6, "qwen3.8-27b": 16.5, "qwen3.6-35b-moe": 22.5, "smollm-135m": 0.6 };

export const MODELS = {
  "qwen3-0.6b": { label: "Qwen3 0.6B · Q8", kind: "gguf",
    gguf: "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json" },
  "qwen3-1.7b": { label: "Qwen3 1.7B · Q8", kind: "gguf",
    gguf: "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/tokenizer.json" },
  "qwen3-4b": { label: "Qwen3 4B · Q8", kind: "gguf",
    gguf: "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/tokenizer.json" },
  "qwen3.8-27b": { label: "Qwen 3.8 27B \u00b7 Q4", kind: "qwen35",
    gguf: "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_0.gguf" },
  // mixture of experts: 256 experts, 8 active per token (~3B of 35B), so decode reads far less than the 27B
  "qwen3.6-35b-moe": { label: "Qwen 3.6 35B MoE \u00b7 Q4", kind: "qwen35",
    gguf: "https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" },
  "smollm-135m": { label: "SmolLM 135M · bf16", kind: "safetensors",
    st: "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/model.safetensors",
    cfg: "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/config.json",
    tok: "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json" },
};

// Context window per room, in tokens: prompt + answer. Each full-attention layer keeps K and V
// for this many positions (4 KB per position each for the 27B, so 16 MiB per attention layer at
// 2048); the kernels only use it as a stride. Generation stops before the cache would overflow.
export const MAX_SEQ = 2048;
// The 27B family keeps its KV cache in f16 with split-K flash attention (engine attnFlash), so its
// rooms get 8192 positions for the price of 4096 in f32: 32 MiB per attention layer, ~0.5 GB for
// the whole model, spread over the devices that hold the layers. The host reads the engine's
// maxSeq, so prompts, answer budgets and "context full" all follow it.
export const MAX_SEQ_LONG = 8192;
export const maxSeqFor = (model) => (MODELS[model]?.kind === "qwen35" ? MAX_SEQ_LONG : MAX_SEQ);
export const MAX_NEW = 400;    // longest answer, tokens
export const MAX_NEW_THINKING = 1200;   // with thinking on, the think block comes out of the same budget
export const MIN_ROOM = 32;    // a prompt must leave at least this many tokens for the answer
