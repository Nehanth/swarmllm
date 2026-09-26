// Prompt-lookup drafts: when the text being written repeats something already in the context
// (a quote, a name, code being edited, a list being continued), the tokens that followed the
// same n-gram earlier are a free guess for what comes next. The trunk verifies them exactly like
// the draft head's guesses, so a wrong guess costs nothing but the lap it rides on.

// ctx: token ids so far (conversation + answer, ending with the token about to be verified).
// Returns up to k tokens that followed the most recent earlier occurrence of ctx's last n tokens
// (n from maxN down to minN), or [] when nothing matches.
export function lookupDrafts(ctx, k, { maxN = 4, minN = 2, window = 4096 } = {}) {
  const L = ctx.length;
  const lo = Math.max(0, L - window);
  for (let n = Math.min(maxN, L - 1); n >= minN; n--) {
    const tail = L - n;
    for (let i = tail - 1; i >= lo; i--) {   // candidate start of an earlier copy of the last n tokens
      let j = 0;
      while (j < n && ctx[i + j] === ctx[tail + j]) j++;
      if (j < n) continue;
      const from = i + n, to = Math.min(from + k, L);
      if (to > from) return ctx.slice(from, to);
    }
  }
  return [];
}
