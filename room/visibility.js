// Who sees the chat text. Every device in the chain still computes the answer; this only decides
// which screens get the question and the streamed answer. DOM-free so it can be unit tested.
//   all   — everyone in the room
//   host  — only the host's screen
//   asker — the host's screen and the device that asked (by peer id, names are not unique)
export const VISIBILITY = ["all", "host", "asker"];

// ids: every connected peer. Returns who gets the full message and who gets the hidden stand-in.
export function chatRecipients(mode, askerId, ids) {
  const full = [], hidden = [];
  for (const id of ids) {
    if (mode === "all" || (mode === "asker" && id === askerId)) full.push(id);
    else hidden.push(id);
  }
  return { full, hidden };
}
