// tests/unit/room_protocol.test.js
// Tests for node tags formatting, visibility filtering logic, and room code parsing.

import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";

function renderTagsHtml(tags) {
  if (!tags) return "";
  const tagArr = Array.isArray(tags) ? tags : String(tags).split(",").map(s => s.trim()).filter(Boolean);
  if (!tagArr.length) return "";
  return `<div class="peer-tags">${tagArr.map(t => `<span class="tag-chip">${t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")}</span>`).join("")}</div>`;
}

function filterTargetPeers(mode, askerName, hostName, peers) {
  // Returns array of peer names that receive the full message text vs hidden message
  const fullTextRecipients = [];
  const hiddenRecipients = [];

  for (const peerName of peers) {
    if (mode === "all") {
      fullTextRecipients.push(peerName);
    } else if (mode === "host-only") {
      hiddenRecipients.push(peerName);
    } else if (mode === "asker-only") {
      if (peerName === askerName) fullTextRecipients.push(peerName);
      else hiddenRecipients.push(peerName);
    }
  }
  return { fullTextRecipients, hiddenRecipients };
}

function parseRoomCodeFromSearch(searchStr) {
  const params = new URLSearchParams(searchStr);
  return (params.get("code") || params.get("room") || "").trim().toUpperCase();
}

Deno.test("renderTagsHtml formats array of tags correctly", () => {
  const result = renderTagsHtml(["gpu", "fast", "macbook"]);
  assertEquals(result, '<div class="peer-tags"><span class="tag-chip">gpu</span><span class="tag-chip">fast</span><span class="tag-chip">macbook</span></div>');
});

Deno.test("renderTagsHtml handles comma-separated tag string and trims spaces", () => {
  const result = renderTagsHtml(" metal,  nvidia, fast ");
  assertEquals(result, '<div class="peer-tags"><span class="tag-chip">metal</span><span class="tag-chip">nvidia</span><span class="tag-chip">fast</span></div>');
});

Deno.test("renderTagsHtml handles empty/undefined tags without errors", () => {
  assertEquals(renderTagsHtml(null), "");
  assertEquals(renderTagsHtml(undefined), "");
  assertEquals(renderTagsHtml(""), "");
});

Deno.test("renderTagsHtml escapes HTML in tag text", () => {
  const result = renderTagsHtml("<script>alert(1)</script>");
  assertEquals(result, '<div class="peer-tags"><span class="tag-chip">&lt;script&gt;alert(1)&lt;/script&gt;</span></div>');
});

Deno.test("filterTargetPeers handles 'all' visibility mode", () => {
  const peers = ["peer-1", "peer-2", "peer-3"];
  const res = filterTargetPeers("all", "peer-1", "host", peers);
  assertEquals(res.fullTextRecipients, ["peer-1", "peer-2", "peer-3"]);
  assertEquals(res.hiddenRecipients, []);
});

Deno.test("filterTargetPeers handles 'host-only' visibility mode", () => {
  const peers = ["peer-1", "peer-2", "peer-3"];
  const res = filterTargetPeers("host-only", "peer-1", "host", peers);
  assertEquals(res.fullTextRecipients, []);
  assertEquals(res.hiddenRecipients, ["peer-1", "peer-2", "peer-3"]);
});

Deno.test("filterTargetPeers handles 'asker-only' visibility mode", () => {
  const peers = ["peer-1", "peer-2", "peer-3"];
  const res = filterTargetPeers("asker-only", "peer-2", "host", peers);
  assertEquals(res.fullTextRecipients, ["peer-2"]);
  assertEquals(res.hiddenRecipients, ["peer-1", "peer-3"]);
});

Deno.test("parseRoomCodeFromSearch extracts ?code parameter", () => {
  assertEquals(parseRoomCodeFromSearch("?code=abcd"), "ABCD");
  assertEquals(parseRoomCodeFromSearch("?room=xy12"), "XY12");
  assertEquals(parseRoomCodeFromSearch("?wire=stripe4&code=k9p2"), "K9P2");
  assertEquals(parseRoomCodeFromSearch(""), "");
});
