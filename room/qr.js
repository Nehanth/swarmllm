// A small QR code encoder for join links: byte mode, error correction level M, versions 1-10
// (up to 213 bytes, far more than a room link needs). No dependencies; DOM-free except qrSVG,
// which only builds a string. Follows ISO/IEC 18004: Reed-Solomon over GF(256) with the 0x11d
// polynomial, the standard block tables, all eight masks scored by the four penalty rules.

// Level M, versions 1..10: [ecc codewords per block, [blocks, data codewords per block], ...]
const M_BLOCKS = [null,
  [10, [1, 16]], [16, [1, 28]], [26, [1, 44]], [18, [2, 32]], [24, [2, 43]],
  [16, [4, 27]], [18, [4, 31]], [22, [2, 38], [2, 39]], [22, [3, 36], [2, 37]], [26, [4, 43], [1, 44]]];
const ALIGN = [null, [], [6, 18], [6, 22], [6, 26], [6, 30], [6, 34], [6, 22, 38], [6, 24, 42], [6, 26, 46], [6, 28, 50]];

// GF(256) log tables
const EXP = new Uint8Array(512), LOG = new Uint8Array(256);
for (let i = 0, x = 1; i < 255; i++) { EXP[i] = x; LOG[x] = i; x <<= 1; if (x & 0x100) x ^= 0x11d; }
for (let i = 255; i < 512; i++) EXP[i] = EXP[i - 255];
const gmul = (a, b) => (a && b ? EXP[LOG[a] + LOG[b]] : 0);

function rsGenerator(n) {
  let g = [1];
  for (let i = 0; i < n; i++) {
    const next = new Array(g.length + 1).fill(0);
    for (let j = 0; j < g.length; j++) { next[j] ^= g[j]; next[j + 1] ^= gmul(g[j], EXP[i]); }
    g = next;
  }
  return g;
}
export function rsEncode(data, n) {
  const g = rsGenerator(n), r = new Array(n).fill(0);
  for (const d of data) {
    const f = d ^ r[0];
    r.shift(); r.push(0);
    for (let j = 0; j < n; j++) r[j] ^= gmul(g[j + 1], f);
  }
  return r;
}

function capacity(v) { const [, ...groups] = M_BLOCKS[v]; return groups.reduce((s, [b, d]) => s + b * d, 0); }

function codewords(bytes, v) {
  const cap = capacity(v);
  const bits = [];
  const put = (val, len) => { for (let i = len - 1; i >= 0; i--) bits.push((val >>> i) & 1); };
  put(0b0100, 4);                       // byte mode
  put(bytes.length, v < 10 ? 8 : 16);   // character count
  for (const b of bytes) put(b, 8);
  put(0, Math.min(4, cap * 8 - bits.length));   // terminator
  while (bits.length % 8) bits.push(0);
  const data = [];
  for (let i = 0; i < bits.length; i += 8) data.push(parseInt(bits.slice(i, i + 8).join(""), 2));
  for (let pad = 0; data.length < cap; pad++) data.push(pad % 2 ? 0x11 : 0xec);
  // split into blocks, add ECC, interleave
  const [ecc, ...groups] = M_BLOCKS[v];
  const blocks = [];
  let off = 0;
  for (const [nb, nd] of groups) for (let b = 0; b < nb; b++) { const d = data.slice(off, off + nd); off += nd; blocks.push({ d, e: rsEncode(d, ecc) }); }
  const out = [];
  const maxD = Math.max(...blocks.map((b) => b.d.length));
  for (let i = 0; i < maxD; i++) for (const b of blocks) if (i < b.d.length) out.push(b.d[i]);
  for (let i = 0; i < ecc; i++) for (const b of blocks) out.push(b.e[i]);
  return out;
}

// BCH-coded format bits (level M = 00) and version bits
function formatBits(mask) {
  const d = (0b00 << 3) | mask;
  let r = d << 10;
  for (let i = 14; i >= 10; i--) if (r & (1 << i)) r ^= 0b10100110111 << (i - 10);
  return ((d << 10) | r) ^ 0b101010000010010;
}
function versionBits(v) {
  let r = v << 12;
  for (let i = 17; i >= 12; i--) if (r & (1 << i)) r ^= 0b1111100100101 << (i - 12);
  return (v << 12) | r;
}

const MASKS = [
  (r, c) => (r + c) % 2 === 0, (r) => r % 2 === 0, (r, c) => c % 3 === 0, (r, c) => (r + c) % 3 === 0,
  (r, c) => (Math.floor(r / 2) + Math.floor(c / 3)) % 2 === 0, (r, c) => (r * c) % 2 + (r * c) % 3 === 0,
  (r, c) => ((r * c) % 2 + (r * c) % 3) % 2 === 0, (r, c) => ((r + c) % 2 + (r * c) % 3) % 2 === 0,
];

function build(v, cw, mask) {
  const n = 17 + 4 * v;
  const m = Array.from({ length: n }, () => new Array(n).fill(0));
  const fn = Array.from({ length: n }, () => new Array(n).fill(false));   // function patterns
  const set = (r, c, val) => { m[r][c] = val ? 1 : 0; fn[r][c] = true; };
  const finder = (r0, c0) => {
    for (let r = -1; r <= 7; r++) for (let c = -1; c <= 7; c++) {
      const rr = r0 + r, cc = c0 + c;
      if (rr < 0 || cc < 0 || rr >= n || cc >= n) continue;
      const on = r >= 0 && r <= 6 && c >= 0 && c <= 6 && (r === 0 || r === 6 || c === 0 || c === 6 || (r >= 2 && r <= 4 && c >= 2 && c <= 4));
      set(rr, cc, on);
    }
  };
  finder(0, 0); finder(0, n - 7); finder(n - 7, 0);
  for (let i = 8; i < n - 8; i++) { set(6, i, i % 2 === 0); set(i, 6, i % 2 === 0); }
  const al = ALIGN[v];
  const first = al[0], last = al[al.length - 1];
  for (const r of al) for (const c of al) {
    if ((r === first && (c === first || c === last)) || (r === last && c === first)) continue;   // finder corners
    for (let dr = -2; dr <= 2; dr++) for (let dc = -2; dc <= 2; dc++) set(r + dr, c + dc, Math.max(Math.abs(dr), Math.abs(dc)) !== 1);
  }
  // format areas (filled after masking) and the dark module
  for (let i = 0; i < 9; i++) { fn[8][i] = fn[i][8] = true; }
  for (let i = 0; i < 8; i++) { fn[8][n - 1 - i] = fn[n - 1 - i][8] = true; }
  set(n - 8, 8, 1);
  if (v >= 7) {
    const vb = versionBits(v);
    for (let i = 0; i < 18; i++) { const bit = (vb >> i) & 1, a = Math.floor(i / 3), b = n - 11 + (i % 3); set(a, b, bit); set(b, a, bit); }
  }
  // data in the zigzag order, masked
  const bits = [];
  for (const w of cw) for (let i = 7; i >= 0; i--) bits.push((w >> i) & 1);
  let k = 0;
  for (let col = n - 1; col > 0; col -= 2) {
    if (col === 6) col--;
    for (let j = 0; j < n; j++) {
      const up = ((n - 1 - col) >> 1) % 2 === 0;
      const r = up ? n - 1 - j : j;
      for (const c of [col, col - 1]) {
        if (fn[r][c]) continue;
        const bit = k < bits.length ? bits[k++] : 0;
        m[r][c] = bit ^ (MASKS[mask](r, c) ? 1 : 0);
      }
    }
  }
  // format bits: one copy around the top-left finder, one split between the other two
  const f = formatBits(mask), bit = (i) => (f >> i) & 1;
  for (let i = 0; i <= 5; i++) m[i][8] = bit(i);
  m[7][8] = bit(6); m[8][8] = bit(7); m[8][7] = bit(8);
  for (let i = 9; i < 15; i++) m[8][14 - i] = bit(i);
  for (let i = 0; i < 8; i++) m[8][n - 1 - i] = bit(i);
  for (let i = 8; i < 15; i++) m[n - 15 + i][8] = bit(i);
  return m;
}

function penalty(m) {
  const n = m.length;
  let p = 0;
  // rule 1: runs of five or more in rows and columns
  for (let r = 0; r < n; r++) for (const line of [m[r], m.map((row) => row[r])]) {
    let run = 1;
    for (let i = 1; i <= n; i++) {
      if (i < n && line[i] === line[i - 1]) run++;
      else { if (run >= 5) p += run - 2; run = 1; }
    }
  }
  // rule 2: 2x2 blocks
  for (let r = 0; r < n - 1; r++) for (let c = 0; c < n - 1; c++) {
    const v = m[r][c];
    if (v === m[r][c + 1] && v === m[r + 1][c] && v === m[r + 1][c + 1]) p += 3;
  }
  // rule 3: finder-like patterns
  const pat = [1, 0, 1, 1, 1, 0, 1];
  const has = (get, i) => pat.every((x, k) => get(i + k) === x);
  for (let r = 0; r < n; r++) for (let c = 0; c + 7 <= n; c++) {
    for (const get of [(k) => m[r][k], (k) => m[k][r]]) {
      if (!has(get, c)) continue;
      const before = [c - 4, c - 3, c - 2, c - 1].every((k) => k < 0 || get(k) === 0);
      const after = [c + 7, c + 8, c + 9, c + 10].every((k) => k >= n || get(k) === 0);
      if (before || after) p += 40;
    }
  }
  // rule 4: dark balance
  let dark = 0;
  for (const row of m) for (const v of row) dark += v;
  p += Math.floor(Math.abs(dark * 100 / (n * n) - 50) / 5) * 10;
  return p;
}

// text -> matrix of 0/1 (1 = dark), smallest version that fits, best mask
export function qrMatrix(text) {
  const bytes = [...new TextEncoder().encode(text)];
  let v = 1;
  while (v <= 10 && capacity(v) < bytes.length + 2 + (v < 10 ? 0 : 1)) v++;
  if (v > 10) throw new Error("text too long for a QR code here");
  const cw = codewords(bytes, v);
  let best = null, bestP = Infinity;
  for (let mask = 0; mask < 8; mask++) {
    const m = build(v, cw, mask), p = penalty(m);
    if (p < bestP) { best = m; bestP = p; }
  }
  return best;
}

// matrix -> an SVG string with a quiet zone of 4 modules
export function qrSVG(text, { size = 200, dark = "#16171c", light = "#fbfaf5" } = {}) {
  const m = qrMatrix(text), n = m.length, q = 4, total = n + 2 * q;
  let d = "";
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) if (m[r][c]) d += `M${c + q} ${r + q}h1v1h-1z`;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${total} ${total}" width="${size}" height="${size}" shape-rendering="crispEdges" role="img" aria-label="QR code"><rect width="${total}" height="${total}" fill="${light}"/><path d="${d}" fill="${dark}"/></svg>`;
}
