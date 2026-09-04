// Wire format for activations between peers: f16 packing, binary/base64 frames, NaN checks.
import { f32ToF16, f16ToF32 } from "../engine/gguf.js";

export function badF32(a) {
  for (let i = 0; i < a.length; i += 97) if (!Number.isFinite(a[i])) return true;
  return !Number.isFinite(a[0]) || !Number.isFinite(a[a.length - 1]);
}

export function f32ToB64(f) {
  const u = new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
  let s = "";
  for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000));
  return btoa(s);
}

export const WIRE_F16 = true;

export function packF16(f) {
  const out = new Uint16Array(f.length);
  for (let i = 0; i < f.length; i++) out[i] = f32ToF16(f[i]);
  return out;
}

export function unpackF16(u) {
  const out = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) out[i] = f16ToF32(u[i]);
  return out;
}

export function asU16(x) {
  if (x instanceof Uint16Array) return x;
  if (x instanceof ArrayBuffer) return new Uint16Array(x);
  if (ArrayBuffer.isView(x)) { const u = new Uint8Array(x.buffer, x.byteOffset, x.byteLength); return new Uint16Array(u.slice().buffer); }
  throw new Error("bad f16 payload");
}

export function packWire(f) { return WIRE_F16 ? { data: packF16(f), enc: "f16" } : { data: f }; }

export function unpackWire(d) { return d.enc === "f16" ? unpackF16(asU16(d.data)) : asF32(d.data); }

export function asF32(x) {
  if (typeof x === "string") return b64ToF32(x);
  if (x instanceof Float32Array) return x;
  if (x instanceof ArrayBuffer) return new Float32Array(x);
  if (ArrayBuffer.isView(x)) {
    const u = new Uint8Array(x.buffer, x.byteOffset, x.byteLength);
    return new Float32Array(u.slice().buffer);   // copy: alignment-safe
  }
  throw new Error("unrecognized hidden-state payload");
}

export function b64ToF32(b) {
  const s = atob(b);
  const u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return new Float32Array(u.buffer);
}
