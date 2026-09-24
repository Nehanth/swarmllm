// Pre-flight: can this browser hold layers, and if not, what exactly to do about it. The verdict
// is a pure function of what the browser reports, so it can be unit tested; probe() gathers it.

// iPadOS reports a Mac user agent; a touch screen gives it away
export function deviceKind({ ua = "", touchPoints = 0, mobile = false } = {}) {
  if (/iPhone|iPod/.test(ua)) return "iPhone";
  if (/iPad/.test(ua) || (/Macintosh/.test(ua) && touchPoints > 1)) return "iPad";
  if (/Android/.test(ua)) return mobile || /Mobile/.test(ua) ? "Android" : "Android tablet";
  if (/Mac/.test(ua)) return "Mac";
  return "Device";
}

// facts: { ua, secure, hasGpuApi, adapter: null | { vendor, architecture }, touchPoints, mobile }
// -> { ok, kind, line }: one sentence, with the remedy when WebGPU is missing
export function verdict(f) {
  const kind = deviceKind(f);
  const ua = f.ua || "";
  if (f.adapter) {
    const g = [f.adapter.vendor, f.adapter.architecture].filter(Boolean).join(" ");
    return { ok: true, kind, line: `WebGPU works here${g ? ` (${g})` : ""}: this ${kind === "Device" ? "device" : kind} can hold part of the model.` };
  }
  const guest = " You can still join a room and ask questions.";
  if (!f.secure) return { ok: false, kind, line: "WebGPU needs a secure page: open this over https (or localhost)." + guest };
  if (f.hasGpuApi) return { ok: false, kind, line: "WebGPU is on, but the browser offered no GPU (a blocklisted driver?). Try chrome://flags/#enable-unsafe-webgpu, or another browser." + guest };
  if (kind === "iPhone" || kind === "iPad") return { ok: false, kind, line: `This ${kind} needs Safari 26 (iOS 26) or later for WebGPU: update iOS.` + guest };
  if (/Firefox\//.test(ua)) return { ok: false, kind, line: "Firefox does not have WebGPU on this platform yet: use Chrome or Edge 113+, or Safari 26+." + guest };
  if (/Linux/.test(ua) && /Chrome\//.test(ua) && !/Android/.test(ua))
    return { ok: false, kind, line: "Chrome on Linux: enable chrome://flags/#enable-unsafe-webgpu and chrome://flags/#enable-vulkan, then reload." + guest };
  if (/Android/.test(ua)) return { ok: false, kind, line: "This Android browser has no WebGPU: use Chrome 121 or later." + guest };
  if (/Safari\//.test(ua) && !/Chrome\//.test(ua)) return { ok: false, kind, line: "Safari needs version 26 for WebGPU: update macOS, or use Chrome." + guest };
  return { ok: false, kind, line: "This browser has no WebGPU: use Chrome or Edge 113+, or Safari 26+." + guest };
}

export async function probe() {
  const f = {
    ua: navigator.userAgent, secure: window.isSecureContext, hasGpuApi: !!navigator.gpu,
    touchPoints: navigator.maxTouchPoints || 0, mobile: !!navigator.userAgentData?.mobile, adapter: null,
  };
  if (navigator.gpu) {
    try {
      const a = await navigator.gpu.requestAdapter();
      if (a) f.adapter = { vendor: a.info?.vendor || "", architecture: a.info?.architecture || "" };
    } catch {}
  }
  return verdict(f);
}
