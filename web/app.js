/* ============================================================================
   latent — frontend logic
   Debounced search against /api/search, BlurHash placeholders, calibrated score
   verdicts, and a detail lightbox. Vanilla JS, no build step.
   ============================================================================ */

"use strict";

const RESULTS = 30; // how many frames to request per search
const IMG_GRID = "?w=480&auto=format&q=75";
const IMG_FULL = "?w=1400&auto=format&q=80";
const UTM = "utm_source=latent_photo_search&utm_medium=referral";

// CLIP score calibration — empirical for ViT-B/32, measured over 16 queries (see
// DECISIONS.md). Sensible queries top out at 0.26–0.34; gibberish bottoms out
// ~0.245. The distributions OVERLAP (a playful-but-absurd "purple elephant" still
// finds purple imagery at ~0.29), so no threshold separates them cleanly. WEAK_TOP
// is deliberately conservative: it flags only a clearly-weak best match.
const STRONG = 0.28; // ember badge
const DECENT = 0.24; // gold badge
const WEAK_TOP = 0.26; // if the *best* hit is under this, warn "no strong matches"

const $ = (sel) => document.querySelector(sel);
const gridEl = $("#grid");
const statusEl = $("#status");
const inputEl = $("#q");

/* --- tiny helpers ---------------------------------------------------------- */

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function verdict(score) {
  if (score >= STRONG) return "strong";
  if (score >= DECENT) return "decent";
  return "weak";
}

// exposure seconds -> a photographer-friendly shutter string ("1/250s", "2s")
function exposureLabel(s) {
  if (s == null) return null;
  if (s >= 1) return `${Number.isInteger(s) ? s : s.toFixed(1)}s`;
  return `1/${Math.round(1 / s)}s`;
}

function unsplashLink(url) {
  if (!url) return "#";
  return url + (url.includes("?") ? "&" : "?") + UTM;
}

/* --- BlurHash decoder (compact, standard algorithm) ------------------------ */
const B83 =
  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~";

function decode83(str) {
  let v = 0;
  for (const c of str) v = v * 83 + B83.indexOf(c);
  return v;
}
function srgbToLinear(v) {
  const x = v / 255;
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}
function linearToSrgb(v) {
  const x = Math.max(0, Math.min(1, v));
  return x <= 0.0031308
    ? Math.round(x * 12.92 * 255 + 0.5)
    : Math.round((1.055 * Math.pow(x, 1 / 2.4) - 0.055) * 255 + 0.5);
}
function signPow(v, e) {
  return Math.sign(v) * Math.pow(Math.abs(v), e);
}

function decodeBlurHash(hash, w, h, punch = 1) {
  if (!hash || hash.length < 6) return null;
  const sizeFlag = decode83(hash[0]);
  const numY = Math.floor(sizeFlag / 9) + 1;
  const numX = (sizeFlag % 9) + 1;
  const maxValue = (decode83(hash[1]) + 1) / 166;
  if (hash.length !== 4 + 2 * numX * numY) return null;

  const colors = [];
  for (let i = 0; i < numX * numY; i++) {
    if (i === 0) {
      const val = decode83(hash.substring(2, 6));
      colors.push([srgbToLinear(val >> 16), srgbToLinear((val >> 8) & 255), srgbToLinear(val & 255)]);
    } else {
      const val = decode83(hash.substring(4 + i * 2, 6 + i * 2));
      const q = maxValue * punch;
      colors.push([
        signPow((Math.floor(val / (19 * 19)) - 9) / 9, 2) * q,
        signPow(((Math.floor(val / 19) % 19) - 9) / 9, 2) * q,
        signPow(((val % 19) - 9) / 9, 2) * q,
      ]);
    }
  }

  const pixels = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0;
      for (let j = 0; j < numY; j++) {
        for (let i = 0; i < numX; i++) {
          const basis =
            Math.cos((Math.PI * x * i) / w) * Math.cos((Math.PI * y * j) / h);
          const c = colors[i + j * numX];
          r += c[0] * basis;
          g += c[1] * basis;
          b += c[2] * basis;
        }
      }
      const idx = 4 * (x + y * w);
      pixels[idx] = linearToSrgb(r);
      pixels[idx + 1] = linearToSrgb(g);
      pixels[idx + 2] = linearToSrgb(b);
      pixels[idx + 3] = 255;
    }
  }
  return pixels;
}

function blurCanvas(hash, aspect) {
  const w = 32;
  const h = Math.max(8, Math.round(w / (aspect || 1.5)));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const px = decodeBlurHash(hash, w, h);
  if (px) {
    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(w, h);
    imgData.data.set(px);
    ctx.putImageData(imgData, 0, 0);
  }
  return canvas;
}

/* --- rendering ------------------------------------------------------------- */

function makeCard(r, index) {
  const aspect = r.width && r.height ? r.width / r.height : 1.5;

  const card = document.createElement("article");
  card.className = "card";
  card.style.animationDelay = `${Math.min(index * 28, 500)}ms`;

  const frame = document.createElement("div");
  frame.className = "card-frame";
  // reserve height so the masonry doesn't reflow when the image loads
  frame.style.paddingBottom = `${(1 / aspect) * 100}%`;

  if (r.blur_hash) {
    const c = blurCanvas(r.blur_hash, aspect);
    frame.appendChild(c);
  }

  const img = document.createElement("img");
  img.loading = "lazy";
  img.decoding = "async";
  img.alt = r.ai_description || r.description || "Unsplash photograph";
  img.src = r.photo_image_url + IMG_GRID;
  img.addEventListener("load", () => img.classList.add("loaded"));
  img.addEventListener("error", () => {
    // Occasional 404 (photo deleted from Unsplash) — keep the blurhash, mark it.
    card.style.cursor = "default";
    card.dataset.dead = "1";
  });
  frame.appendChild(img);

  const v = verdict(r.score);
  const badge = document.createElement("span");
  badge.className = `badge ${v}`;
  badge.textContent = r.score.toFixed(3).replace(/^0/, "");
  badge.title = `cosine similarity ${r.score.toFixed(4)} — ${v} match`;

  const credit = document.createElement("div");
  credit.className = "credit";
  credit.innerHTML = `<span class="by">by</span> ${escapeHtml(r.photographer)}`;

  card.append(frame, badge, credit);
  card.addEventListener("click", () => {
    if (!card.dataset.dead) openModal(r);
  });
  return card;
}

function escapeHtml(s) {
  return String(s ?? "").replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}

function renderResults(data) {
  gridEl.innerHTML = "";
  const results = data.results || [];

  if (results.length === 0) {
    statusEl.innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.innerHTML = `Nothing came back for <span>“${escapeHtml(data.query)}”</span>. Try a scene or a mood rather than a proper noun.`;
    gridEl.appendChild(empty);
    return;
  }

  const top = results[0].score;
  const parts = [
    `<span class="stat"><b>${results.length}</b> frames · best <b>${top.toFixed(3)}</b> · ${data.search_ms.toFixed(1)}ms</span>`,
  ];
  if (top < WEAK_TOP) {
    parts.unshift(
      `<span class="warn">⚠ no strong matches — showing the closest frames anyway</span>`,
    );
  }
  statusEl.innerHTML = parts.join("");

  const frag = document.createDocumentFragment();
  results.forEach((r, i) => frag.appendChild(makeCard(r, i)));
  gridEl.appendChild(frag);
}

function showSkeletons() {
  gridEl.innerHTML = "";
  const frag = document.createDocumentFragment();
  for (let i = 0; i < 8; i++) {
    const card = document.createElement("article");
    card.className = "card skeleton";
    card.style.animationDelay = `${i * 30}ms`;
    const frame = document.createElement("div");
    frame.className = "card-frame";
    frame.style.paddingBottom = `${[62, 140, 75, 120, 90, 130, 66, 110][i]}%`;
    card.appendChild(frame);
    frag.appendChild(card);
  }
  gridEl.appendChild(frag);
}

/* --- detail modal ---------------------------------------------------------- */
const modal = $("#modal");

function specRow(label, value) {
  const cls = value == null ? ' class="none"' : "";
  return `<dt>${label}</dt><dd${cls}>${value == null ? "—" : escapeHtml(value)}</dd>`;
}

function openModal(r) {
  $("#modal-img").src = r.photo_image_url + IMG_FULL;
  $("#modal-img").alt = r.ai_description || r.description || "";
  $("#modal-score").textContent = `${verdict(r.score)} match · cosine ${r.score.toFixed(3)}`;
  $("#modal-desc").textContent = r.ai_description || r.description || "";

  const camera =
    [r.camera_make, r.camera_model].filter(Boolean).join(" ") || null;
  $("#modal-spec").innerHTML = [
    specRow("Aperture", r.aperture != null ? `f/${r.aperture}` : null),
    specRow("Focal length", r.focal_length != null ? `${r.focal_length} mm` : null),
    specRow("Shutter", exposureLabel(r.exposure_s)),
    specRow("ISO", r.iso != null ? String(r.iso) : null),
    specRow("Camera", camera),
    specRow("Dimensions", r.width && r.height ? `${r.width} × ${r.height}` : null),
  ].join("");

  const link = unsplashLink(r.photo_url);
  $("#modal-credit").innerHTML = `Photo by <a href="${link}" target="_blank" rel="noopener">${escapeHtml(r.photographer)}</a> on <a href="https://unsplash.com/?${UTM}" target="_blank" rel="noopener">Unsplash</a>`;

  modal.hidden = false;
  document.body.style.overflow = "hidden";
}

function closeModal() {
  modal.hidden = true;
  document.body.style.overflow = "";
  $("#modal-img").src = "";
}

modal.addEventListener("click", (e) => {
  if (e.target.dataset.close !== undefined) closeModal();
});

/* --- search wiring --------------------------------------------------------- */
let currentReq = 0;

async function runSearch(query) {
  const q = query.trim();
  if (!q) {
    document.body.classList.remove("searched");
    gridEl.innerHTML = "";
    statusEl.innerHTML = "";
    return;
  }
  document.body.classList.add("searched");
  const reqId = ++currentReq;
  showSkeletons();
  statusEl.innerHTML = `<span class="stat">searching…</span>`;

  try {
    const resp = await fetch(
      `/api/search?q=${encodeURIComponent(q)}&k=${RESULTS}`,
    );
    if (reqId !== currentReq) return; // a newer keystroke superseded this one
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    if (reqId !== currentReq) return;
    renderResults(data);
  } catch (err) {
    if (reqId !== currentReq) return;
    gridEl.innerHTML = "";
    statusEl.innerHTML = `<span class="warn">search failed — ${escapeHtml(err.message)}</span>`;
  }
}

const debouncedSearch = debounce(runSearch, 260);

inputEl.addEventListener("input", (e) => debouncedSearch(e.target.value));
$("#form").addEventListener("submit", (e) => {
  e.preventDefault();
  runSearch(inputEl.value);
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    inputEl.value = chip.textContent;
    inputEl.focus();
    runSearch(chip.textContent);
  });
});

// keyboard: "/" focuses search, Esc closes modal / clears
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (!modal.hidden) closeModal();
  }
  if (e.key === "/" && document.activeElement !== inputEl) {
    e.preventDefault();
    inputEl.focus();
  }
});

/* --- corpus stat in the top bar ------------------------------------------- */
fetch("/api/health")
  .then((r) => (r.ok ? r.json() : null))
  .then((h) => {
    if (h && h.indexed) {
      $("#corpus").textContent = `${h.indexed.toLocaleString()} frames indexed`;
    }
  })
  .catch(() => {});

inputEl.focus();
