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

// Unsplash URLs are imgix endpoints that take sizing params; local-library URLs are
// our own /api/photo/{id}/thumb, already sized at index time. One helper keeps the
// card renderer from having to know which corpus a result came from.
const isRemote = (url) => /^https?:/i.test(url || "");
const gridSrc = (r) => r.photo_image_url + (isRemote(r.photo_image_url) ? IMG_GRID : "");
// In library mode photo_url is /api/photo/{id}/full — the original off disk, which is
// exactly what a lightbox wants; for Unsplash we ask the CDN for a 1400px render.
const fullSrc = (r) =>
  isRemote(r.photo_image_url) ? r.photo_image_url + IMG_FULL : r.photo_url;

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
  img.alt = r.ai_description || r.description || "photograph";
  img.src = gridSrc(r);
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

  // "more like this" — nearest neighbours of this photo's own stored embedding
  const more = document.createElement("button");
  more.className = "card-more";
  more.type = "button";
  more.title = "Find visually similar photos";
  more.innerHTML =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="3" width="8" height="8" rx="1.5"/><rect x="13" y="3" width="8" height="8" rx="1.5"/><rect x="3" y="13" width="8" height="8" rx="1.5"/><rect x="13" y="13" width="8" height="8" rx="1.5"/></svg>Similar';
  more.addEventListener("click", (e) => {
    e.stopPropagation(); // don't open the lightbox
    runSimilar(r);
  });

  card.append(frame, badge, more, credit);
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
    empty.innerHTML = data.filtered
      ? `No frames match those filters. ${data.exif_count.toLocaleString()} of ${data.corpus.toLocaleString()} frames carry EXIF — try loosening a filter.`
      : `Nothing came back for <span>“${escapeHtml(data.query)}”</span>. Try a scene or a mood rather than a proper noun.`;
    gridEl.appendChild(empty);
    return;
  }

  const top = results[0].score;
  const parts = [
    `<span class="stat"><b>${results.length}</b> frames · best <b>${top.toFixed(3)}</b> · ${data.search_ms.toFixed(1)}ms · ${escapeHtml(data.store)}</span>`,
  ];
  if (data.filtered) {
    parts.push(
      `<span class="stat">filtered · searching <b>${data.exif_count.toLocaleString()}</b> of ${data.corpus.toLocaleString()} frames with EXIF</span>`,
    );
  }
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
  $("#modal-img").src = fullSrc(r);
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

  // Attribution is source-aware: Unsplash photos carry the photographer + UTM credit
  // the licence expects; your own files just say where on disk they came from.
  if (isRemote(r.photo_image_url)) {
    const link = unsplashLink(r.photo_url);
    $("#modal-credit").innerHTML = `Photo by <a href="${link}" target="_blank" rel="noopener">${escapeHtml(r.photographer)}</a> on <a href="https://unsplash.com/?${UTM}" target="_blank" rel="noopener">Unsplash</a>`;
  } else {
    $("#modal-credit").innerHTML = `<span class="local-credit">${escapeHtml(r.photographer)} / ${escapeHtml(r.description || "")}</span> · <a href="${r.photo_url}" target="_blank" rel="noopener">open original</a>`;
  }

  $("#modal-more").onclick = () => runSimilar(r);

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

/* --- filters --------------------------------------------------------------- */
// One place that reads the FilterSpec params out of the panel. Empty fields are
// omitted entirely, so an inactive filter never reaches the API.
const filterInputs = () => Array.from(document.querySelectorAll("[data-filter]"));

function readFilters() {
  const out = {};
  for (const el of filterInputs()) {
    const v = el.value.trim();
    if (v !== "") out[el.dataset.filter] = v;
  }
  return out;
}

function filterCount() {
  return Object.keys(readFilters()).length;
}

function applyFilterParams(params) {
  for (const [k, v] of Object.entries(readFilters())) params.set(k, v);
  return params;
}

function refreshFilterChrome() {
  const n = filterCount();
  const countEl = $("#filter-count");
  countEl.textContent = String(n);
  countEl.hidden = n === 0;
  $("#filter-clear").hidden = n === 0;
  // mark set selects so they read as active
  for (const el of filterInputs()) {
    if (el.tagName === "SELECT") el.classList.toggle("set", el.value.trim() !== "");
  }
}

function updateExifNote(data) {
  const el = $("#exif-note");
  if (!el || !data || data.corpus == null) return;
  el.innerHTML = `<b>${data.exif_count.toLocaleString()}</b> of ${data.corpus.toLocaleString()} frames carry EXIF — only these can match filters`;
}

/* --- corpus source: unsplash | library ------------------------------------- */
// Both corpora are loaded server-side, so switching is just a query param — the
// same encoder, the same FilterSpec, the same rendering. Nothing below this line
// knows whether the frames came from a CDN or from a folder on this machine.
let activeSource = "unsplash";

// Every request carries the active source + the active filters. One place, so no
// search path can forget either.
function searchParams(extra = {}) {
  const params = new URLSearchParams({ k: RESULTS, source: activeSource, ...extra });
  return applyFilterParams(params);
}

/* --- search state: text | image | similar ---------------------------------- */
// The current query, so changing a filter re-runs *whatever* is on screen.
let activeMode = null;
let currentReq = 0;

function resetToIdle() {
  activeMode = null;
  document.body.classList.remove("searched");
  hideModeBanner();
  gridEl.innerHTML = "";
  statusEl.innerHTML = "";
}

async function execute(fetchFn) {
  document.body.classList.add("searched");
  const reqId = ++currentReq;
  showSkeletons();
  statusEl.innerHTML = `<span class="stat">searching…</span>`;
  try {
    let resp;
    try {
      resp = await fetchFn();
    } catch (networkError) {
      // A *network* failure (not an HTTP error) on a deployed free tier almost always
      // means one thing: the container went to sleep while this tab stayed open. So
      // explain it, wait for it to come back, and re-fire the search the user already
      // asked for — rather than showing "search failed" for something they can't fix.
      if (reqId !== currentReq) return;
      const woke = await waitForBackend();
      if (reqId !== currentReq) return;
      if (!woke) throw networkError;
      resp = await fetchFn();
    }
    if (reqId !== currentReq) return; // a newer request superseded this one
    if (!resp.ok) {
      const msg = resp.status === 404 ? "photo not in the index" : `HTTP ${resp.status}`;
      throw new Error(msg);
    }
    const data = await resp.json();
    if (reqId !== currentReq) return;
    renderResults(data);
    updateExifNote(data);
  } catch (err) {
    if (reqId !== currentReq) return;
    gridEl.innerHTML = "";
    statusEl.innerHTML = `<span class="warn">search failed — ${escapeHtml(err.message)}</span>`;
  }
}

function runText(query) {
  const q = query.trim();
  if (!q) {
    resetToIdle();
    return;
  }
  activeMode = { kind: "text", text: q };
  hideModeBanner();
  execute(() => fetch(`/api/search?${searchParams({ q })}`));
}

function runSimilar(r) {
  // r may be a full result (from a card/modal) or a rehydrated {id,label,thumb}
  const id = r.photo_id ?? r.id;
  const label = r.photographer ?? r.label ?? "this frame";
  const thumb = r.photo_image_url ? gridSrc(r) : r.thumb;
  activeMode = { kind: "similar", id, label, thumb };
  closeModal();
  showModeBanner({ thumbUrl: thumb, text: `Frames visually similar to <b>${escapeHtml(label)}</b>` });
  execute(() => fetch(`/api/similar/${encodeURIComponent(id)}?${searchParams()}`));
}

function runImage(file) {
  if (!file || !file.type.startsWith("image/")) return;
  activeMode = { kind: "image", file, name: file.name };
  const thumbUrl = URL.createObjectURL(file);
  showModeBanner({ thumbUrl, text: `Frames like your upload <b>${escapeHtml(file.name)}</b>` });
  const body = new FormData();
  body.append("file", file);
  execute(() => fetch(`/api/search/by-image?${searchParams()}`, { method: "POST", body }));
}

// re-run the current query with the current filters (used on every filter change)
function rerunActive() {
  if (!activeMode) return;
  if (activeMode.kind === "text") runText(activeMode.text);
  else if (activeMode.kind === "similar") runSimilar({ id: activeMode.id, label: activeMode.label, thumb: activeMode.thumb });
  else if (activeMode.kind === "image") runImage(activeMode.file);
}

/* --- mode banner (image / similar context) --------------------------------- */
const modeBanner = $("#mode-banner");

function showModeBanner({ thumbUrl, text }) {
  const thumb = $("#mode-thumb");
  if (thumbUrl) {
    thumb.style.backgroundImage = `url("${thumbUrl}")`;
    thumb.classList.remove("icon");
    thumb.innerHTML = "";
  } else {
    thumb.style.backgroundImage = "";
    thumb.classList.add("icon");
    thumb.innerHTML =
      '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7"><rect x="3" y="3" width="8" height="8" rx="1.5"/><rect x="13" y="3" width="8" height="8" rx="1.5"/><rect x="3" y="13" width="8" height="8" rx="1.5"/><rect x="13" y="13" width="8" height="8" rx="1.5"/></svg>';
  }
  $("#mode-label").innerHTML = text;
  modeBanner.hidden = false;
}

function hideModeBanner() {
  modeBanner.hidden = true;
}

$("#mode-clear").addEventListener("click", () => {
  hideModeBanner();
  const q = inputEl.value.trim();
  if (q) runText(q);
  else resetToIdle();
});

/* --- wiring: text search --------------------------------------------------- */
const debouncedText = debounce(runText, 260);

inputEl.addEventListener("input", (e) => debouncedText(e.target.value));
$("#form").addEventListener("submit", (e) => {
  e.preventDefault();
  runText(inputEl.value);
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    inputEl.value = chip.textContent;
    inputEl.focus();
    runText(chip.textContent);
  });
});

/* --- wiring: filters panel ------------------------------------------------- */
$("#filters-toggle").addEventListener("click", () => {
  const panel = $("#filters");
  const open = panel.hidden;
  panel.hidden = !open;
  $("#filters-toggle").setAttribute("aria-expanded", String(open));
});

const debouncedRerun = debounce(rerunActive, 320);

filterInputs().forEach((el) => {
  // "input" fires for <select>, text, and number inputs alike — one handler, debounced
  el.addEventListener("input", () => {
    refreshFilterChrome();
    debouncedRerun();
  });
});

$("#filter-clear").addEventListener("click", () => {
  for (const el of filterInputs()) el.value = "";
  refreshFilterChrome();
  rerunActive();
});

/* --- wiring: search by image (button + drag-and-drop) ---------------------- */
const imageInput = $("#image-input");
$("#image-btn").addEventListener("click", () => imageInput.click());
imageInput.addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (file) runImage(file);
  imageInput.value = ""; // allow re-selecting the same file
});

const dropzone = $("#dropzone");
let dragDepth = 0;

function hasFiles(e) {
  if (!imagesSupported) return false; // text-only deploy: no vision tower to encode with
  return e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files");
}

window.addEventListener("dragenter", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth++;
  dropzone.hidden = false;
});
window.addEventListener("dragover", (e) => {
  if (hasFiles(e)) e.preventDefault();
});
window.addEventListener("dragleave", (e) => {
  if (!hasFiles(e)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) dropzone.hidden = true;
});
window.addEventListener("drop", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  dropzone.hidden = true;
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) runImage(file);
});

/* --- keyboard: "/" focuses search, Esc closes modal ------------------------ */
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (!modal.hidden) closeModal();
  }
  if (e.key === "/" && document.activeElement !== inputEl) {
    e.preventDefault();
    inputEl.focus();
  }
});

/* --- wiring: corpus source toggle ------------------------------------------ */
// The toggle only appears if the server reports more than one source, so a clone of
// this repo with no local library never shows a button that would 404.
const sourceToggle = $("#source-toggle");

function selectSource(source) {
  if (source === activeSource) return;
  activeSource = source;
  sourceToggle.querySelectorAll(".source-btn").forEach((b) => {
    b.classList.toggle("is-active", b.dataset.source === source);
  });
  loadHealth();
  // "similar" is keyed to a photo id that only exists in one corpus — drop back to
  // the text query rather than firing a request we know would 404.
  if (activeMode && activeMode.kind === "similar") {
    hideModeBanner();
    activeMode = inputEl.value.trim() ? { kind: "text", text: inputEl.value.trim() } : null;
  }
  rerunActive();
}

sourceToggle.querySelectorAll(".source-btn").forEach((btn) => {
  btn.addEventListener("click", () => selectSource(btn.dataset.source));
});

/* --- corpus stat in the top bar + EXIF note -------------------------------- */
function loadHealth() {
  return fetch(`/api/health?source=${encodeURIComponent(activeSource)}`)
    .then((r) => (r.ok ? r.json() : null))
    .then((h) => {
      if (!h || !h.indexed) return null;
      const label = h.source === "library" ? "my library" : h.store;
      $("#corpus").textContent = `${h.indexed.toLocaleString()} frames · ${label}`;
      updateExifNote({ corpus: h.indexed, exif_count: h.exif_count });
      if (h.sources && h.sources.length > 1) sourceToggle.hidden = false;
      applyCapabilities(h);
      return h;
    })
    .catch(() => null);
}

/* --- cold start (Session 11b) ---------------------------------------------
   The deployed API sleeps after 15 minutes of inactivity; waking it takes about a
   minute. The banner is deliberately NOT shown immediately — locally, and on a warm
   container, health answers in milliseconds and a flash of "waking the server…"
   would be a lie. It appears only once the wait is long enough to need explaining.
*/
const wakingEl = $("#waking");
const WAKING_HTML = wakingEl.innerHTML; // kept so the give-up message is reversible
const WAKE_BANNER_AFTER_MS = 1500;
const WAKE_GIVE_UP_AFTER_MS = 180000;
let waking = null; // in-flight wait, shared so concurrent callers don't each poll

function waitForBackend() {
  return (waking ??= pollUntilAwake().finally(() => (waking = null)));
}

async function pollUntilAwake() {
  const startedAt = Date.now();
  let attempt = 0;
  // A timer, not a check inside the loop: the loop spends most of its time asleep in
  // the backoff, so an inline check would only notice the deadline had passed on the
  // *next* iteration — showing a "please wait" notice several seconds after the wait
  // it is meant to explain had already started.
  const banner = setTimeout(() => (wakingEl.hidden = false), WAKE_BANNER_AFTER_MS);

  try {
    while (Date.now() - startedAt < WAKE_GIVE_UP_AFTER_MS) {
      const health = await loadHealth();
      if (health) {
        wakingEl.hidden = true;
        wakingEl.innerHTML = WAKING_HTML;
        return health;
      }
      attempt++;
      // Back off to 5s: a booting container is loading a 254 MB model, and hammering
      // it with retries competes for the one tenth of a CPU it has to do that with.
      await new Promise((r) => setTimeout(r, Math.min(500 * 2 ** attempt, 5000)));
    }
  } finally {
    clearTimeout(banner);
  }

  wakingEl.innerHTML =
    `<span class="waking-text"><b>The server didn't wake up.</b> ` +
    `It may be over the free tier's monthly hours — try again in a while.</span>`;
  wakingEl.hidden = false;
  return null;
}

/* Which features this backend actually has. The Render deploy runs the CLIP *text*
   tower only — no vision model in 512 MB — so search-by-image can't work there. We
   hide the affordance rather than letting someone drag a photo in and get a 501. */
let imagesSupported = true;

function applyCapabilities(health) {
  if (typeof health.supports_images !== "boolean") return;
  imagesSupported = health.supports_images;
  const imageBtn = $("#image-btn");
  if (imageBtn) imageBtn.hidden = !imagesSupported;
  if (!imagesSupported) dropzone.hidden = true;
}

waitForBackend();
refreshFilterChrome();
inputEl.focus();
