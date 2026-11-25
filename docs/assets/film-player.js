// Player for the "how one test is built" film (film.js): play and pause, a scrubber, chapter buttons.
// The pictures load only when the film comes near the screen, and nothing is drawn while it is off screen.
(() => {
  const base = new URL(".", document.currentScript.src);
  const $ = (id) => document.getElementById(id);
  const canvas = $("film-canvas"), ctx = canvas.getContext("2d"), range = $("film-progress");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
  const ICON_PAUSE = '<svg viewBox="0 0 16 16"><path d="M5 3v10M11 3v10" stroke="currentColor" stroke-width="2.7"/></svg>';
  const ICON_PLAY = '<svg viewBox="0 0 16 16"><path d="M4 2l9 6-9 6Z" fill="currentColor"/></svg>';
  let film = null, elapsed = 0, playing = !reduced.matches, last = null, drag = false, chapter = -1, reducedKey = "", onScreen = false;
  const clock = (t) => `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

  function icon() {
    $("film-play").setAttribute("aria-label", playing ? "Pause animation" : "Play animation");
    $("film-play").innerHTML = playing ? ICON_PAUSE : ICON_PLAY;
  }
  function render(force = false) {
    if (!film) return;
    const st = film.stateAt(elapsed), key = `${st.k}:${Math.floor(st.p * 3)}`;
    if (force || !reduced.matches || key !== reducedKey) { film.draw(elapsed, { reduced: reduced.matches }); reducedKey = key; }
    range.value = elapsed;
    range.style.setProperty("--p", `${(elapsed / film.total) * 100}%`);
    $("film-time").textContent = `${clock(elapsed)} / ${clock(film.total)}`;
    if (st.k !== chapter) {
      [...$("film-chapters").children].forEach((b, i) => (i === st.k ? b.setAttribute("aria-current", "step") : b.removeAttribute("aria-current")));
      $("film-announcement").textContent = film.chapters[st.k].title;
      chapter = st.k;
    }
  }
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2), w = Math.round(canvas.clientWidth * dpr);
    if (!w) return;
    canvas.width = w;
    canvas.height = Math.round((w * 9) / 16);
    ctx.setTransform(w / 1000, 0, 0, canvas.height / 562.5, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    render(true);
  }
  const seek = (t) => { elapsed = Math.max(0, Math.min(film.total - 0.001, t)); last = null; render(true); };
  const toggle = () => { playing = !playing; last = null; icon(); };

  async function start() {
    const images = {};
    await Promise.all(["plate", "cars", "walk"].map(async (key) => {
      const img = new Image();
      img.src = new URL(`film-${key}.webp`, base).href;
      await img.decode();
      images[key] = img;
    }));
    film = window.createOnCueFilm(ctx, images, window.FILM_SCENE, window.FILM_ATLAS);
    range.max = film.total;
    $("film-chapters").innerHTML = film.chapters.map((c, i) =>
      `<button type="button" data-chapter="${i}" aria-label="Chapter ${i + 1}: ${c.title}"><span class="num">${String(i + 1).padStart(2, "0")}</span>${c.name}</button>`).join("");
    $("film-play").addEventListener("click", toggle);
    canvas.addEventListener("click", toggle);
    canvas.addEventListener("keydown", (e) => {
      if (e.code === "Space") { e.preventDefault(); toggle(); }
      if (e.code === "ArrowLeft" || e.code === "ArrowRight") { e.preventDefault(); seek(elapsed + (e.code === "ArrowRight" ? 2 : -2)); }
    });
    $("film-restart").addEventListener("click", () => seek(0));
    $("film-chapters").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) seek(film.chapters[+b.dataset.chapter].start); });
    range.addEventListener("input", () => seek(+range.value));
    range.addEventListener("pointerdown", () => (drag = true));
    window.addEventListener("pointerup", () => { drag = false; last = null; });
    window.addEventListener("resize", resize);
    reduced.addEventListener("change", () => render(true));
    $("film-loading").hidden = true;
    icon();
    resize();
    requestAnimationFrame(tick);
  }
  function tick(ts) {
    if (onScreen && last !== null && playing && !drag && !document.hidden) {
      elapsed = (elapsed + Math.min((ts - last) / 1000, 0.1)) % film.total;
      render();
    }
    last = ts;
    requestAnimationFrame(tick);
  }
  let started = false;
  new IntersectionObserver((entries) => {
    onScreen = entries[0].isIntersecting;
    last = null;
    if (onScreen && !started) { started = true; start(); }
  }, { rootMargin: "300px" }).observe(canvas);
  icon();
})();
