// Click a standalone figure to view it at full size; click anywhere or press Esc to close.
// Event delegation keeps it working across Material's instant navigation.
(function () {
  var overlay = null;

  function close() {
    if (overlay) {
      overlay.remove();
      overlay = null;
      document.documentElement.style.overflow = "";
    }
  }

  document.addEventListener("click", function (event) {
    if (overlay) {
      close();
      return;
    }
    var img = event.target.closest(".md-content .md-typeset p > img:only-child");
    if (!img || img.closest("a")) return;
    overlay = document.createElement("div");
    overlay.className = "site-zoom";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-label", img.alt || "figure");
    var big = document.createElement("img");
    big.src = img.currentSrc || img.src;
    big.alt = img.alt;
    // Fit the viewport; vector figures may grow freely, raster ones at most 1.6x.
    var w = img.naturalWidth || img.width;
    var h = img.naturalHeight || img.height;
    var limit = /\.svg($|\?)/i.test(big.src) ? Infinity : 1.6;
    var scale = Math.min(window.innerWidth * 0.96 / w, window.innerHeight * 0.94 / h, limit);
    big.style.width = Math.round(w * scale) + "px";
    overlay.appendChild(big);
    document.body.appendChild(overlay);
    document.documentElement.style.overflow = "hidden";
  });

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") close();
  });

  if (typeof document$ !== "undefined") document$.subscribe(close);
})();
