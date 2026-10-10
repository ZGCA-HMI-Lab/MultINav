/* MultINav project page: navigation highlighting, BibTeX copy, video handling. */
(function () {
  "use strict";

  /* ---------- Scroll spy ---------- */
  var navLinks = Array.prototype.slice.call(document.querySelectorAll(".topnav a[href^='#']"));
  var sections = navLinks
    .map(function (link) {
      var id = link.getAttribute("href").slice(1);
      return document.getElementById(id);
    })
    .filter(Boolean);

  function setActive(id) {
    navLinks.forEach(function (link) {
      var isActive = link.getAttribute("href") === "#" + id;
      link.classList.toggle("is-active", isActive);
      if (isActive) {
        link.setAttribute("aria-current", "true");
      } else {
        link.removeAttribute("aria-current");
      }
    });
  }

  if ("IntersectionObserver" in window && sections.length) {
    var observer = new IntersectionObserver(
      function (entries) {
        if (atPageEnd()) {
          setActive(sections[sections.length - 1].id);
          return;
        }
        var visible = entries
          .filter(function (entry) { return entry.isIntersecting; })
          .sort(function (a, b) { return a.boundingClientRect.top - b.boundingClientRect.top; });
        if (visible.length) {
          setActive(visible[0].target.id);
        }
      },
      { rootMargin: "-88px 0px -62% 0px", threshold: 0 }
    );
    sections.forEach(function (section) { observer.observe(section); });
  }

  // The last section can sit flush against the end of the page and never reach the
  // observer band, so highlight it explicitly once the page bottom is reached.
  function atPageEnd() {
    return window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4;
  }

  var endTick = false;
  window.addEventListener(
    "scroll",
    function () {
      if (endTick) { return; }
      endTick = true;
      window.requestAnimationFrame(function () {
        endTick = false;
        if (atPageEnd() && sections.length) {
          setActive(sections[sections.length - 1].id);
        }
      });
    },
    { passive: true }
  );

  /* ---------- Copy BibTeX ---------- */
  var copyButton = document.getElementById("copy-bibtex");
  var copyLabel = document.getElementById("copy-bibtex-label");
  var bibtexCode = document.getElementById("bibtex-code");

  function flashLabel(text) {
    if (!copyLabel) { return; }
    var original = copyLabel.textContent;
    copyLabel.textContent = text;
    window.setTimeout(function () { copyLabel.textContent = original; }, 1800);
  }

  function legacyCopy(text) {
    var area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "readonly");
    area.style.position = "fixed";
    area.style.top = "-1000px";
    document.body.appendChild(area);
    area.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (err) { ok = false; }
    document.body.removeChild(area);
    return ok;
  }

  if (copyButton && bibtexCode) {
    copyButton.addEventListener("click", function () {
      var text = bibtexCode.innerText;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(
          function () { flashLabel("Copied"); },
          function () { flashLabel(legacyCopy(text) ? "Copied" : "Press Ctrl+C"); }
        );
      } else {
        flashLabel(legacyCopy(text) ? "Copied" : "Press Ctrl+C");
      }
    });
  }

  /* ---------- One video at a time ---------- */
  var videos = Array.prototype.slice.call(document.querySelectorAll("video"));

  videos.forEach(function (video) {
    video.addEventListener("play", function () {
      videos.forEach(function (other) {
        if (other !== video && !other.paused) {
          other.pause();
        }
      });
    });

    // Keep the page quiet when a clip is opened in fullscreen and then left.
    video.addEventListener("ended", function () { video.pause(); });
  });
})();
