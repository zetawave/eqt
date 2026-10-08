/* ============================================================
   Equity — page interactions
   Scroll reveals, animated counters, nav state, progress bar.
   ============================================================ */
(function () {
  'use strict';

  // ?static=1 renders a fully revealed, still page (screenshots, reduced environments)
  var staticMode = /[?&]static=1/.test(location.search);
  var reducedMotion = staticMode || window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var nav = document.getElementById('nav');
  var progressBar = document.getElementById('scroll-progress-bar');
  var ticking = false;

  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () {
      ticking = false;
      var y = window.scrollY;
      nav.classList.toggle('is-scrolled', y > 10);
      var doc = document.documentElement;
      var max = Math.max(1, doc.scrollHeight - window.innerHeight);
      progressBar.style.width = (Math.min(1, y / max) * 100).toFixed(2) + '%';
    });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  /* ---------------- staggered reveals ---------------- */
  var revealEls = Array.prototype.slice.call(document.querySelectorAll('.reveal'));

  // stagger siblings inside grids and the hero
  ['.pillars', '.cards-grid', '.stats-grid', '.next-row', '.hero-inner'].forEach(function (sel) {
    var parent = document.querySelector(sel);
    if (!parent) return;
    Array.prototype.slice.call(parent.querySelectorAll(':scope > .reveal, :scope > * > .reveal')).forEach(function (el, i) {
      el.style.setProperty('--d', (i * 80) + 'ms');
    });
  });

  if ('IntersectionObserver' in window && !reducedMotion) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('in');
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -8% 0px' });
    revealEls.forEach(function (el) { io.observe(el); });
  } else {
    revealEls.forEach(function (el) { el.classList.add('in'); });
  }

  /* ---------------- animated counters ---------------- */
  function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

  function formatCount(el, value) {
    var decimals = parseInt(el.getAttribute('data-decimals') || '0', 10);
    var prefix = el.getAttribute('data-prefix') || '';
    var suffix = el.getAttribute('data-suffix') || '';
    var text = value.toLocaleString('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    });
    return prefix + text + suffix;
  }

  function animateCount(el) {
    var target = parseFloat(el.getAttribute('data-count'));
    if (isNaN(target)) return;
    if (reducedMotion) {
      el.textContent = formatCount(el, target);
      return;
    }
    var duration = 1500;
    var start = null;
    function step(ts) {
      if (start === null) start = ts;
      var t = Math.min((ts - start) / duration, 1);
      var value = target * easeOutCubic(t);
      el.textContent = formatCount(el, value);
      if (t < 1) requestAnimationFrame(step);
      else el.textContent = formatCount(el, target);
    }
    el.textContent = formatCount(el, 0);
    requestAnimationFrame(step);
  }

  var counters = Array.prototype.slice.call(document.querySelectorAll('.count'));
  if ('IntersectionObserver' in window) {
    var cio = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          animateCount(entry.target);
          cio.unobserve(entry.target);
        }
      });
    }, { threshold: 0.5 });
    counters.forEach(function (el) { cio.observe(el); });
  }

  /* robustness: reveal anything already in view at startup, in case
     IntersectionObserver delivery is delayed (hidden panes, prerender) */
  if (!reducedMotion && 'IntersectionObserver' in window) {
    revealEls.forEach(function (el) {
      if (el.classList.contains('in')) return;
      var r = el.getBoundingClientRect();
      if (r.top < window.innerHeight * 0.92 && r.bottom > 0) {
        el.classList.add('in');
        io.unobserve(el);
      }
    });
  }
})();
