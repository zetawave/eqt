/* ============================================================
   Equity — pipeline diagram
   A living schematic of the decode path: experts stream from
   storage through direct I/O into the slot cache and into the
   compute cores, while the router lookahead asks for the next
   layer ahead of time. Pure canvas, decorative.
   ============================================================ */
(function () {
  'use strict';

  var canvas = document.getElementById('pipeline-canvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  if (!ctx) return;

  var reducedMotion = /[?&]static=1/.test(location.search) ||
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var COLORS = {
    teal: '#2dd4bf',
    blue: '#5b9bf0',
    violet: '#8b5cf6',
    text: '#e8eaef',
    dim: '#8b93a3',
    faint: 'rgba(255,255,255,0.35)',
    nodeStroke: 'rgba(255,255,255,0.14)',
    nodeFill: 'rgba(255,255,255,0.028)'
  };

  var nodes = [];      // {x,y,w,h,name,sub,accent,flash}
  var segs = [];       // main chain: arrays of {x1,y1,x2,y2}
  var lookPts = [];    // sampled lookahead curve
  var segLabels = [];  // {x,y,text}
  var H = 320;
  var isNarrow = false;

  /* ---------------- layout ---------------- */
  function layout() {
    var W = canvas.clientWidth;
    isNarrow = W < 640;
    H = isNarrow ? 480 : 320;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    canvas.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    nodes = [];
    segs = [];
    segLabels = [];

    var defs = [
      { name: 'STORAGE', sub: isNarrow ? '22.7 GB experts' : '22.7 GB · experts on f2fs', accent: COLORS.teal },
      { name: 'DIRECT I/O', sub: 'O_DIRECT · zero-copy', accent: COLORS.teal },
      { name: 'SLOT CACHE', sub: isNarrow ? '2 GiB quotas' : '2 GiB · per-layer quotas', accent: COLORS.blue },
      { name: 'COMPUTE', sub: isNarrow ? '4 threads' : '4 threads · dotprod · i8mm', accent: COLORS.violet },
      { name: 'TOKEN', sub: '5.54 tok/s', accent: '#e8eaef' }
    ];

    if (!isNarrow) {
      var nw = 158, nh = 58;
      var gap = (W - 2 * 42 - defs.length * nw) / (defs.length - 1);
      var cy = H * 0.60;
      for (var i = 0; i < defs.length; i++) {
        var x = 42 + i * (nw + gap);
        nodes.push({
          x: x, y: cy - nh / 2, w: nw, h: nh,
          name: defs[i].name, sub: defs[i].sub, accent: defs[i].accent, flash: 0
        });
        if (i > 0) {
          var prev = nodes[i - 1];
          segs.push([{ x: prev.x + prev.w, y: cy }, { x: x, y: cy }]);
        }
      }
      // inter-segment labels
      segLabels.push({ x: (nodes[0].x + nodes[0].w + nodes[1].x) / 2, y: cy - 48, text: 'page-congruent' });
      segLabels.push({ x: (nodes[1].x + nodes[1].w + nodes[2].x) / 2, y: cy - 48, text: '91% hits' });
      segLabels.push({ x: (nodes[2].x + nodes[2].w + nodes[3].x) / 2, y: cy - 48, text: 'top-8 · 40 layers' });
      segLabels.push({ x: (nodes[3].x + nodes[3].w + nodes[4].x) / 2, y: cy - 48, text: 'greedy · exact' });

      // router lookahead: dashed curve from above COMPUTE back to STORAGE
      var fromX = nodes[3].x + nodes[3].w / 2;
      var fromY = nodes[3].y - 26;
      var toX = nodes[0].x + nodes[0].w / 2;
      var toY = nodes[0].y - 26;
      var cxp = (fromX + toX) / 2;
      var cyp = Math.min(fromY, toY) - 74;
      lookPts = [];
      for (var s = 0; s <= 44; s++) {
        var tt = s / 44;
        var mt = 1 - tt;
        lookPts.push({
          x: mt * mt * fromX + 2 * mt * tt * cxp + tt * tt * toX,
          y: mt * mt * fromY + 2 * mt * tt * cyp + tt * tt * toY
        });
      }
      nodes.push({
        x: nodes[3].x + nodes[3].w / 2 - 62, y: fromY - 34, w: 124, h: 34,
        name: 'ROUTER l (l+1)', sub: '', accent: COLORS.violet, flash: 0, small: true
      });
      nodes[nodes.length - 1].x = Math.max(42, Math.min(W - 42 - 124, nodes[nodes.length - 1].x));
    } else {
      // vertical chain
      var vw = Math.min(W - 48, 300), vh = 52;
      var top = 54, step = (H - top - 40 - vh) / (defs.length - 1);
      var cx = W / 2 - vw / 2;
      for (var j = 0; j < defs.length; j++) {
        var yy = top + j * step;
        nodes.push({
          x: cx, y: yy, w: vw, h: vh,
          name: defs[j].name, sub: defs[j].sub, accent: defs[j].accent, flash: 0
        });
        if (j > 0) {
          var pv = nodes[j - 1];
          segs.push([{ x: W / 2, y: pv.y + pv.h }, { x: W / 2, y: yy }]);
        }
      }
      // lookahead curve along the right edge, from COMPUTE up to STORAGE
      var fX = cx + vw, fY = nodes[3].y + nodes[3].h / 2;
      var tX = cx + vw, tY = nodes[0].y + nodes[0].h / 2;
      var bx = Math.min(W - 14, fX + 26);
      lookPts = [];
      for (var s2 = 0; s2 <= 44; s2++) {
        var t2 = s2 / 44;
        var m2 = 1 - t2;
        lookPts.push({
          x: m2 * m2 * fX + 2 * m2 * t2 * bx + t2 * t2 * tX,
          y: m2 * m2 * fY + 2 * m2 * t2 * (H * 0.5) + t2 * t2 * tY
        });
      }
    }
  }

  /* ---------------- helpers ---------------- */
  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function pointOnSegs(u) {
    // u in [0, segs.length)
    var i = Math.min(Math.floor(u), segs.length - 1);
    var f = Math.min(u - i, 1);
    var a = segs[i][0], b = segs[i][1];
    return { x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f, seg: i };
  }

  function pointOnLook(t) {
    var i = Math.min(Math.floor(t * (lookPts.length - 1)), lookPts.length - 2);
    var f = t * (lookPts.length - 1) - i;
    var a = lookPts[i], b = lookPts[i + 1];
    return { x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f };
  }

  function drawPacket(x, y, color, r) {
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  /* ---------------- state ---------------- */
  var packets = [];    // {u}
  var lookPackets = []; // {t}
  var spawnAcc = 0, lookAcc = 1.2;
  var time = 0;
  var running = false, inView = true;

  new IntersectionObserver(function (entries) {
    inView = entries[0].isIntersecting;
  }, { threshold: 0.05 }).observe(canvas);

  /* ---------------- draw ---------------- */
  function draw(dt) {
    time += dt;
    var W = canvas.clientWidth;

    ctx.clearRect(0, 0, W, H);

    // lookahead dashed curve
    if (lookPts.length) {
      ctx.save();
      ctx.strokeStyle = 'rgba(139,92,246,0.5)';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([5, 6]);
      ctx.lineDashOffset = -time * 22;
      ctx.beginPath();
      ctx.moveTo(lookPts[0].x, lookPts[0].y);
      for (var i = 1; i < lookPts.length; i++) ctx.lineTo(lookPts[i].x, lookPts[i].y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(139,92,246,0.85)';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      if (!isNarrow) {
        var mid = lookPts[Math.floor(lookPts.length / 2)];
        ctx.fillText('router lookahead · 81.8% accurate · prefetch layer l+1', mid.x, mid.y - 10);
      } else {
        ctx.fillText('router lookahead · 81.8%', lookPts[22].x - 4, lookPts[22].y);
      }
      ctx.restore();
    }

    // main chain segments
    ctx.strokeStyle = 'rgba(255,255,255,0.16)';
    ctx.lineWidth = 1.4;
    for (var s = 0; s < segs.length; s++) {
      var a = segs[s][0], b = segs[s][1];
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      // arrowhead
      var ang = Math.atan2(b.y - a.y, b.x - a.x);
      ctx.save();
      ctx.translate(b.x, b.y);
      ctx.rotate(ang);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(-6, -3.5);
      ctx.lineTo(-6, 3.5);
      ctx.closePath();
      ctx.fillStyle = 'rgba(255,255,255,0.35)';
      ctx.fill();
      ctx.restore();
    }

    // segment labels
    ctx.fillStyle = COLORS.dim;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    for (var l = 0; l < segLabels.length; l++) ctx.fillText(segLabels[l].text, segLabels[l].x, segLabels[l].y);

    // nodes
    for (var n = 0; n < nodes.length; n++) {
      var nd = nodes[n];
      if (nd.flash > 0.005) nd.flash *= Math.exp(-dt * 3.2);
      ctx.save();
      if (nd.flash > 0.005) {
        ctx.shadowColor = nd.accent;
        ctx.shadowBlur = 26 * nd.flash;
      }
      roundRect(nd.x, nd.y, nd.w, nd.h, nd.small ? 9 : 13);
      ctx.fillStyle = nd.flash > 0.05
        ? 'rgba(' + hexToRgb(nd.accent) + ',' + (0.06 + 0.10 * nd.flash).toFixed(3) + ')'
        : COLORS.nodeFill;
      ctx.fill();
      ctx.strokeStyle = nd.flash > 0.05
        ? 'rgba(' + hexToRgb(nd.accent) + ',' + (0.35 + 0.45 * nd.flash).toFixed(3) + ')'
        : COLORS.nodeStroke;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.restore();

      ctx.textAlign = 'center';
      if (nd.small) {
        ctx.fillStyle = COLORS.violet;
        ctx.font = '600 10px "JetBrains Mono", monospace';
        ctx.fillText(nd.name, nd.x + nd.w / 2, nd.y + nd.h / 2 + 3.5);
      } else {
        ctx.fillStyle = COLORS.text;
        ctx.font = '600 12px "Space Grotesk", "Inter", sans-serif';
        var nameY = nd.sub ? nd.y + nd.h / 2 - 5 : nd.y + nd.h / 2 + 4;
        ctx.fillText(nd.name, nd.x + nd.w / 2, nameY);
        if (nd.sub) {
          ctx.fillStyle = COLORS.dim;
          ctx.font = '9.5px "JetBrains Mono", monospace';
          ctx.fillText(nd.sub, nd.x + nd.w / 2, nd.y + nd.h / 2 + 13);
        }
        // accent tick
        ctx.fillStyle = nd.accent;
        roundRect(nd.x + 10, nd.y + 8, 14, 2.5, 1.25);
        ctx.fill();
      }
    }

    // spawn + advance main packets
    spawnAcc += dt;
    if (spawnAcc > 0.85 && packets.length < 6) {
      spawnAcc = 0;
      packets.push({ u: 0 });
    }
    for (var p = packets.length - 1; p >= 0; p--) {
      var pk = packets[p];
      pk.u += dt * 0.62;
      if (pk.u >= segs.length) {
        packets.splice(p, 1);
        nodes[nodes.length - (isNarrow ? 0 : 1)].flash = 1; // TOKEN node
        continue;
      }
      var prevSeg = Math.floor(pk.u - dt * 0.62);
      var curSeg = Math.floor(pk.u);
      if (curSeg !== prevSeg && curSeg >= 0 && curSeg < nodes.length - 1) {
        nodes[curSeg].flash = 1; // arrived at next node
      }
      var pos = pointOnSegs(pk.u);
      drawPacket(pos.x, pos.y, '#4fe3cf', 3.2);
      // short trail
      var tail = pointOnSegs(Math.max(0, pk.u - 0.06));
      ctx.strokeStyle = 'rgba(45,212,191,0.4)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(tail.x, tail.y);
      ctx.lineTo(pos.x, pos.y);
      ctx.stroke();
    }

    // lookahead packets (router -> storage)
    lookAcc += dt;
    if (lookAcc > 1.7 && lookPackets.length < 2) {
      lookAcc = 0;
      lookPackets.push({ t: 0 });
    }
    for (var lp = lookPackets.length - 1; lp >= 0; lp--) {
      var lkp = lookPackets[lp];
      lkp.t += dt * 0.55;
      if (lkp.t >= 1) {
        lookPackets.splice(lp, 1);
        nodes[0].flash = 1; // prefetch arrives at STORAGE
        continue;
      }
      var lpos = pointOnLook(lkp.t);
      drawPacket(lpos.x, lpos.y, '#a78bfa', 2.6);
    }
  }

  function hexToRgb(hex) {
    var h = hex.replace('#', '');
    var r = parseInt(h.substring(0, 2), 16);
    var g = parseInt(h.substring(2, 4), 16);
    var b = parseInt(h.substring(4, 6), 16);
    return r + ',' + g + ',' + b;
  }

  /* ---------------- loop ---------------- */
  var last = 0;
  function frame(ts) {
    var dt = Math.min((ts - last) / 1000 || 0.016, 0.05);
    last = ts;
    if (inView && !document.hidden) draw(dt);
    if (!reducedMotion) requestAnimationFrame(frame);
  }

  if ('ResizeObserver' in window) {
    new ResizeObserver(function () {
      layout();
      if (reducedMotion) draw(0); // resizing clears the canvas; repaint the still
    }).observe(canvas);
  }
  window.addEventListener('resize', layout);

  layout();
  if (reducedMotion) {
    packets.push({ u: 1.5 });
    draw(0);
  } else {
    packets.push({ u: 0.4 });
    requestAnimationFrame(frame);
  }
})();
