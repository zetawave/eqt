/* ============================================================
   Equity — hero 3D scene
   A tower of routed experts (one ring per transformer layer).
   Every tick the "router" selects a few experts; they flash,
   beams stream up from the storage ring below, and token
   particles rise through the core. Purely decorative.
   ============================================================ */
(function () {
  'use strict';

  var canvas = document.getElementById('hero-canvas');
  if (!canvas || typeof THREE === 'undefined') return;

  var reducedMotion = /[?&]static=1/.test(location.search) ||
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas: canvas,
      alpha: true,
      antialias: true,
      powerPreference: 'high-performance'
    });
  } catch (err) {
    return; // no WebGL: the CSS aurora still carries the hero
  }
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  var scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x0b0c0f, 0.016);

  var camera = new THREE.PerspectiveCamera(46, 1, 0.1, 120);
  var CAM_Z = 23;
  camera.position.set(0, 2.4, CAM_Z);

  /* ---------- palette ---------- */
  var COL_BG = new THREE.Color(0x262b36);
  var COL_TEAL = new THREE.Color(0x2dd4bf);
  var COL_VIOLET = new THREE.Color(0x8b5cf6);
  var COL_WHITE = new THREE.Color(0xdffcf6);

  /* ---------- expert tower ---------- */
  var LAYERS = 40;
  var PER_LAYER = 8;
  var COUNT = LAYERS * PER_LAYER;
  var TOWER_H = 15;
  var RADIUS = 5.3;

  var tower = new THREE.Group();
  scene.add(tower);

  var geo = new THREE.BoxGeometry(0.52, 0.52, 0.52);
  var mat = new THREE.MeshBasicMaterial({ toneMapped: false });
  var experts = new THREE.InstancedMesh(geo, mat, COUNT);
  experts.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  tower.add(experts);

  var dummy = new THREE.Object3D();
  var positions = [];   // Vector3 per expert
  var baseColors = [];  // base Color per expert
  var hotColors = [];   // layer-mapped hot Color per expert
  var heat = new Float32Array(COUNT);
  var tmpColor = new THREE.Color();

  for (var i = 0; i < COUNT; i++) {
    var layer = Math.floor(i / PER_LAYER);
    var slot = i % PER_LAYER;
    var angle = (slot / PER_LAYER) * Math.PI * 2 + layer * 0.34;
    var y = -TOWER_H / 2 + (layer / (LAYERS - 1)) * TOWER_H;
    var r = RADIUS + (Math.random() - 0.5) * 0.3;
    var p = new THREE.Vector3(Math.cos(angle) * r, y, Math.sin(angle) * r);
    positions.push(p);

    var s = 0.72 + Math.random() * 0.42;
    dummy.position.copy(p);
    dummy.rotation.set(Math.random() * 0.4, Math.random() * Math.PI, Math.random() * 0.4);
    dummy.scale.setScalar(s);
    dummy.updateMatrix();
    experts.setMatrixAt(i, dummy.matrix);

    var base = COL_BG.clone();
    base.offsetHSL((Math.random() - 0.5) * 0.04, 0, (Math.random() - 0.5) * 0.05);
    baseColors.push(base);
    experts.setColorAt(i, base);

    hotColors.push(new THREE.Color().lerpColors(COL_TEAL, COL_VIOLET, layer / (LAYERS - 1)));
  }
  experts.instanceColor.needsUpdate = true;

  /* ---------- procedural glow texture ---------- */
  function makeGlowTexture() {
    var c = document.createElement('canvas');
    c.width = c.height = 128;
    var g = c.getContext('2d');
    var grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(0.25, 'rgba(255,255,255,0.55)');
    grad.addColorStop(0.6, 'rgba(255,255,255,0.12)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grad;
    g.fillRect(0, 0, 128, 128);
    var t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }
  var glowTex = makeGlowTexture();

  /* ---------- storage ring ---------- */
  var ringY = -TOWER_H / 2 - 1.6;
  var ring = new THREE.Mesh(
    new THREE.TorusGeometry(RADIUS + 0.4, 0.045, 8, 120),
    new THREE.MeshBasicMaterial({ color: 0x2dd4bf, transparent: true, opacity: 0.55, toneMapped: false })
  );
  ring.rotation.x = Math.PI / 2;
  ring.position.y = ringY;
  scene.add(ring);

  // orbiting data blocks on the ring
  var BLOCKS = 22;
  var blocks = new THREE.InstancedMesh(
    new THREE.BoxGeometry(0.16, 0.1, 0.3),
    new THREE.MeshBasicMaterial({ color: 0x6fe8d8, transparent: true, opacity: 0.85, toneMapped: false }),
    BLOCKS
  );
  blocks.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(blocks);

  // storage glow puddle
  var storageGlow = new THREE.Sprite(new THREE.SpriteMaterial({
    map: glowTex, color: 0x1fc9b4, transparent: true, opacity: 0.35,
    blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false
  }));
  storageGlow.scale.set(14, 5, 1);
  storageGlow.position.set(0, ringY, 0);
  scene.add(storageGlow);

  /* ---------- beam pool (storage -> expert) ---------- */
  var BEAMS = 26;
  var beamGeo = new THREE.CylinderGeometry(0.022, 0.022, 1, 5, 1, true);
  var beams = [];
  for (var b = 0; b < BEAMS; b++) {
    var bm = new THREE.Mesh(beamGeo, new THREE.MeshBasicMaterial({
      color: 0x2dd4bf, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false
    }));
    bm.visible = false;
    scene.add(bm);
    beams.push({ mesh: bm, life: 0, maxLife: 0.5 });
  }

  var UP = new THREE.Vector3(0, 1, 0);
  var vDir = new THREE.Vector3();
  var vMid = new THREE.Vector3();

  function spawnBeam(from, to, color) {
    for (var k = 0; k < beams.length; k++) {
      var beam = beams[k];
      if (beam.life > 0) continue;
      beam.life = beam.maxLife;
      beam.mesh.visible = true;
      beam.mesh.material.color.copy(color);
      vDir.subVectors(to, from);
      var len = vDir.length();
      vMid.addVectors(from, to).multiplyScalar(0.5);
      beam.mesh.position.copy(vMid);
      beam.mesh.scale.set(1, len, 1);
      beam.mesh.quaternion.setFromUnitVectors(UP, vDir.normalize());
      return;
    }
  }

  /* ---------- glow sprite pool (expert flashes) ---------- */
  var GLOWS = 26;
  var glows = [];
  for (var g = 0; g < GLOWS; g++) {
    var sp = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false
    }));
    sp.visible = false;
    scene.add(sp);
    glows.push({ sprite: sp, life: 0, maxLife: 0.85 });
  }

  function spawnGlow(pos, color) {
    for (var k = 0; k < glows.length; k++) {
      var gl = glows[k];
      if (gl.life > 0) continue;
      gl.life = gl.maxLife;
      gl.sprite.visible = true;
      gl.sprite.position.copy(pos);
      gl.sprite.material.color.copy(color);
      return;
    }
  }

  /* ---------- ambient particle field ---------- */
  var isMobile = window.matchMedia('(max-width: 760px)').matches;
  var P_COUNT = isMobile ? 260 : 620;
  var pGeo = new THREE.BufferGeometry();
  var pPos = new Float32Array(P_COUNT * 3);
  var pSpeed = new Float32Array(P_COUNT);
  var pCol = new Float32Array(P_COUNT * 3);
  for (var p = 0; p < P_COUNT; p++) {
    var pr = 6.5 + Math.random() * 9;
    var pa = Math.random() * Math.PI * 2;
    pPos[p * 3] = Math.cos(pa) * pr;
    pPos[p * 3 + 1] = -10 + Math.random() * 21;
    pPos[p * 3 + 2] = Math.sin(pa) * pr;
    pSpeed[p] = 0.15 + Math.random() * 0.4;
    var pc = Math.random() < 0.5 ? COL_TEAL : COL_VIOLET;
    var dim = 0.35 + Math.random() * 0.65;
    pCol[p * 3] = pc.r * dim;
    pCol[p * 3 + 1] = pc.g * dim;
    pCol[p * 3 + 2] = pc.b * dim;
  }
  pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
  pGeo.setAttribute('color', new THREE.BufferAttribute(pCol, 3));
  var particles = new THREE.Points(pGeo, new THREE.PointsMaterial({
    size: 0.14, map: glowTex, vertexColors: true, transparent: true, opacity: 0.75,
    blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true, toneMapped: false
  }));
  scene.add(particles);

  /* ---------- token stream through the core ---------- */
  var T_COUNT = 34;
  var tGeo = new THREE.BufferGeometry();
  var tPos = new Float32Array(T_COUNT * 3);
  var tSpeed = new Float32Array(T_COUNT);
  var tLife = new Float32Array(T_COUNT);
  for (var t = 0; t < T_COUNT; t++) {
    resetToken(t, true);
  }
  function resetToken(idx, scatter) {
    var a = Math.random() * Math.PI * 2;
    var rr = 0.3 + Math.random() * 1.1;
    tPos[idx * 3] = Math.cos(a) * rr;
    tPos[idx * 3 + 1] = scatter ? -TOWER_H / 2 + Math.random() * TOWER_H : -TOWER_H / 2 - 1;
    tPos[idx * 3 + 2] = Math.sin(a) * rr;
    tSpeed[idx] = 2.6 + Math.random() * 2.6;
    tLife[idx] = scatter ? Math.random() : 0;
  }
  tGeo.setAttribute('position', new THREE.BufferAttribute(tPos, 3));
  var tokens = new THREE.Points(tGeo, new THREE.PointsMaterial({
    size: 0.3, map: glowTex, color: 0xbdf5ec, transparent: true, opacity: 0.9,
    blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true, toneMapped: false
  }));
  tower.add(tokens);

  /* ---------- router pulse ---------- */
  var cursorLayer = 0;
  var tickAcc = 0;
  var TICK = 0.105;

  function routerTick() {
    var layer = cursorLayer;
    cursorLayer = (cursorLayer + 1) % LAYERS;
    var t = layer / (LAYERS - 1);
    tmpColor.lerpColors(COL_TEAL, COL_VIOLET, t);
    // top-2 "routing" in this layer
    for (var n = 0; n < 2; n++) {
      var idx = layer * PER_LAYER + Math.floor(Math.random() * PER_LAYER);
      heat[idx] = 1;
      spawnGlow(positions[idx], tmpColor);
      // beam rises from the storage ring directly below the expert
      var from = new THREE.Vector3(positions[idx].x, ringY, positions[idx].z);
      spawnBeam(from, positions[idx], tmpColor);
    }
  }

  /* ---------- interaction ---------- */
  var mouseX = 0, mouseY = 0, scrollRatio = 0;
  var lookTarget = new THREE.Vector3(0, 0.6, 0);

  function updateLookTarget() {
    // bias the tower to the right of the headline on wide screens
    lookTarget.x = window.innerWidth > 900 ? -4.2 : 0;
  }
  updateLookTarget();

  window.addEventListener('pointermove', function (e) {
    mouseX = (e.clientX / window.innerWidth) * 2 - 1;
    mouseY = (e.clientY / window.innerHeight) * 2 - 1;
  }, { passive: true });

  window.addEventListener('scroll', function () {
    var doc = document.documentElement;
    var max = Math.max(1, doc.scrollHeight - window.innerHeight);
    scrollRatio = Math.min(1, window.scrollY / max);
  }, { passive: true });

  /* ---------- sizing ---------- */
  function resize() {
    var w = canvas.clientWidth || window.innerWidth;
    var h = canvas.clientHeight || window.innerHeight;
    var dpr = Math.min(window.devicePixelRatio || 1, isMobile ? 1.5 : 1.75);
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    updateLookTarget();
  }
  window.addEventListener('resize', resize);
  resize();

  /* ---------- render loop ---------- */
  var running = true;
  var visible = true;
  var clock = new THREE.Clock();

  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      visible = entries[0].isIntersecting;
    }, { threshold: 0.02 }).observe(canvas);
  }
  document.addEventListener('visibilitychange', function () {
    if (!document.hidden) clock.getDelta(); // avoid a huge first delta
  });

  var ringAngle = 0;

  function frame() {
    var dt = Math.min(clock.getDelta(), 0.05);

    if (visible && running && !document.hidden) {
      // router pulse
      tickAcc += dt;
      while (tickAcc >= TICK) {
        tickAcc -= TICK;
        routerTick();
      }

      // heat decay + instance colors
      var decay = Math.exp(-dt * 2.1);
      var dirty = false;
      for (var i2 = 0; i2 < COUNT; i2++) {
        if (heat[i2] > 0.003) {
          heat[i2] *= decay;
          tmpColor.copy(baseColors[i2]).lerp(hotColors[i2], heat[i2]);
          experts.setColorAt(i2, tmpColor);
          dirty = true;
        }
      }
      if (dirty) experts.instanceColor.needsUpdate = true;

      // slow tower rotation
      tower.rotation.y += dt * 0.06;

      // ring blocks orbit
      ringAngle += dt * 0.5;
      for (var bl = 0; bl < BLOCKS; bl++) {
        var ba = ringAngle + (bl / BLOCKS) * Math.PI * 2;
        dummy.position.set(Math.cos(ba) * (RADIUS + 0.4), ringY, Math.sin(ba) * (RADIUS + 0.4));
        dummy.rotation.set(0, -ba, 0);
        dummy.scale.setScalar(1);
        dummy.updateMatrix();
        blocks.setMatrixAt(bl, dummy.matrix);
      }
      blocks.instanceMatrix.needsUpdate = true;
      storageGlow.material.opacity = 0.28 + Math.sin(clock.elapsedTime * 2.2) * 0.08;

      // beams
      for (var k2 = 0; k2 < beams.length; k2++) {
        var beam = beams[k2];
        if (beam.life <= 0) continue;
        beam.life -= dt;
        var f = Math.max(beam.life / beam.maxLife, 0);
        beam.mesh.material.opacity = f * 0.85;
        if (beam.life <= 0) beam.mesh.visible = false;
      }

      // glow sprites
      for (var k3 = 0; k3 < glows.length; k3++) {
        var gl = glows[k3];
        if (gl.life <= 0) continue;
        gl.life -= dt;
        var gf = Math.max(gl.life / gl.maxLife, 0);
        gl.sprite.material.opacity = gf * 0.9;
        var sc = 0.7 + (1 - gf) * 2.1;
        gl.sprite.scale.set(sc, sc, 1);
        if (gl.life <= 0) gl.sprite.visible = false;
      }

      // ambient particles drift upward, wrap
      for (var p2 = 0; p2 < P_COUNT; p2++) {
        pPos[p2 * 3 + 1] += pSpeed[p2] * dt;
        if (pPos[p2 * 3 + 1] > 11) pPos[p2 * 3 + 1] = -11;
      }
      pGeo.attributes.position.needsUpdate = true;

      // token stream
      for (var t2 = 0; t2 < T_COUNT; t2++) {
        tLife[t2] += dt * 0.22;
        tPos[t2 * 3 + 1] += tSpeed[t2] * dt;
        if (tPos[t2 * 3 + 1] > TOWER_H / 2 + 1.5) resetToken(t2, false);
      }
      tGeo.attributes.position.needsUpdate = true;

      // camera: parallax + scroll pull-back
      var targetY = 2.4 - mouseY * 0.7 + scrollRatio * 5;
      var targetX = mouseX * 1.3;
      camera.position.x += (targetX - camera.position.x) * Math.min(dt * 3.2, 1);
      camera.position.y += (targetY - camera.position.y) * Math.min(dt * 3.2, 1);
      camera.position.z = CAM_Z + scrollRatio * 4;
      camera.lookAt(lookTarget);

      renderer.render(scene, camera);
    }

    if (!reducedMotion) requestAnimationFrame(frame);
  }

  if (reducedMotion) {
    // one composed still frame, no animation
    for (var s = 0; s < 46; s++) {
      var idx2 = Math.floor(Math.random() * COUNT);
      heat[idx2] = 0.75;
      tmpColor.copy(baseColors[idx2]).lerp(hotColors[idx2], heat[idx2]);
      experts.setColorAt(idx2, tmpColor);
    }
    experts.instanceColor.needsUpdate = true;
    renderer.render(scene, camera);
  } else {
    // prime a few pulses so the first frame is alive
    for (var w = 0; w < 8; w++) routerTick();
    requestAnimationFrame(frame);
  }
})();
