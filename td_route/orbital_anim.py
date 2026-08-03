"""
Render a solved orbital route (see orbital_sim.build_trajectory) as:

  * save_html  -- a SELF-CONTAINED interactive Three.js page (embedded library):
      left  = 3D Earth + orbits (numbers only, chaser orbit in RED), per-orbit
              checkboxes, play/pause + scrub slider;
      right = 4 live panels (current-arc ToF vs departure day, current-arc dV vs
              departure day, chaser altitude vs elapsed day, cumulative dV vs
              elapsed day);
      bottom = elapsed time and total dV counters.
  * save_mp4   -- the same layout rendered with matplotlib + imageio-ffmpeg.

Both consume the identical trajectory dict, so the MP4 and the HTML show the
same motion.
"""
from __future__ import annotations

import json
import math
import os

_VENDOR = os.path.join(os.path.dirname(__file__), "vendor")


# ========================================================================== #
# MP4 (matplotlib + imageio-ffmpeg)
# ========================================================================== #
def _rot(x, y, inc_deg, raan_deg):
    """Rotate in-plane (x, y) by inclination then RAAN -> ECI xyz."""
    inc, raan = math.radians(inc_deg), math.radians(raan_deg)
    y1, z1 = y * math.cos(inc), y * math.sin(inc)
    return (x * math.cos(raan) - y1 * math.sin(raan),
            x * math.sin(raan) + y1 * math.cos(raan), z1)


def _orbit_xyz(a, inc_deg, raan_deg, n=200, e=0.0, argp_deg=0.0):
    """Sample a conic (circle when e=0, else an ellipse with its focus at the
    origin and periapsis argp from the ascending node)."""
    import numpy as np
    nu = np.linspace(0.0, 2 * math.pi, n)
    r = a * (1 - e * e) / (1 + e * np.cos(nu))
    u = nu + math.radians(argp_deg)
    x, y = r * np.cos(u), r * np.sin(u)
    inc, raan = math.radians(inc_deg), math.radians(raan_deg)
    y1, z1 = y * math.cos(inc), y * math.sin(inc)
    return (x * math.cos(raan) - y1 * math.sin(raan),
            x * math.sin(raan) + y1 * math.cos(raan), z1)


def _pos_from(r, u_deg, inc_deg, raan_deg):
    """ECI position at radius r and argument of latitude u."""
    u = math.radians(u_deg)
    return _rot(r * math.cos(u), r * math.sin(u), inc_deg, raan_deg)


def _conic_pos(cr, nu_deg, inc_deg):
    """Chaser position on cr = [a, e, argp, raan] at true anomaly nu."""
    a, e, argp, raan = cr
    r = a * (1 - e * e) / (1 + e * math.cos(math.radians(nu_deg)))
    return _pos_from(r, argp + nu_deg, inc_deg, raan)


def _point_xyz(a, inc_deg, raan_deg, nu):        # legacy helper (nu in radians)
    return _pos_from(a, math.degrees(nu), inc_deg, raan_deg)


def save_mp4(traj: dict, path: str, fps: int = 20, dpi: int = 110):
    import numpy as np
    import imageio_ffmpeg
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    M, nodes, legs, frames = traj["meta"], traj["nodes"], traj["legs"], traj["frames"]
    inc, RE = M["inclination"], M["RE"]
    alts = [nd["alt"] for nd in nodes]
    amax = max(nd["a"] for nd in nodes)
    lo, hi = min(alts), max(alts)
    col = {nd["id"]: plt.cm.viridis((nd["alt"] - lo) / (hi - lo or 1))
           for nd in nodes}

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.35, 1], hspace=0.55, wspace=0.28,
                          left=0.02, right=0.96, top=0.95, bottom=0.09)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    axp = [fig.add_subplot(gs[r, 1]) for r in range(4)]
    txt = fig.text(0.5, 0.02, "", ha="center", fontsize=12)

    # Earth mesh (static geometry, drawn each frame)
    u = np.linspace(0, 2 * math.pi, 30); v = np.linspace(0, math.pi, 16)
    ex = RE * np.outer(np.cos(u), np.sin(v))
    ey = RE * np.outer(np.sin(u), np.sin(v))
    ez = RE * np.outer(np.ones_like(u), np.cos(v))

    tall = [f["t"] for f in frames]
    altall = [f["alt"] for f in frames]
    cumall = [f["cum_dv"] for f in frames]

    def draw(k):
        fr = frames[k]; leg = legs[fr["leg"]]; cur = leg["curve"]
        ax3d.clear()
        ax3d.plot_surface(ex, ey, ez, color="#3a6ea5", alpha=0.22, linewidth=0)
        tj = legs[fr["leg"]]["j"]                       # next target
        for nd in nodes:
            raan = nd["raan0"] + nd["rate"] * fr["t"]
            isT = (nd["id"] == tj)
            xs, ys, zs = _orbit_xyz(nd["a"], inc, raan)
            ax3d.plot(xs, ys, zs, color=col[nd["id"]],
                      lw=2.0 if isT else 1.0, alpha=1.0 if isT else 0.45)
            # the target's POSITION (true anomaly now matters)
            u = nd["theta0"] + nd["n"] * fr["t"]
            qx, qy, qz = _pos_from(nd["a"], u, inc, raan)
            ax3d.scatter([qx], [qy], [qz], color=col[nd["id"]],
                         s=34 if isT else 14, depthshade=False)
            px, py, pz = _point_xyz(nd["a"], inc, raan, 0.6)
            ax3d.text(px, py, pz, str(nd["id"]),
                      fontsize=11 if isT else 9, color="0.1",
                      ha="center", va="center")
        # chaser: the real conic (transfer ELLIPSE while flying) + its position
        cr = fr["cr"]
        xs, ys, zs = _orbit_xyz(cr[0], inc, cr[3], e=cr[1], argp_deg=cr[2])
        ax3d.plot(xs, ys, zs, color="red", lw=2.6)
        cx, cy, cz = _conic_pos(cr, fr["cnu"], inc)
        ax3d.scatter([cx], [cy], [cz], color="red", s=55, depthshade=False)
        R = amax * 1.02
        ax3d.set_xlim(-R, R); ax3d.set_ylim(-R, R); ax3d.set_zlim(-R, R)
        ax3d.set_box_aspect((1, 1, 1)); ax3d.set_xticks([]); ax3d.set_yticks([])
        ax3d.set_zticks([]); ax3d.view_init(elev=24, azim=35)   # FIXED viewpoint

        # p1 current-arc ToF
        ax = axp[0]; ax.clear()
        ax.plot(cur["t"], [y if y is not None else np.nan for y in cur["tof"]],
                color="tab:blue", lw=1.5)
        ax.axvline(cur["t_dep"], color="tab:green", lw=1.3)
        ax.plot([cur["t_dep"]], [cur["tof_dep"]], "o", color="tab:green", ms=5)
        ax.set_title(f"arc {leg['i']}→{leg['j']}  ToF vs departure day",
                     fontsize=9)
        ax.set_ylabel("ToF (d)", fontsize=8)

        # p2 current-arc dV
        ax = axp[1]; ax.clear()
        ax.plot(cur["t"], [y if y is not None else np.nan for y in cur["dv"]],
                color="tab:red", lw=1.5)
        ax.axhline(cur["floor"], color="0.5", ls="--", lw=0.8)
        ax.axvline(cur["t_dep"], color="tab:green", lw=1.3)
        ax.plot([cur["t_dep"]], [cur["dv_dep"]], "o", color="tab:green", ms=5)
        ax.set_title(f"arc {leg['i']}→{leg['j']}  ΔV vs departure day",
                     fontsize=9)
        ax.set_ylabel("ΔV (m/s)", fontsize=8)

        # p3 altitude vs elapsed
        ax = axp[2]; ax.clear()
        ax.plot(tall[:k + 1], altall[:k + 1], color="tab:orange", lw=1.6)
        ax.plot([fr["t"]], [fr["alt"]], "o", color="tab:orange", ms=5)
        ax.set_xlim(0, M["total_days"]); ax.set_title("chaser altitude (km) vs elapsed day", fontsize=9)
        ax.set_ylabel("alt (km)", fontsize=8)

        # p4 cumulative dV vs elapsed
        ax = axp[3]; ax.clear()
        ax.plot(tall[:k + 1], cumall[:k + 1], color="tab:purple", lw=1.6)
        ax.axhline(M["dv_budget"], color="tab:green", ls=":", lw=1.0)
        ax.plot([fr["t"]], [fr["cum_dv"]], "o", color="tab:purple", ms=5)
        ax.set_xlim(0, M["total_days"]); ax.set_ylim(0, M["dv_budget"] * 1.05)
        ax.set_title("cumulative ΔV (m/s) vs elapsed day", fontsize=9)
        ax.set_xlabel("elapsed day", fontsize=8)
        for ax in axp:
            ax.grid(alpha=0.3); ax.tick_params(labelsize=7)

        if fr["flying"]:
            state = ("TRANSFER %d→%d (Hohmann %.0f min, minutes/frame)"
                     % (leg["i"], leg["j"], leg["tof"] * 1440))
        elif fr["t"] >= M["total_days"] - 1e-9:
            state = "ARRIVED @ %d — rendezvous complete" % leg["j"]
        else:
            state = ("waiting @ %d for the %d window/phase (days/frame)"
                     % (leg["i"], leg["j"]))
        val = ("" if M.get("total_value") is None
               else f"value {fr.get('cum_value', 0):.2f}/{M['total_value']:.2f}      ")
        txt.set_text(f"elapsed {fr['t']:.1f} / {M['total_days']:.0f} d      "
                     f"{val}total ΔV {fr['cum_dv']:.0f} / {M['dv_budget']:.0f} m/s"
                     f"      {'→'.join(map(str, M['route']))}   [{state}]")

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)
    anim.save(path, writer=FFMpegWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"saved -> {path}")


# ========================================================================== #
# Interactive HTML (Three.js, self-contained)
# ========================================================================== #
def save_html(trajs, path: str, vendor_dir: str | None = None,
              title: str = "Orbital route animation", labels=None):
    """Write the self-contained Three.js page.

    trajs : one trajectory dict OR a list of them (e.g. every Pareto-front
            solution).  A list adds a dropdown to switch which solution plays.
    labels: dropdown labels (default: the route of each trajectory).
    """
    if isinstance(trajs, dict):
        trajs = [trajs]
    if labels is None:
        labels = ["→".join(map(str, t["meta"]["route"])) for t in trajs]
    vendor_dir = vendor_dir or _VENDOR
    with open(os.path.join(vendor_dir, "three.min.js"), encoding="utf-8") as f:
        three = f.read()
    with open(os.path.join(vendor_dir, "OrbitControls.js"), encoding="utf-8") as f:
        controls = f.read()
    data = json.dumps({"trajs": trajs, "labels": labels})
    html = (_HTML_TEMPLATE
            .replace("/*__THREE__*/", three)
            .replace("/*__CONTROLS__*/", controls)
            .replace("/*__DATA__*/", data)
            .replace("__TITLE__", title))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"saved -> {path}  ({len(trajs)} solution(s); open in a browser)")


_HTML_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
  html,body{margin:0;height:100%;background:#0b0e14;color:#dfe6f2;
    font-family:system-ui,Segoe UI,Arial,sans-serif;overflow:hidden}
  #app{display:flex;height:100vh}
  #left{width:56%;display:flex;flex-direction:column;border-right:1px solid #223}
  #scene{flex:1;min-height:0}
  #checks{padding:6px 8px;font-size:12px;line-height:1.9;border-top:1px solid #223}
  #checks label{margin-right:10px;white-space:nowrap;cursor:pointer}
  #checks input{vertical-align:-2px;margin-right:2px}
  #controls{padding:6px 10px;border-top:1px solid #223;display:flex;
    align-items:center;gap:10px;font-size:13px}
  #controls button{background:#1b2436;color:#dfe6f2;border:1px solid #33415c;
    border-radius:5px;padding:4px 12px;cursor:pointer}
  #scrub{flex:1}
  #right{width:44%;display:flex;flex-direction:column}
  #right canvas{width:100%;flex:1;min-height:0;border-bottom:1px solid #223}
  #status{padding:8px 12px;font-size:15px;border-top:1px solid #223;
    background:#0e1320;display:flex;gap:26px}
  #status b{color:#7fd1ff}
  .sw{display:inline-block;width:10px;height:10px;border-radius:2px;
    margin-right:3px;vertical-align:0px}
</style></head>
<body>
<div id="app">
  <div id="left">
    <div id="scene"></div>
    <div id="checks"></div>
    <div id="controls">
      <span>solution</span><select id="solsel"></select>
      <button id="play">Pause</button>
      <input id="scrub" type="range" min="0" max="100" value="0" step="1">
      <span>speed</span>
      <input id="speed" type="range" min="1" max="60" value="18" step="1" style="width:90px">
    </div>
  </div>
  <div id="right">
    <canvas id="p1"></canvas>
    <canvas id="p2"></canvas>
    <canvas id="p3"></canvas>
    <canvas id="p4"></canvas>
  </div>
</div>
<div id="status">
  <div>elapsed <b><span id="t_now">0</span></b> / <span id="t_tot">0</span> d</div>
  <div id="valbox">mission value <b><span id="val_now">0</span></b> / <span id="val_tot">0</span></div>
  <div>total &Delta;V <b><span id="dv_now">0</span></b> / <span id="dv_bud">0</span> m/s</div>
  <div><b><span id="leg_now">-</span></b> <span style="color:#8aa0c0">(<span id="scale_now">-</span>)</span></div>
</div>

<script>/*__THREE__*/</script>
<script>/*__CONTROLS__*/</script>
<script>
const DATA = /*__DATA__*/;
const TRAJS = DATA.trajs, LABELS = DATA.labels;
const SC = 1/1000;                       // km -> scene units
const DEG = Math.PI/180;
const nodes = TRAJS[0].nodes;            // network is shared across solutions
const inc = TRAJS[0].meta.inclination;
// current solution (mutable via the dropdown)
let T = TRAJS[0], frames = T.frames, legs = T.legs, M = T.meta;

// ---------- FIXED axis ranges (computed once over ALL solutions) ----------
// The panels must not rescale as the curve grows -- otherwise the axes move
// under you and the plot is hard to read. Every plot() call below passes an
// explicit, constant domain; only the DATA grows with time.
const HAS_MISSION = TRAJS[0].meta.has_mission;
const mx = a => a.reduce((p,c)=>c>p?c:p, -Infinity);   // max of a big array
const mn = a => a.reduce((p,c)=>c<p?c:p,  Infinity);
const GX = mx(TRAJS.map(t=>t.meta.total_days));        // elapsed-time axis (all sols)
const ALT_LO = mn(nodes.map(n=>n.alt)) - 60, ALT_HI = mx(nodes.map(n=>n.alt)) + 60;
const DVMAX = TRAJS[0].meta.dv_budget;
let VX = 1, VMAX = 1, VALMAX = 1;                      // mission-mode ranges
let AX_LO = 0, AX_HI = 1, TOF_HI = 1, DVC_HI = 1;      // arc-curve ranges
if(HAS_MISSION){
  const cs = Object.values(TRAJS[0].mission_curves);
  VX   = mx(cs.map(c=>c.t[c.t.length-1]));
  VMAX = mx(cs.map(c=>mx(c.v))) * 1.12;
  VALMAX = mx(TRAJS.map(t=>t.meta.total_value||0)) * 1.08;
} else {
  const allT=[], allTof=[], allDv=[];
  TRAJS.forEach(t=>t.legs.forEach(l=>{
    l.curve.t.forEach(x=>allT.push(x));
    l.curve.tof.forEach(y=>{ if(y!=null) allTof.push(y); });
    l.curve.dv.forEach(y=>{ if(y!=null) allDv.push(y); }); }));
  AX_LO=mn(allT); AX_HI=mx(allT); TOF_HI=mx(allTof)*1.05; DVC_HI=mx(allDv)*1.05;
}

// viridis-ish palette by altitude order
const PAL = ['#440154','#472d7b','#3b528b','#2c728e','#21918c',
             '#28ae80','#5ec962','#addc30','#dde318','#fde725'];
const nodeColor = {};
nodes.forEach((nd,i)=>{ nodeColor[nd.id]=PAL[i%PAL.length]; });

// ---------- Three.js scene ----------
const sceneEl = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({antialias:true, preserveDrawingBuffer:true});
renderer.setPixelRatio(devicePixelRatio);
sceneEl.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0b0e14);
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
camera.up.set(0,0,1); camera.position.set(14,-16,11);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff,0.75));
const dl = new THREE.DirectionalLight(0xffffff,0.7); dl.position.set(1,-1,1); scene.add(dl);

// Earth
const earth = new THREE.Mesh(
  new THREE.SphereGeometry(M.RE*SC, 40, 30),
  new THREE.MeshPhongMaterial({color:0x2a5a8a, transparent:true, opacity:0.55,
     emissive:0x0a1a2a}));
scene.add(earth);
// equatorial plane hint
const eq = new THREE.Mesh(new THREE.RingGeometry(M.RE*SC*1.001, M.RE*SC*1.02, 64),
   new THREE.MeshBasicMaterial({color:0x27406a, side:THREE.DoubleSide}));
scene.add(eq);

function ringPts(a, incDeg, n){          // ring at RAAN=0, inclination incDeg
  const ic=Math.cos(incDeg*DEG), is=Math.sin(incDeg*DEG), p=[];
  for(let k=0;k<=n;k++){ const nu=2*Math.PI*k/n;
    const x=a*Math.cos(nu), y=a*Math.sin(nu);
    p.push(new THREE.Vector3(x*SC, y*ic*SC, y*is*SC)); }
  return p;
}
// Position at radius r and argument of latitude u (both measured in the orbit
// plane from the ascending node), for a plane of inclination inc and node raan.
function posFrom(r, uDeg, incDeg, raanDeg){
  const ic=Math.cos(incDeg*DEG), is=Math.sin(incDeg*DEG);
  const rc=Math.cos(raanDeg*DEG), rs=Math.sin(raanDeg*DEG);
  const cu=Math.cos(uDeg*DEG), su=Math.sin(uDeg*DEG);
  const x=r*cu, y=r*su;
  const y1=y*ic, z1=y*is, x1=x;
  return new THREE.Vector3((x1*rc-y1*rs)*SC, (x1*rs+y1*rc)*SC, z1*SC);
}
function orbitPoint(a, incDeg, raanDeg, nuRad){   // circle, nu in RADIANS
  return posFrom(a, nuRad*180/Math.PI, incDeg, raanDeg);
}
// Chaser conic: cr = [a, e, argp, raan]; true anomaly nu (deg).
function conicPos(cr, nuDeg){
  const [a,e,argp,raan]=cr;
  const r=a*(1-e*e)/(1+e*Math.cos(nuDeg*DEG));
  return posFrom(r, argp+nuDeg, inc, raan);
}
function textSprite(txt, color){
  const c=document.createElement('canvas'); c.width=64; c.height=64;
  const g=c.getContext('2d'); g.font='bold 40px system-ui';
  g.textAlign='center'; g.textBaseline='middle';
  g.lineWidth=5; g.strokeStyle='rgba(0,0,0,0.85)'; g.strokeText(txt,32,34);
  g.fillStyle=color; g.fillText(txt,32,34);
  const t=new THREE.Texture(c); t.needsUpdate=true;
  const s=new THREE.Sprite(new THREE.SpriteMaterial({map:t, depthTest:false}));
  s.scale.set(1.1,1.1,1.1); return s;
}

// Node orbits: real TUBES (not lines -- WebGL pins line width to 1px, which is
// too faint to see). Radius is baked per node; RAAN drift is rotation.z. Each
// keeps its OWN altitude colour; the next target is emphasised by opacity +
// the thicker overlay ring below, never by recolouring.
const nodeObj = {};
nodes.forEach(nd=>{
  const g=new THREE.TorusGeometry(nd.a*SC, 0.030, 6, 170); g.rotateX(inc*DEG);
  const line=new THREE.Mesh(g, new THREE.MeshBasicMaterial({
    color:nodeColor[nd.id], transparent:true, opacity:0.55}));
  const label=textSprite(String(nd.id), '#ffffff');
  // the target's actual POSITION on its orbit (true anomaly matters now:
  // rendezvous requires the chaser to meet this dot, not just its ring)
  const dot=new THREE.Mesh(new THREE.SphereGeometry(0.10,12,10),
    new THREE.MeshBasicMaterial({color:nodeColor[nd.id]}));
  scene.add(line); scene.add(label); scene.add(dot);
  nodeObj[nd.id]={line, label, dot, nd};
});

// WebGL ignores LineBasicMaterial.linewidth, so a genuinely THICK ring needs a
// tube: a unit torus (radius 1) tilted to the inclination, then scaled by a*SC
// and spun to the RAAN -- same transform as the node lines.
function ringMesh(tube, color, opacity){
  const g=new THREE.TorusGeometry(1, tube, 8, 180); g.rotateX(inc*DEG);
  return new THREE.Mesh(g, new THREE.MeshBasicMaterial({color:color,
    transparent:opacity<1, opacity:opacity}));
}
// NEXT TARGET orbit: a THICKER ring laid over the node the chaser is flying to.
// It is tinted with that node's OWN colour every frame (see updateScene), so the
// target keeps its identity and you can tell when the red chaser ring has
// converged onto it.
const tgtRing = ringMesh(0.007, 0xffffff, 1.0); scene.add(tgtRing);

// Chaser orbit: BOLD red conic. A circle while parked on a node, the TRANSFER
// ELLIPSE while flying -- built as a unit torus scaled to (a, b) and shifted by
// -a*e so the focus sits at Earth's centre, then rotated argp / inc / raan
// through a nested group (three.js has no ZXZ Euler order).
const agentOuter = new THREE.Group(); scene.add(agentOuter);       // rotation.z = raan
const agentMid   = new THREE.Group(); agentOuter.add(agentMid);    // rotation.x = inc
const agentArgp  = new THREE.Group(); agentMid.add(agentArgp);     // rotation.z = argp
const agentLine  = new THREE.Mesh(new THREE.TorusGeometry(1, 0.011, 8, 220),
   new THREE.MeshBasicMaterial({color:0xff2020}));
agentArgp.add(agentLine);
// the chaser's own POSITION (true anomaly) -- must land on the target dot
const agentDot = new THREE.Mesh(new THREE.SphereGeometry(0.16,14,12),
   new THREE.MeshBasicMaterial({color:0xff4040}));
scene.add(agentDot);

function raanOf(nd, t){ return nd.raan0 + nd.rate*t; }
function phaseOf(nd, t){ return nd.theta0 + nd.n*t; }   // argument of latitude

function updateScene(fr){
  const t=fr.t;
  const tj=legs[fr.leg].j;                 // the node the chaser is flying TO
  nodes.forEach(nd=>{
    const o=nodeObj[nd.id]; const raan=raanOf(nd,t);
    o.line.rotation.z=raan*DEG;
    o.label.position.copy(orbitPoint(nd.a, inc, raan, 0.6));
    o.dot.position.copy(posFrom(nd.a, phaseOf(nd,t), inc, raan));
    const isT=(nd.id===tj);                // emphasise the next target
    o.line.material.opacity = isT ? 1.0 : 0.5;
    o.label.scale.setScalar(isT ? 1.8 : 1.05);
    o.dot.scale.setScalar(isT ? 1.7 : 1.0);
  });
  // thicker ring on the next target orbit, in THAT NODE'S OWN colour
  // (follows its RAAN drift)
  const tnd=nodeObj[tj].nd;
  tgtRing.material.color.set(nodeColor[tj]);
  tgtRing.scale.setScalar(tnd.a*SC);
  tgtRing.rotation.z=raanOf(tnd,t)*DEG;
  tgtRing.visible=nodeObj[tj].line.visible;
  // chaser conic: cr = [a, e, argp, raan]
  const cr=fr.cr, a=cr[0], e=cr[1], b=a*Math.sqrt(Math.max(0,1-e*e));
  agentOuter.rotation.z = cr[3]*DEG;
  agentMid.rotation.x   = inc*DEG;
  agentArgp.rotation.z  = cr[2]*DEG;
  agentLine.scale.set(a*SC, b*SC, 1);     // unit torus -> ellipse (a, b)
  agentLine.position.x = -a*e*SC;          // shift so the FOCUS is at Earth
  agentDot.position.copy(conicPos(cr, fr.cnu));
}

// ---------- checkboxes ----------
const checks=document.getElementById('checks');
nodes.forEach(nd=>{
  const l=document.createElement('label');
  l.innerHTML='<span class="sw" style="background:'+nodeColor[nd.id]+'"></span>'+
    '<input type="checkbox" checked data-id="'+nd.id+'">'+nd.id;
  checks.appendChild(l);
  l.querySelector('input').addEventListener('change',e=>{
    const o=nodeObj[nd.id];
    o.line.visible=e.target.checked; o.label.visible=e.target.checked;
    o.dot.visible=e.target.checked;
  });
});
const al=document.createElement('label');
al.innerHTML='<span class="sw" style="background:#ff3b3b"></span>'+
  '<input type="checkbox" checked id="agentChk">chaser';
checks.appendChild(al);
al.querySelector('input').addEventListener('change',e=>{
  agentLine.visible=e.target.checked; agentDot.visible=e.target.checked;
});

// ---------- 2D panels ----------
function ctxOf(id){ const c=document.getElementById(id);
  const r=c.getBoundingClientRect(); c.width=r.width*devicePixelRatio; c.height=r.height*devicePixelRatio;
  const g=c.getContext('2d'); g.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  return {g, W:r.width, H:r.height}; }

function plot(id, series, opt){
  const {g,W,H}=ctxOf(id); g.clearRect(0,0,W,H);
  const ml=46,mr=12,mt=18,mb=20;
  let xs=[],ys=[];
  series.forEach(s=>s.x.forEach((x,i)=>{ if(s.y[i]!=null){xs.push(x);ys.push(s.y[i]);} }));
  (opt.hlines||[]).forEach(h=>ys.push(h.y));
  if(!xs.length){ xs=[0,1]; ys=[0,1]; }
  let xmin=opt.xmin!=null?opt.xmin:Math.min(...xs), xmax=opt.xmax!=null?opt.xmax:Math.max(...xs);
  let ymin=opt.ymin!=null?opt.ymin:Math.min(...ys), ymax=opt.ymax!=null?opt.ymax:Math.max(...ys);
  if(xmax<=xmin)xmax=xmin+1; if(ymax<=ymin)ymax=ymin+1;
  // pad only when the range was auto-derived; an explicit domain is honoured
  // exactly so the axes stay pinned frame to frame.
  if(opt.ymin==null || opt.ymax==null){
    const pad=(ymax-ymin)*0.08; ymin-=pad; ymax+=pad; }
  const X=x=>ml+(x-xmin)/(xmax-xmin)*(W-ml-mr);
  const Y=y=>H-mb-(y-ymin)/(ymax-ymin)*(H-mt-mb);
  // axes
  g.strokeStyle='#33415c'; g.lineWidth=1; g.beginPath();
  g.moveTo(ml,mt); g.lineTo(ml,H-mb); g.lineTo(W-mr,H-mb); g.stroke();
  g.fillStyle='#8aa0c0'; g.font='10px system-ui'; g.textAlign='right'; g.textBaseline='middle';
  for(let k=0;k<=3;k++){ const yv=ymin+(ymax-ymin)*k/3; g.fillText(yv.toFixed(0), ml-4, Y(yv));
    g.strokeStyle='#1a2334'; g.beginPath(); g.moveTo(ml,Y(yv)); g.lineTo(W-mr,Y(yv)); g.stroke(); }
  g.textAlign='center'; g.textBaseline='top';
  for(let k=0;k<=3;k++){ const xv=xmin+(xmax-xmin)*k/3; g.fillText(xv.toFixed(0), X(xv), H-mb+3); }
  // hlines
  (opt.hlines||[]).forEach(h=>{ g.strokeStyle=h.color; g.setLineDash(h.dash||[4,3]); g.beginPath();
    g.moveTo(ml,Y(h.y)); g.lineTo(W-mr,Y(h.y)); g.stroke(); g.setLineDash([]); });
  // vlines
  (opt.vlines||[]).forEach(v=>{ g.strokeStyle=v.color; g.lineWidth=1.4; g.beginPath();
    g.moveTo(X(v.x),mt); g.lineTo(X(v.x),H-mb); g.stroke(); g.lineWidth=1; });
  // series polylines (break on null)
  series.forEach(s=>{ g.strokeStyle=s.color; g.lineWidth=s.w||1.6;
    if(s.dash)g.setLineDash(s.dash); g.beginPath(); let pen=false;
    s.x.forEach((x,i)=>{ const y=s.y[i]; if(y==null){pen=false;return;}
      if(!pen){g.moveTo(X(x),Y(y));pen=true;} else g.lineTo(X(x),Y(y)); });
    g.stroke(); g.setLineDash([]); });
  // markers
  (opt.markers||[]).forEach(m=>{ if(m.y==null)return; g.fillStyle=m.color;
    g.beginPath(); g.arc(X(m.x),Y(m.y),3.5,0,7); g.fill(); });
  // title
  g.fillStyle='#9fb4d4'; g.font='11px system-ui'; g.textAlign='left'; g.textBaseline='top';
  g.fillText(opt.title, ml+2, 2);
}

function updatePanels(k){
  const fr=frames[k]; const leg=legs[fr.leg];
  const tx=frames.slice(0,k+1).map(f=>f.t), ax=frames.slice(0,k+1).map(f=>f.alt);
  const altPanel=(id)=>plot(id,[{x:tx,y:ax,color:'#ffd24a'}],
    {title:'chaser altitude (km) vs elapsed day', xmin:0,xmax:GX,
     ymin:ALT_LO, ymax:ALT_HI, markers:[{x:fr.t,y:fr.alt,color:'#ffd24a'}]});
  const dvPanel=(id)=>plot(id,[{x:tx,y:frames.slice(0,k+1).map(f=>f.cum_dv),color:'#c58bff'}],
    {title:'cumulative ΔV (m/s) vs elapsed day', xmin:0,xmax:GX, ymin:0, ymax:DVMAX,
     hlines:[{y:M.dv_budget,color:'#3ad07a',dash:[2,3]}],
     markers:[{x:fr.t,y:fr.cum_dv,color:'#c58bff'}]});

  if(M.has_mission){
    // p1: mission value curve v_j(t) of the node being flown TO. Green = the
    // arrival (value collected); blue = where "now" sits on that landscape.
    const mc=T.mission_curves[String(leg.j)];
    plot('p1',[{x:mc.t,y:mc.v,color:'#ffd24a'}],{
      title:'node '+leg.j+' mission value v(t)   (green = arrival, blue = now)',
      xmin:0, xmax:VX, ymin:0, ymax:VMAX,
      vlines:[{x:leg.t_arr,color:'#3ad07a'},{x:fr.t,color:'#4aa3ff'}],
      markers:[{x:leg.t_arr,y:leg.arr_value,color:'#3ad07a'}]});
    altPanel('p2');
    plot('p3',[{x:tx,y:frames.slice(0,k+1).map(f=>f.cum_value),color:'#3ad07a'}],{
      title:'cumulative mission value vs elapsed day',
      xmin:0,xmax:GX, ymin:0, ymax:VALMAX,
      hlines:[{y:M.total_value,color:'#7a8aa0',dash:[2,3]}],
      markers:[{x:fr.t,y:fr.cum_value,color:'#3ad07a'}]});
    dvPanel('p4');
  } else {
    const cur=leg.curve;
    plot('p1',[{x:cur.t,y:cur.tof,color:'#4aa3ff'}],{
      title:'arc '+leg.i+'→'+leg.j+'  ToF vs departure day',
      xmin:AX_LO, xmax:AX_HI, ymin:0, ymax:TOF_HI,
      vlines:[{x:cur.t_dep,color:'#3ad07a'}], markers:[{x:cur.t_dep,y:cur.tof_dep,color:'#3ad07a'}]});
    plot('p2',[{x:cur.t,y:cur.dv,color:'#ff5b5b'}],{
      title:'arc '+leg.i+'→'+leg.j+'  ΔV vs departure day',
      xmin:AX_LO, xmax:AX_HI, ymin:0, ymax:DVC_HI,
      hlines:[{y:cur.floor,color:'#7a8aa0'}],
      vlines:[{x:cur.t_dep,color:'#3ad07a'}], markers:[{x:cur.t_dep,y:cur.dv_dep,color:'#3ad07a'}]});
    altPanel('p3');
    dvPanel('p4');
  }
}

// ---------- status ----------
document.getElementById('dv_bud').textContent=M.dv_budget.toFixed(0);
document.getElementById('valbox').style.display = M.has_mission ? '' : 'none';
function updateStatus(k){ const fr=frames[k]; const leg=legs[fr.leg];
  document.getElementById('t_now').textContent=fr.t.toFixed(2);
  document.getElementById('dv_now').textContent=fr.cum_dv.toFixed(0);
  if(M.has_mission){
    document.getElementById('val_now').textContent=(fr.cum_value||0).toFixed(2);
    document.getElementById('val_tot').textContent=(M.total_value||0).toFixed(2);
  }
  // the two timescales: month-long waits vs the ~50-minute transfer
  let s;
  if(fr.flying){
    s='TRANSFER '+leg.i+'→'+leg.j+'  (Hohmann, '+(leg.tof*1440).toFixed(0)+' min)';
  } else if(fr.t>=M.total_days-1e-9){
    s='ARRIVED @ '+leg.j+'  — rendezvous complete';
  } else {
    s='waiting @ '+leg.i+'  for the '+leg.j+' window/phase';
  }
  document.getElementById('leg_now').textContent=s;
  document.getElementById('scale_now').textContent =
    fr.flying ? 'minutes/frame' : 'days/frame';
}

// ---------- animation loop + solution selector ----------
let idx=0, playing=true;
const scrub=document.getElementById('scrub');
const speedEl=document.getElementById('speed');
const playBtn=document.getElementById('play');
const solsel=document.getElementById('solsel');
LABELS.forEach((lab,i)=>{ const o=document.createElement('option');
  o.value=i; o.textContent=(i+1)+'.  '+lab; solsel.appendChild(o); });
function loadTraj(i){ T=TRAJS[i]; frames=T.frames; legs=T.legs; M=T.meta;
  document.getElementById('t_tot').textContent=M.total_days.toFixed(0);
  document.getElementById('valbox').style.display = M.has_mission ? '' : 'none';
  idx=0; scrub.max=frames.length-1; scrub.value=0; render(); }
solsel.onchange=()=>{ playing=false; playBtn.textContent='Play'; loadTraj(+solsel.value); };
playBtn.onclick=()=>{ playing=!playing; playBtn.textContent=playing?'Pause':'Play'; };
scrub.oninput=()=>{ idx=+scrub.value; playing=false; playBtn.textContent='Play'; render(); };
scrub.max=frames.length-1;
document.getElementById('t_tot').textContent=M.total_days.toFixed(0);

const FPS=60; let acc=0;
// setInterval drives advancement (robust even when requestAnimationFrame is
// throttled); rendering happens on each tick and on camera interaction.
setInterval(()=>{ if(!playing) return;
  acc+=(+speedEl.value)/FPS; while(acc>=1){ idx=(idx+1)%frames.length; acc-=1; }
  scrub.value=idx; render(); }, 1000/FPS);
function render(){ const k=Math.min(idx,frames.length-1);
  updateScene(frames[k]); updatePanels(k); updateStatus(k);
  controls.update(); renderer.render(scene,camera);
}
function resize(){ const r=sceneEl.getBoundingClientRect();
  renderer.setSize(r.width,r.height); camera.aspect=r.width/r.height; camera.updateProjectionMatrix();
  render(); }
window.addEventListener('resize',resize);
// A ResizeObserver also catches the FIRST real layout: when this script runs the
// pane can still be 0-high, and a window 'resize' may never fire afterwards --
// which would leave the WebGL canvas 0x0 and the 3D view permanently blank.
if(window.ResizeObserver){ new ResizeObserver(()=>resize()).observe(sceneEl); }
controls.addEventListener('change', render);
resize(); render();
</script>
</body></html>
"""
