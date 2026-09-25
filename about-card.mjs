// The "about" card: what this is, and the papers it was built from.
//
// ONE MODULE for every page that shows it, for the reason ui-chrome.mjs and
// error-overlay.mjs give: near-identical copies across the main*.js pages are
// how one gets updated and the others not. The references live HERE, once.
//
// It builds its own button and card, so a page adopts it with one call and no
// markup. Call it at MODULE TOP LEVEL, not inside init(): it needs nothing
// from the GPU, and a page whose WebGPU setup failed is exactly one where
// someone may want to read what it was supposed to be.
//
// Every DOI below was resolved against Crossref when this was written
// (2026-09-24), not typed from memory -- one that had been was wrong. Entries
// with no DOI (Schiller & Naumann 1933, Revelles et al. 2000) carry none
// rather than a guessed link.

const REFS = [
  { group: 'The falling card', items: [
    { cite: 'Pesavento, U. & Wang, Z. J. (2004). Falling paper: Navier-Stokes solutions, model of fluid forces, and center of mass elevation. Phys. Rev. Lett. 93, 144501.',
      doi: '10.1103/PhysRevLett.93.144501',
      note: 'The problem this page reproduces, and its parameters.' },
    { cite: 'Andersen, A., Pesavento, U. & Wang, Z. J. (2005). Unsteady aerodynamics of fluttering and tumbling plates. J. Fluid Mech. 541, 65-90.',
      doi: '10.1017/S002211200500594X',
      note: 'Where flutter turns into tumbling.' },
    { cite: 'Field, S. B., Klaus, M., Moore, M. G. & Nori, F. (1997). Chaotic dynamics of falling disks. Nature 388, 252-254.',
      doi: '10.1038/40817',
      note: 'The regime map in dimensionless moment of inertia (I*).' },
  ]},
  { group: 'Adaptive mesh refinement', items: [
    { cite: 'Jaber, K., Essel, E. E. & Sullivan, P. E. (2025). GPU-native adaptive mesh refinement with application to lattice Boltzmann simulations. Comput. Phys. Commun. 311, 109543.',
      doi: '10.1016/j.cpc.2025.109543', arxiv: '2308.08085',
      note: 'The AMR design (AGAL), including the vorticity refinement ladder.' },
    { cite: 'Chen, H., Filippova, O., Hoch, J., Molvig, K., Shock, R., Teixeira, C. & Zhang, R. (2006). Grid refinement in lattice Boltzmann methods based on volumetric formulation. Physica A 362, 158-167.',
      doi: '10.1016/j.physa.2005.09.036',
      note: 'Explode/coalesce: the coarse/fine interface used here.' },
    { cite: 'Chen, H. (1998). Volumetric formulation of the lattice Boltzmann method for fluid dynamics: basic concept. Phys. Rev. E 58, 3955-3963.',
      doi: '10.1103/PhysRevE.58.3955',
      note: 'Why that interface conserves mass at a corner.' },
    { cite: 'Rohde, M., Kandhai, D., Derksen, J. J. & van den Akker, H. E. A. (2006). A generic, mass conservative local grid refinement technique for lattice-Boltzmann schemes. Int. J. Numer. Methods Fluids 51, 439-468.',
      doi: '10.1002/fld.1140',
      note: 'A mass-conservative alternative, read alongside Chen et al.' },
    { cite: 'Dupuis, A. & Chopard, B. (2003). Theory and applications of an alternative lattice Boltzmann grid refinement algorithm. Phys. Rev. E 67, 066707.',
      doi: '10.1103/PhysRevE.67.066707',
      note: 'Rescaling populations between levels.' },
    { cite: 'Berger, M. J. & Colella, P. (1989). Local adaptive mesh refinement for shock hydrodynamics. J. Comput. Phys. 82, 64-84.',
      doi: '10.1016/0021-9991(89)90035-1',
      note: 'Refluxing, tried and superseded.' },
    { cite: 'Schornbaum, F. & Rüde, U. (2016). Massively parallel algorithms for the lattice Boltzmann method on nonuniform grids. SIAM J. Sci. Comput. 38, C96-C126.',
      doi: '10.1137/15m1035240',
      note: 'Direction-selective transfer across the interface (waLBerla).' },
  ]},
  { group: 'The lattice Boltzmann method', items: [
    { cite: 'Krüger, T., Kusumaatmaja, H., Kuzmin, A., Shardt, O., Silva, G. & Viggen, E. M. (2017). The Lattice Boltzmann Method: Principles and Practice. Springer.',
      doi: '10.1007/978-3-319-44649-3',
      note: 'The reference text.' },
    { cite: 'Guo, Z., Zheng, C. & Shi, B. (2002). Discrete lattice effects on the forcing term in the lattice Boltzmann method. Phys. Rev. E 65, 046308.',
      doi: '10.1103/PhysRevE.65.046308',
      note: 'Body forcing.' },
    { cite: 'Ladd, A. J. C. (1994). Numerical simulations of particulate suspensions via a discretized Boltzmann equation. Part 1. J. Fluid Mech. 271, 285-309.',
      doi: '10.1017/S0022112094001771',
      note: 'Bounce-back with moving walls; momentum exchange.' },
    { cite: 'Mei, R., Yu, D., Shyy, W. & Luo, L.-S. (2002). Force evaluation in the lattice Boltzmann method involving curved geometry. Phys. Rev. E 65, 041203.',
      doi: '10.1103/PhysRevE.65.041203',
      note: 'Measuring the force on a body.' },
    { cite: 'Lallemand, P. & Luo, L.-S. (2003). Lattice Boltzmann method for moving boundaries. J. Comput. Phys. 184, 406-421.',
      doi: '10.1016/S0021-9991(02)00022-0',
      note: 'Refilling cells a moving body uncovers.' },
    { cite: 'Ginzburg, I., Verhaeghe, F. & d\'Humières, D. (2008). Two-relaxation-time lattice Boltzmann scheme: about parametrization, velocity, pressure and mixed boundary conditions. Commun. Comput. Phys. 3, 427-478.',
      doi: '10.4208/cicp.2008.v3.p427',
      note: 'The TRT collision, measured and not adopted.' },
  ]},
  { group: 'Validation references', items: [
    { cite: 'Schiller, L. & Naumann, A. (1933). Über die grundlegenden Berechnungen bei der Schwerkraftaufbereitung. Z. Ver. Dtsch. Ing. 77, 318-320.',
      note: 'Sphere drag correlation.' },
    { cite: 'Johnson, T. A. & Patel, V. C. (1999). Flow past a sphere up to a Reynolds number of 300. J. Fluid Mech. 378, 19-70.',
      doi: '10.1017/S0022112098003206',
      note: 'Sphere wake and shedding.' },
    { cite: 'Williamson, C. H. K. (1996). Vortex dynamics in the cylinder wake. Annu. Rev. Fluid Mech. 28, 477-539.',
      doi: '10.1146/annurev.fl.28.010196.002401',
      note: 'Cylinder Strouhal number.' },
    { cite: 'Henderson, R. D. (1995). Details of the drag curve near the onset of vortex shedding. Phys. Fluids 7, 2102-2104.',
      doi: '10.1063/1.868459',
      note: 'Cylinder drag.' },
    { cite: 'Dennis, S. C. R. & Chang, G.-Z. (1970). Numerical solutions for steady flow past a circular cylinder at Reynolds numbers up to 100. J. Fluid Mech. 42, 471-489.',
      doi: '10.1017/S0022112070001428',
      note: 'Steady cylinder flow.' },
    { cite: 'Coutanceau, M. & Bouard, R. (1977). Experimental determination of the main features of the viscous flow in the wake of a circular cylinder in uniform translation. Part 1. J. Fluid Mech. 79, 231-256.',
      doi: '10.1017/S0022112077000135',
      note: 'Recirculation length.' },
    { cite: 'Braza, M., Chassaing, P. & Ha Minh, H. (1986). Numerical study and physical analysis of the pressure and velocity fields in the near wake of a circular cylinder. J. Fluid Mech. 165, 79-130.',
      doi: '10.1017/S0022112086003014',
      note: 'Cylinder shedding.' },
    { cite: 'He, X. & Doolen, G. (1997). Lattice Boltzmann method on curvilinear coordinates system: flow around a circular cylinder. J. Comput. Phys. 134, 306-315.',
      doi: '10.1006/jcph.1997.5709',
      note: 'Cylinder flow with LBM.' },
    { cite: 'Brachet, M. E., Meiron, D. I., Orszag, S. A., Nickel, B. G., Morf, R. H. & Frisch, U. (1983). Small-scale structure of the Taylor-Green vortex. J. Fluid Mech. 130, 411-452.',
      doi: '10.1017/S0022112083001159',
      note: 'Taylor-Green vortex.' },
  ]},
  { group: 'Rendering', items: [
    { cite: 'Reinhard, E., Stark, M., Shirley, P. & Ferwerda, J. (2002). Photographic tone reproduction for digital images. ACM Trans. Graph. 21, 267-276.',
      doi: '10.1145/566654.566575',
      note: 'The tone curve on the vorticity colours.' },
    { cite: 'Wald, I., Zellmann, S., Usher, W., Morrical, N., Lang, U. & Pascucci, V. (2021). Ray tracing structured AMR data using ExaBricks. IEEE Trans. Vis. Comput. Graph. 27, 625-634.',
      doi: '10.1109/TVCG.2020.3030470',
      note: 'Rendering AMR volumes (3D view).' },
    { cite: 'Museth, K. (2021). NanoVDB: a GPU-friendly and portable VDB data structure for real-time rendering and simulation. ACM SIGGRAPH 2021 Talks.',
      doi: '10.1145/3450623.3464653',
      note: 'Sparse volume layout (3D view).' },
    { cite: 'Revelles, J., Ureña, C. & Lastra, M. (2000). An efficient parametric algorithm for octree traversal. WSCG 2000.',
      note: 'Octree ray traversal (3D view).' },
  ]},
];

// The source repository, derived from where the page is served rather than
// written in: a GitHub Pages URL is https://OWNER.github.io/REPO/..., and the
// project's convention is never to hardcode its own owner/name. Anywhere else
// (the local dev server) there is simply no link.
function sourceUrl() {
  const m = /^([^.]+)\.github\.io$/.exec(location.hostname);
  const repo = location.pathname.split('/').filter(Boolean)[0];
  return m && repo ? `https://github.com/${m[1]}/${repo}` : null;
}

const CSS = `
#about {
  position: absolute; bottom: 8px; left: 104px;  /* right of #pause */
  font-size: 12px; color: #bbb; background: rgba(0,0,0,0.55);
  padding: 4px 10px; border: 1px solid #444; border-radius: 3px;
  cursor: pointer; font-family: monospace; opacity: 0.55;
  transition: opacity 0.15s ease;
}
#about:hover, #about:focus-visible { opacity: 1; background: rgba(50,50,50,0.8); }
#about[aria-expanded="true"] { opacity: 1; color: #9fd3ff; border-color: #2d5a7a; }
#about-card {
  position: absolute; left: 50%; top: 44px; bottom: 44px;
  transform: translateX(-50%);
  width: min(680px, calc(100% - 24px));
  overflow-y: auto; overscroll-behavior: contain;
  background: rgba(8,10,16,0.94); color: #ddd;
  border: 1px solid #444; border-radius: 6px;
  padding: 14px 18px 16px;
  font: 12px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  z-index: 5;
}
#about-card[hidden] { display: none; }
#about-card h2 { font-size: 15px; margin: 0 28px 6px 0; color: #fff; }
#about-card h3 { font-size: 11px; margin: 14px 0 4px; color: #8ab; text-transform: uppercase; letter-spacing: 0.06em; }
#about-card p { margin: 0 0 6px; color: #bbb; }
#about-card ol { margin: 0; padding-left: 18px; }
#about-card li { margin: 0 0 7px; }
#about-card .note { display: block; color: #8a8f99; font-style: italic; }
#about-card a { color: #8cc8ff; }
#about-card a:hover { color: #bde0ff; }
#about-card .links a { margin-right: 10px; font-family: monospace; font-size: 11px; }
#about-close {
  position: absolute; top: 8px; right: 10px;
  background: none; border: none; color: #999; font-size: 18px; line-height: 1;
  cursor: pointer; padding: 2px 6px;
}
#about-close:hover, #about-close:focus-visible { color: #fff; }
`;

export function installAboutCard(container) {
  if (!container || document.getElementById('about')) return null;

  const style = document.createElement('style');
  style.textContent = CSS;
  document.head.appendChild(style);

  const button = document.createElement('button');
  button.id = 'about';
  button.textContent = 'about';
  button.setAttribute('aria-controls', 'about-card');
  button.setAttribute('aria-expanded', 'false');

  const card = document.createElement('div');
  card.id = 'about-card';
  card.setAttribute('role', 'dialog');
  card.setAttribute('aria-label', 'About getting-air, and references');
  card.hidden = true;

  const el = (tag, props = {}, ...kids) => {
    const n = document.createElement(tag);
    Object.assign(n, props);
    for (const k of kids) if (k != null) n.append(k);
    return n;
  };
  const link = (href, text) => el('a', { href, textContent: text, target: '_blank', rel: 'noopener noreferrer' });

  const close = el('button', { id: 'about-close', textContent: '×', title: 'Close' });
  close.setAttribute('aria-label', 'Close');
  card.append(close, el('h2', { textContent: 'getting-air' }));
  card.append(el('p', { textContent:
    'A falling card, simulated with the lattice Boltzmann method on the GPU (WebGPU). '
    + 'The project also has adaptive-mesh-refinement and 3D versions of the solver. '
    + 'These are the papers the code was built from and checked against.' }));
  const src = sourceUrl();
  if (src) card.append(el('p', {}, 'Source: ', link(src, src.replace(/^https:\/\//, ''))));

  for (const { group, items } of REFS) {
    card.append(el('h3', { textContent: group }));
    const ol = el('ol');
    for (const r of items) {
      const links = el('span', { className: 'links' });
      if (r.doi) links.append(link(`https://doi.org/${r.doi}`, 'doi'));
      if (r.arxiv) links.append(link(`https://arxiv.org/abs/${r.arxiv}`, 'arXiv'));
      ol.append(el('li', {}, r.cite + ' ', links, el('span', { className: 'note', textContent: r.note })));
    }
    card.append(ol);
  }

  const set = (open) => {
    card.hidden = !open;
    button.setAttribute('aria-expanded', open ? 'true' : 'false');
  };
  button.onclick = () => set(card.hidden);
  close.onclick = () => { set(false); button.focus(); };
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !card.hidden) { set(false); button.focus(); }
  });

  container.append(button, card);
  return { open: () => set(true), close: () => set(false), isOpen: () => !card.hidden };
}
