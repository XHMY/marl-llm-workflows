/* Chart helpers and the figure framework. Colors come from the CSS tokens in style.css. */
(function () {
  "use strict";

  var WF_ORDER = ["Voting", "Orch-Workers", "Eval-Opt"];
  var SCALES = ["0.6B", "1.7B", "4B"];
  var WF_SYMBOL = { "Voting": "circle", "Orch-Workers": "square", "Eval-Opt": "triangle-up" };

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /* Theme tokens, read fresh on every render so the charts follow the page theme. */
  function tokens() {
    var t = {
      page: cssVar("--page"), surface: cssVar("--surface"), surface2: cssVar("--surface-2"),
      ink: cssVar("--ink"), ink2: cssVar("--ink-2"), muted: cssVar("--muted"), axisInk: cssVar("--axis-ink"),
      grid: cssVar("--grid"), axis: cssVar("--axis"),
      c: [null, cssVar("--c1"), cssVar("--c2"), cssVar("--c3"), cssVar("--c4"), cssVar("--c5"), cssVar("--c6"), cssVar("--c7"), cssVar("--c8")],
      neutralBar: cssVar("--neutral-bar"), neutralMark: cssVar("--neutral-mark"),
      seqLo: cssVar("--seq-lo"), seqHi: cssVar("--seq-hi"),
      font: cssVar("--font") || "system-ui, sans-serif"
    };
    t.wf = { "Voting": t.c[1], "Orch-Workers": t.c[2], "Eval-Opt": t.c[3] };
    return t;
  }

  function hexToRgb(hex) {
    var h = hex.replace("#", "");
    if (h.length === 3) h = h.split("").map(function (x) { return x + x; }).join("");
    var n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  /* A light step of a hue: mixes the hue toward the surface (opaque, so gaps stay clean). */
  function tint(hex, amount, surface) {
    var a = hexToRgb(hex), s = hexToRgb(surface || "#ffffff");
    var k = amount == null ? 0.45 : amount;
    var m = a.map(function (v, i) { return Math.round(v * k + s[i] * (1 - k)); });
    return "rgb(" + m.join(",") + ")";
  }
  function alpha(hex, a) {
    var c = hexToRgb(hex);
    return "rgba(" + c.join(",") + "," + a + ")";
  }

  /* number formatting */
  function f1(v) { return v == null || isNaN(v) ? "—" : Number(v).toFixed(1); }
  function f2(v) { return v == null || isNaN(v) ? "—" : Number(v).toFixed(2); }
  function sgn(v, d) {
    if (v == null || isNaN(v)) return "—";
    var s = Number(v).toFixed(d == null ? 1 : d);
    if (Number(s) === 0) s = Number(0).toFixed(d == null ? 1 : d);
    return (Number(s) > 0 ? "+" : Number(s) < 0 ? "−" : "") + s.replace("-", "");
  }
  function kn(o) { return o ? o.k + "/" + o.n : "—"; }
  function pct(v, d) { return v == null ? "—" : (v * 100).toFixed(d == null ? 1 : d) + "%"; }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  /* Plotly layout scaffolding with recessive axes */
  function axis(t, extra) {
    var a = {
      gridcolor: t.grid, gridwidth: 1, linecolor: t.axis, linewidth: 1, showline: true,
      zeroline: false, ticks: "outside", ticklen: 3, tickcolor: t.axis,
      tickfont: { color: t.ink2, size: 11 }, title: { font: { color: t.ink2, size: 12 }, standoff: 8 },
      automargin: true, fixedrange: true
    };
    return Object.assign(a, extra || {});
  }
  function layout(t, extra) {
    var l = {
      paper_bgcolor: t.surface, plot_bgcolor: t.surface,
      font: { family: t.font, size: 12, color: t.ink2 },
      margin: { l: 52, r: 14, t: 10, b: 44 },
      showlegend: false, hovermode: "closest", dragmode: false,
      hoverlabel: { bgcolor: t.surface, bordercolor: t.axis, font: { color: t.ink, size: 12, family: t.font }, align: "left" },
      xaxis: axis(t), yaxis: axis(t),
      barcornerradius: 4
    };
    Object.keys(extra || {}).forEach(function (k) {
      if ((k.indexOf("xaxis") === 0 || k.indexOf("yaxis") === 0) && typeof extra[k] === "object") {
        l[k] = axis(t, extra[k]);
      } else {
        l[k] = extra[k];
      }
    });
    return l;
  }
  var PLOT_CONFIG = { displayModeBar: false, responsive: true, scrollZoom: false };

  /* ---- legend swatches (mirror the mark: line for lines, rect for bars, dot for dots) ---- */
  function swatch(it) {
    var c = it.color || "currentColor";
    var w = 22, h = 12, s = "";
    switch (it.type) {
      case "line":
        s = '<line x1="1" y1="6" x2="21" y2="6" stroke="' + c + '" stroke-width="2.2" stroke-linecap="round"' +
          (it.dash === "dash" ? ' stroke-dasharray="5 3"' : it.dash === "dot" ? ' stroke-dasharray="1.5 3"' : it.dash === "dashdot" ? ' stroke-dasharray="6 2.5 1.5 2.5"' : "") + "/>";
        if (it.marker) s += '<circle cx="11" cy="6" r="3.2" fill="' + (it.open ? "none" : c) + '" stroke="' + c + '" stroke-width="1.5"/>';
        break;
      case "rect":
        s = '<rect x="3" y="1" width="16" height="10" rx="2.5" fill="' + c + '"/>';
        break;
      case "split":
        var cs = it.colors;
        s = cs.map(function (cc, i) { return '<rect x="' + (2 + i * 6) + '" y="1" width="5.3" height="10" rx="1.5" fill="' + cc + '"/>'; }).join("");
        break;
      case "dot":
        s = '<circle cx="11" cy="6" r="4.2" fill="' + (it.open ? "none" : c) + '" stroke="' + c + '" stroke-width="1.6"/>';
        break;
      case "diamond":
        s = '<path d="M11 0.8 L16.2 6 L11 11.2 L5.8 6 Z" fill="' + (it.open ? "none" : c) + '" stroke="' + c + '" stroke-width="1.6"' + (it.dashOutline ? ' stroke-dasharray="2 1.5"' : "") + "/>";
        break;
      case "square":
        s = '<rect x="6.8" y="1.8" width="8.4" height="8.4" fill="' + (it.open ? "none" : c) + '" stroke="' + c + '" stroke-width="1.6"/>';
        break;
      case "triangle":
        s = '<path d="M11 1.2 L16 10.5 L6 10.5 Z" fill="' + (it.open ? "none" : c) + '" stroke="' + c + '" stroke-width="1.6"/>';
        break;
      case "tick":
        s = it.horizontal
          ? '<line x1="4" y1="6" x2="18" y2="6" stroke="' + c + '" stroke-width="2.6"/>'
          : '<line x1="11" y1="0.5" x2="11" y2="11.5" stroke="' + c + '" stroke-width="2.6"/>';
        break;
      case "band":
        s = '<rect x="1" y="1" width="20" height="10" rx="2" fill="' + c + '"/>';
        break;
      case "errbar":
        s = '<line x1="3" y1="6" x2="19" y2="6" stroke="' + c + '" stroke-width="1.2"/><line x1="3" y1="2.5" x2="3" y2="9.5" stroke="' + c + '" stroke-width="1.2"/><line x1="19" y1="2.5" x2="19" y2="9.5" stroke="' + c + '" stroke-width="1.2"/>';
        break;
      default:
        s = '<rect x="3" y="1" width="16" height="10" rx="2.5" fill="' + c + '"/>';
    }
    return '<svg width="' + w + '" height="' + h + '" viewBox="0 0 22 12" aria-hidden="true">' + s + "</svg>";
  }
  function renderLegend(el, items) {
    el.innerHTML = "";
    (items || []).forEach(function (it) {
      var span = document.createElement("span");
      span.className = "it";
      span.innerHTML = swatch(it);
      var lab = document.createElement("span");
      lab.textContent = it.label;
      span.appendChild(lab);
      el.appendChild(span);
    });
    el.style.display = items && items.length ? "" : "none";
  }

  function renderTable(el, tbl) {
    el.innerHTML = "";
    if (!tbl) return;
    var tables = Array.isArray(tbl) ? tbl : [tbl];
    tables.forEach(function (tb) {
      if (tb.title) {
        var h = document.createElement("div");
        h.className = "plot-title";
        h.textContent = tb.title;
        el.appendChild(h);
      }
      var wrap = document.createElement("div");
      wrap.className = "tbl-scroll";
      var table = document.createElement("table");
      table.className = "data";
      var thead = document.createElement("thead");
      var tr = document.createElement("tr");
      tb.columns.forEach(function (c, i) {
        var th = document.createElement("th");
        th.scope = "col";
        th.textContent = typeof c === "string" ? c : c.label;
        if (tb.numeric && tb.numeric[i]) th.className = "num";
        tr.appendChild(th);
      });
      thead.appendChild(tr);
      table.appendChild(thead);
      var tbody = document.createElement("tbody");
      tb.rows.forEach(function (r) {
        var row = document.createElement("tr");
        var cells = Array.isArray(r) ? r : r.cells;
        if (!Array.isArray(r) && r.sep) row.className = "sep";
        cells.forEach(function (v, i) {
          var td = document.createElement("td");
          td.textContent = v == null ? "—" : String(v);
          var cls = [];
          if (tb.numeric && tb.numeric[i]) cls.push("num");
          if (tb.noteCol === i) cls.push("note");
          if (tb.wrapCol === i) cls.push("wrap");
          td.className = cls.join(" ");
          row.appendChild(td);
        });
        tbody.appendChild(row);
      });
      table.appendChild(tbody);
      wrap.appendChild(table);
      el.appendChild(wrap);
    });
  }

  /* ---- controls ---- */
  var uid = 0;
  function buildControl(ctl, state, onChange) {
    var wrap = document.createElement("div");
    wrap.className = "ctl" + (ctl.type === "select" ? " sel" : "");
    if (ctl.label && ctl.type !== "check") {
      var lab = document.createElement("span");
      lab.className = "ctl-label";
      lab.textContent = ctl.label;
      lab.id = "ctl-" + (++uid);
      wrap.appendChild(lab);
    }
    if (ctl.type === "seg") {
      var g = document.createElement("div");
      g.className = "seg";
      g.setAttribute("role", "radiogroup");
      if (ctl.label) g.setAttribute("aria-labelledby", "ctl-" + uid);
      var name = "r" + (++uid);
      ctl.options.forEach(function (o) {
        var l = document.createElement("label");
        var inp = document.createElement("input");
        inp.type = "radio"; inp.name = name; inp.value = o.v;
        inp.checked = state[ctl.key] === o.v;
        inp.addEventListener("change", function () {
          if (inp.checked) { state[ctl.key] = o.v; onChange(); }
        });
        var sp = document.createElement("span");
        sp.textContent = o.l;
        l.appendChild(inp); l.appendChild(sp);
        g.appendChild(l);
      });
      wrap.appendChild(g);
    } else if (ctl.type === "check") {
      var l2 = document.createElement("label");
      l2.className = "chk";
      var cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = !!state[ctl.key];
      cb.addEventListener("change", function () { state[ctl.key] = cb.checked; onChange(); });
      var sp2 = document.createElement("span");
      sp2.textContent = ctl.label;
      l2.appendChild(cb);
      if (ctl.swatch) {
        var sw = document.createElement("span");
        sw.className = "sw";
        sw.setAttribute("data-sw", ctl.key);
        sw.setAttribute("aria-hidden", "true");
        l2.appendChild(sw);
      }
      l2.appendChild(sp2);
      wrap.appendChild(l2);
    } else if (ctl.type === "select") {
      var s = document.createElement("select");
      if (ctl.label) s.setAttribute("aria-labelledby", "ctl-" + uid);
      ctl.options.forEach(function (o) {
        var op = document.createElement("option");
        op.value = o.v; op.textContent = o.l;
        if (state[ctl.key] === o.v) op.selected = true;
        s.appendChild(op);
      });
      s.addEventListener("change", function () { state[ctl.key] = s.value; onChange(); });
      wrap.appendChild(s);
    }
    return wrap;
  }

  /* ---- figure framework ----
     spec: { id, num, title, caption, controls, defaults,
             render(state, t, width) -> { plots: [{data, layout, title}], legend: [...], badges: [...], table, html, twoCol } } */
  var FIGS = [];
  function figure(parent, spec) {
    var fig = document.createElement("figure");
    fig.className = "fig";
    fig.id = spec.id;
    var head = document.createElement("div");
    head.className = "fig-head";
    var num = document.createElement("span");
    num.className = "fig-num";
    num.textContent = spec.num;
    var h = document.createElement("h3");
    h.textContent = spec.title;
    head.appendChild(num); head.appendChild(h);
    fig.appendChild(head);

    var state = Object.assign({}, spec.defaults || {});
    var ctlRow = document.createElement("div");
    ctlRow.className = "controls";
    fig.appendChild(ctlRow);
    var badgeRow = document.createElement("div");
    badgeRow.className = "badge-row";
    fig.appendChild(badgeRow);
    var legend = document.createElement("div");
    legend.className = "legend";
    fig.appendChild(legend);
    var plots = document.createElement("div");
    plots.className = "plots";
    fig.appendChild(plots);
    var cap = document.createElement("figcaption");
    cap.textContent = spec.caption || "";
    fig.appendChild(cap);
    var tv = document.createElement("details");
    tv.className = "tableview";
    var sm = document.createElement("summary");
    sm.textContent = "Table view";
    tv.appendChild(sm);
    var tvBody = document.createElement("div");
    tv.appendChild(tvBody);
    fig.appendChild(tv);
    var below = document.createElement("div");
    fig.appendChild(below);
    parent.appendChild(fig);

    var obj = { spec: spec, state: state, el: fig, plotEls: [] };
    function update() {
      var t = tokens();
      var width = plots.clientWidth || fig.clientWidth || 800;
      var out;
      try {
        out = spec.render(state, t, width) || {};
      } catch (e) {
        plots.innerHTML = '<p class="error">This figure could not be drawn.</p>';
        if (window.console) console.error(spec.id, e);
        return;
      }
      renderLegend(legend, out.legend);
      if (out.swatches) {
        Object.keys(out.swatches).forEach(function (k) {
          var sw = ctlRow.querySelector('[data-sw="' + k + '"]');
          if (sw) sw.innerHTML = swatch(out.swatches[k]);
        });
      }
      badgeRow.innerHTML = "";
      (out.badges || []).forEach(function (b) {
        var s = document.createElement("span");
        s.className = "badge";
        if (b.strong) {
          var bb = document.createElement("b");
          bb.textContent = b.strong;
          s.appendChild(bb);
        }
        if (b.text) s.appendChild(document.createTextNode(b.text));
        badgeRow.appendChild(s);
      });
      badgeRow.style.display = out.badges && out.badges.length ? "" : "none";
      plots.className = "plots" + (out.twoCol ? " two" : "");
      var ps = out.plots || [];
      // rebuild plot containers when the count changes
      if (obj.plotEls.length !== ps.length || out.html) {
        obj.plotEls.forEach(function (p) { try { Plotly.purge(p.plot); } catch (e) {} });
        plots.innerHTML = "";
        obj.plotEls = ps.map(function () {
          var w = document.createElement("div");
          w.className = "plot-wrap";
          var tt = document.createElement("div");
          tt.className = "plot-title";
          var p = document.createElement("div");
          p.className = "plot";
          w.appendChild(tt); w.appendChild(p);
          plots.appendChild(w);
          return { wrap: w, title: tt, plot: p };
        });
        if (out.html) {
          var hp = document.createElement("div");
          hp.className = "html-panel";
          hp.appendChild(out.html);
          plots.appendChild(hp);
        }
      }
      ps.forEach(function (p, i) {
        var pe = obj.plotEls[i];
        pe.title.textContent = p.title || "";
        pe.title.style.display = p.title ? "" : "none";
        pe.plot.setAttribute("role", "img");
        pe.plot.setAttribute("aria-label", (p.title || spec.title) + ". Values are listed in the table view below the figure.");
        var lay = p.layout;
        pe.plot.style.height = (lay.height || 360) + "px";
        Plotly.react(pe.plot, p.data, lay, PLOT_CONFIG);
      });
      if (out.caption) cap.textContent = out.caption;
      renderTable(tvBody, out.table);
      tv.style.display = out.table ? "" : "none";
      below.innerHTML = "";
      if (out.below) below.appendChild(out.below);
    }
    (spec.controls || []).forEach(function (c) { ctlRow.appendChild(buildControl(c, state, update)); });
    ctlRow.style.display = spec.controls && spec.controls.length ? "" : "none";
    obj.update = update;
    FIGS.push(obj);
    update();
    return obj;
  }

  function rerenderAll() { FIGS.forEach(function (f) { f.update(); }); }

  var lastW = window.innerWidth, timer = null;
  window.addEventListener("resize", function () {
    clearTimeout(timer);
    timer = setTimeout(function () {
      if (Math.abs(window.innerWidth - lastW) < 8) return;
      lastW = window.innerWidth;
      rerenderAll();
    }, 180);
  });

  /* readable ticks for a log axis */
  function logTicks(lo, hi) {
    var cand = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200];
    var v = cand.filter(function (c) { return c >= lo * 0.999 && c <= hi * 1.001; });
    return { tickvals: v, ticktext: v.map(function (c) { return String(c); }) };
  }

  /* deterministic jitter */
  function jitter(i, amp) {
    var x = Math.sin((i + 1) * 12.9898) * 43758.5453;
    return ((x - Math.floor(x)) - 0.5) * 2 * amp;
  }

  window.VIZ = {
    WF_ORDER: WF_ORDER, SCALES: SCALES, WF_SYMBOL: WF_SYMBOL,
    tokens: tokens, tint: tint, alpha: alpha, f1: f1, f2: f2, sgn: sgn, kn: kn, pct: pct, esc: esc,
    axis: axis, layout: layout, figure: figure, rerenderAll: rerenderAll, jitter: jitter, swatch: swatch, renderTable: renderTable, logTicks: logTicks
  };
})();
