/* Page text from content.json, the setup schematic, and the four message sections. */
(function () {
  "use strict";
  var V = window.VIZ;

  /* ---- theme toggle: auto -> light -> dark ---- */
  var btn = document.getElementById("theme-btn");
  function currentTheme() { return document.documentElement.getAttribute("data-theme") || "auto"; }
  function setThemeLabel() {
    var th = currentTheme();
    btn.textContent = "Theme: " + th;
    btn.setAttribute("aria-label", "Color theme: " + (th === "auto" ? "automatic" : th) + ". Activate to change.");
  }
  btn.addEventListener("click", function () {
    var order = ["auto", "light", "dark"];
    var next = order[(order.indexOf(currentTheme()) + 1) % order.length];
    if (next === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", next);
    try { if (next === "auto") localStorage.removeItem("theme"); else localStorage.setItem("theme", next); } catch (e) {}
    setThemeLabel();
    V.rerenderAll();
  });
  setThemeLabel();
  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    var onMq = function () { if (currentTheme() === "auto") V.rerenderAll(); };
    if (mq.addEventListener) mq.addEventListener("change", onMq); else if (mq.addListener) mq.addListener(onMq);
  }

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  var ICON = {
    arxiv: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 1h7l3 3v11H3z" opacity=".35"/><path d="M5 7h6v1.2H5zm0 2.5h6v1.2H5zm0 2.5h4v1.2H5z"/></svg>',
    pdf: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 1h7l3 3v11H3z" opacity=".35"/><path d="M7.4 5h1.2v4.1l1.5-1.5.85.85L8 11.4 5.05 8.45l.85-.85 1.5 1.5z"/></svg>',
    code: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'
  };

  function renderHero(C) {
    document.getElementById("title").textContent = C.title;
    document.getElementById("subtitle").textContent = C.subtitle;
    var au = document.getElementById("authors");
    au.innerHTML = "";
    C.authors.forEach(function (a, i) {
      var s = el("span");
      if (a.url) {
        var ln = el("a"); ln.href = a.url; ln.textContent = a.name;
        ln.target = "_blank"; ln.rel = "noopener";
        s.appendChild(ln);
      } else {
        s.appendChild(document.createTextNode(a.name));
      }
      var sp = el("sup", null, a.affiliations.join(","));
      s.appendChild(sp);
      au.appendChild(s);
      if (i < C.authors.length - 1) au.appendChild(document.createTextNode(", "));
    });
    var af = document.getElementById("affils");
    af.innerHTML = "";
    C.affiliations.forEach(function (name, i) {
      var s = el("span");
      s.appendChild(el("sup", null, String(i + 1)));
      s.appendChild(document.createTextNode(name));
      af.appendChild(s);
      af.appendChild(document.createTextNode(" "));
    });
    var ln = document.getElementById("links");
    ln.innerHTML = "";
    [["arXiv", C.links.arxiv, ICON.arxiv], ["PDF", C.links.pdf, ICON.pdf], ["Code", C.links.code, ICON.code]].forEach(function (x) {
      if (!x[1]) return;
      var a = el("a", "btn");
      a.href = x[1];
      a.innerHTML = x[2];
      a.appendChild(document.createTextNode(x[0]));
      ln.appendChild(a);
    });
    document.getElementById("abstract").textContent = C.abstract;
    if (window.innerWidth < 700) document.getElementById("abstract-box").open = false;
    var fd = document.getElementById("findings");
    fd.innerHTML = "";
    C.findings.forEach(function (f, i) {
      var a = el("a", "finding");
      a.href = "#m" + (i + 1);
      a.appendChild(el("span", "n", "Message " + (i + 1)));
      a.appendChild(el("h3", null, f.title.replace(/\.$/, "")));
      a.appendChild(el("p", null, f.text));
      fd.appendChild(a);
    });
    document.getElementById("takeaway-text").textContent = C.takeaway;
    document.getElementById("bibtex").textContent = C.bibtex;
    if (C.links.code) {
      var fc = document.getElementById("foot-code");
      fc.href = C.links.code;
      fc.textContent = C.links.code.replace(/^https?:\/\//, "");
    }
  }

  function renderSetup(C) {
    document.getElementById("setup-text").textContent = C.setup;
    var sc = document.getElementById("schematic");
    sc.innerHTML = "";
    var draw = { "Voting": SCHEMATIC.voting, "Orch-Workers": SCHEMATIC.orch, "Eval-Opt": SCHEMATIC.evalopt };
    var order = ["Voting", "Orch-Workers", "Eval-Opt"];
    order.forEach(function (name) {
      var w = C.workflows.filter(function (x) { return x.name === name; })[0];
      if (!w) return;
      var card = el("div", "wf-card");
      card.appendChild(el("h3", null, w.name));
      card.appendChild(el("p", null, w.one_line));
      var holder = el("div");
      holder.innerHTML = draw[name]();
      card.appendChild(holder);
      sc.appendChild(card);
    });
    var key = document.getElementById("node-key");
    key.innerHTML = "";
    [["Solving role", "var(--node-solving)"], ["Decision role", "var(--node-decision)"], ["Orchestrator (planning)", "var(--node-planning)"]].forEach(function (k) {
      var s = el("span");
      var i = el("i");
      i.style.background = k[1];
      s.appendChild(i);
      s.appendChild(document.createTextNode(k[0]));
      key.appendChild(s);
    });
    var st = el("span");
    st.innerHTML = '<svg width="24" height="16" viewBox="0 0 24 16" aria-hidden="true"><rect x="7" y="1" width="15" height="10" rx="2" fill="var(--node-solving)" stroke="var(--node-stroke)"/><rect x="4.5" y="3" width="15" height="10" rx="2" fill="var(--node-solving)" stroke="var(--node-stroke)"/><rect x="2" y="5" width="15" height="10" rx="2" fill="var(--node-solving)" stroke="var(--node-stroke)"/></svg>';
    st.appendChild(document.createTextNode("Parallel role: three instances per episode"));
    key.appendChild(st);
    var rw = el("span");
    rw.innerHTML = '<svg width="22" height="14" viewBox="0 0 22 14" aria-hidden="true"><line x1="11" y1="13" x2="11" y2="3" stroke="var(--c4)" stroke-width="1.6"/><path d="M7.5 5 L11 1 L14.5 5" fill="none" stroke="var(--c4)" stroke-width="1.6"/></svg>';
    rw.appendChild(document.createTextNode("Broadcast outcome reward"));
    key.appendChild(rw);

    document.getElementById("policy-card").innerHTML = '<h3 style="margin:0 0 8px">Policy sharing</h3>' + SCHEMATIC.policyCard();
    var g = document.getElementById("grid-card");
    g.innerHTML = "";
    g.appendChild(el("h3", null, "Experimental design grid"));
    g.firstChild.style.margin = "0 0 10px";
    var dl = el("dl");
    [
      ["Workflows", "Voting, Orch-Workers, Eval-Opt"],
      ["Strategies", "Isolated policies (IP), shared policy (SP)"],
      ["Scales", "Qwen3 0.6B, 1.7B, 4B"],
      ["Tasks", "Math (DAPO-Math-17K), Code (DeepCoder)"],
      ["Cells", "3 workflows × 2 strategies × 3 scales × 2 tasks = 36, each trained once with GRPO"],
      ["Baselines", "Base model, alone and in every role; single-agent RL (SA-RL), as one response and placed in every role of the untrained workflow"],
      ["Accuracy", "Validation pass@1 at the selected peak, the best checkpoint at step ≤ 300; differences below about 2 points are ties"]
    ].forEach(function (r) {
      dl.appendChild(el("dt", null, r[0]));
      dl.appendChild(el("dd", null, r[1]));
    });
    g.appendChild(dl);
  }

  function renderMessages(C, data) {
    var host = document.getElementById("messages");
    host.innerHTML = "";
    C.messages.forEach(function (m, i) {
      var sec = el("section", "block wrap");
      sec.id = m.id;
      sec.setAttribute("aria-labelledby", m.id + "-h");
      var lab = el("div", "msg-label");
      lab.appendChild(el("span", "num", "Message " + (i + 1)));
      lab.appendChild(el("span", "sec", "Section " + m.section));
      sec.appendChild(lab);
      var h = el("h2", null, m.question);
      h.id = m.id + "-h";
      sec.appendChild(h);
      var ans = el("div", "answer");
      ans.appendChild(el("div", "lab", "Answer"));
      ans.appendChild(el("p", null, m.answer));
      sec.appendChild(ans);
      var figs = el("div", "figs");
      sec.appendChild(figs);
      var res = el("div", "result");
      res.appendChild(el("h3", null, "Result"));
      res.appendChild(el("p", null, m.result));
      sec.appendChild(res);
      var ev = el("div", "evidence");
      ev.appendChild(el("h3", null, "Evidence"));
      m.evidence.forEach(function (e) {
        var d = el("details");
        d.appendChild(el("summary", null, e.title));
        d.appendChild(el("p", null, e.text));
        ev.appendChild(d);
      });
      sec.appendChild(ev);
      host.appendChild(sec);
      var D = data[m.id];
      if (D && window.FIGURES[m.id]) {
        try { window.FIGURES[m.id](figs, D); } catch (e) {
          figs.appendChild(el("p", "error", "The figures for this section could not be drawn."));
          if (window.console) console.error(e);
        }
      }
    });
  }

  document.getElementById("copy-bib").addEventListener("click", function () {
    var b = this, txt = document.getElementById("bibtex").textContent;
    function done(ok) { b.textContent = ok ? "Copied" : "Select and copy"; setTimeout(function () { b.textContent = "Copy"; }, 1600); }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(txt).then(function () { done(true); }, function () { done(false); });
    } else {
      var r = document.createRange();
      r.selectNodeContents(document.getElementById("bibtex"));
      var sel = window.getSelection();
      sel.removeAllRanges(); sel.addRange(r);
      var ok = false;
      try { ok = document.execCommand("copy"); } catch (e) {}
      done(ok);
    }
  });

  function load(name) {
    return fetch("data/" + name + ".json").then(function (r) {
      if (!r.ok) throw new Error(name + ": HTTP " + r.status);
      return r.json();
    });
  }
  Promise.all(["content", "m1_gain", "m2_eval_opt", "m3_ip_vs_sp", "m4_decline"].map(load)).then(function (res) {
    var C = res[0];
    renderHero(C);
    renderSetup(C);
    if (!window.Plotly) {
      document.getElementById("messages").innerHTML = '<p class="wrap error">The charting library did not load; the figures need a network connection.</p>';
      return;
    }
    renderMessages(C, { m1: res[1], m2: res[2], m3: res[3], m4: res[4] });
    if (location.hash) {
      var t = document.getElementById(location.hash.slice(1));
      if (t) setTimeout(function () { t.scrollIntoView(); }, 50);
    }
  }).catch(function (err) {
    var l = document.getElementById("messages");
    l.innerHTML = "";
    var p = el("p", "wrap error", "The page data could not be loaded (" + err.message + "). If you opened this file directly, serve the folder over HTTP instead.");
    l.appendChild(p);
  });
})();
