/* Interactive figures, one builder per figure, grouped by message. */
(function () {
  "use strict";
  var V = window.VIZ;
  var WF = V.WF_ORDER, SC = V.SCALES;
  var f1 = V.f1, f2 = V.f2, sgn = V.sgn, kn = V.kn, esc = V.esc;

  var SCALE_OPTS = SC.map(function (s) { return { v: s, l: s }; });
  var SCALE_OPTS_ALL = [{ v: "All", l: "All" }].concat(SCALE_OPTS);
  var TASK_OPTS = [{ v: "Math", l: "Math" }, { v: "Code", l: "Code" }];
  var WF_OPTS_ALL = [{ v: "All", l: "All" }].concat(WF.map(function (w) { return { v: w, l: w }; }));

  function cellName(wf, pol, scale, task) { return wf + "-" + pol + "-" + scale + "-" + task; }
  function narrow(w) { return w < 560; }
  function hoverLines(lines) { return lines.filter(function (x) { return x != null && x !== ""; }).join("<br>"); }
  function wfLegend(t, type, extra) {
    return WF.map(function (w) {
      return Object.assign({ type: type || "line", color: t.wf[w], label: w }, extra || {});
    });
  }
  function hline(y, color, width, dash, xref, yref) {
    return { type: "line", xref: xref || "paper", yref: yref || "y", x0: 0, x1: 1, y0: y, y1: y, line: { color: color, width: width || 1, dash: dash || "solid" }, layer: "below" };
  }
  function vline(x, color, width, dash, yref, xref) {
    return { type: "line", xref: xref || "x", yref: yref || "paper", x0: x, x1: x, y0: 0, y1: 1, line: { color: color, width: width || 1, dash: dash || "solid" }, layer: "below" };
  }

  /* ======================= Message 1 ======================= */

  function m1a(el, D) {
    var blocks = D.main_results.blocks;
    var sis = D.sa_in_structure.rows;
    V.figure(el, {
      id: "fig-m1a", num: "Figure 1a",
      title: "Accuracy at the selected peak: base workflow, IP and SP against SA-RL",
      controls: [
        { type: "seg", key: "task", label: "Task", options: TASK_OPTS },
        { type: "seg", key: "scale", label: "Scale", options: SCALE_OPTS },
        { type: "check", key: "sa", label: "SA-RL placed in the untrained workflow" }
      ],
      defaults: { task: "Math", scale: "1.7B", sa: true },
      caption: "Grouped bars per workflow: the untrained workflow (gray) and the trained cell under IP (solid) and SP (muted) at its selected peak. Lines: the base model alone and one SA-RL response. Diamonds: the SA-RL adapter placed in every role of the untrained workflow, the comparison that asks whether joint training adds anything. Math: separate evaluation of each checkpoint on the validation set; Code: re-evaluation of the checkpoint selected on the validation series logged during training.",
      render: function (s, t, W) {
        var b = blocks.filter(function (x) { return x.task === s.task && x.scale === s.scale; })[0];
        var rows = WF.map(function (w) { return b.workflows.filter(function (x) { return x.workflow === w; })[0]; });
        var plotW = Math.max(200, W - 80), catPx = plotW / 3;
        var bw = Math.min(0.25, 28 / catPx), gap = 2.5 / catPx;
        function off(i) { return -1.5 * bw - gap + i * (bw + gap); }
        var codeNote = s.task === "Code" ? "Re-evaluated on the full Code validation set" : null;
        var data = [];
        data.push({
          type: "bar", x: WF, y: rows.map(function (r) { return r.base_workflow; }), width: bw, offset: off(0),
          marker: { color: t.neutralBar }, hoverinfo: "text",
          hovertext: rows.map(function (r) { return hoverLines(["<b>" + f1(r.base_workflow) + "%</b>  untrained " + r.workflow + ", " + s.scale + ", " + s.task, "Base model in every role"]); })
        });
        ["IP", "SP"].forEach(function (pol, k) {
          data.push({
            type: "bar", x: WF, y: rows.map(function (r) { return r[pol].value; }), width: bw, offset: off(k + 1),
            marker: { color: rows.map(function (r) { return pol === "IP" ? t.wf[r.workflow] : V.tint(t.wf[r.workflow], 0.42, t.surface); }) },
            hoverinfo: "text",
            hovertext: rows.map(function (r) {
              var c = r[pol];
              return hoverLines([
                "<b>" + f1(c.value) + "%</b>  " + c.cell,
                "Selected peak at step " + c.step,
                codeNote,
                "Gain over the base workflow: " + sgn(c.gain_over_base_workflow_pp) + " points",
                "Minus one SA-RL response: " + sgn(c.residual_vs_sa_rl_pp) + " points",
                "Minus SA-RL in the same structure: " + sgn(c.residual_vs_sa_in_structure_pp) + " points"
              ]);
            })
          });
        });
        var saRows = WF.map(function (w) { return sis.filter(function (r) { return r.workflow === w && r.scale === s.scale && r.task === s.task; })[0]; });
        if (s.sa) {
          data.push({
            type: "scatter", mode: "markers", x: WF, y: saRows.map(function (r) { return r.sa_rl_in_structure; }),
            marker: { symbol: "diamond", size: 13, color: t.ink, line: { color: t.surface, width: 2 } },
            hoverinfo: "text",
            hovertext: saRows.map(function (r) {
              return hoverLines([
                "<b>" + f1(r.sa_rl_in_structure) + "%</b>  SA-RL adapter in every role of the untrained " + r.workflow,
                "SA-RL checkpoint at its selected peak (step " + r.sa_rl_step + ")",
                "IP minus this: " + sgn(r.IP.diff_pp) + " points",
                "SP minus this: " + sgn(r.SP.diff_pp) + " points"
              ]);
            })
          });
        }
        var ymax = Math.max.apply(null, data.reduce(function (a, d) { return a.concat(d.y); }, []).concat([b.sa_rl.value, b.base_single]));
        var shapes = [hline(b.sa_rl.value, t.ink, 1.5), hline(b.base_single, t.axisInk, 1.2, "dot")];
        var ann = [
          { xref: "paper", yref: "y", x: 0, y: b.sa_rl.value, xanchor: "left", yanchor: "bottom", xshift: 4, showarrow: false, text: "SA-RL, one response " + f1(b.sa_rl.value), font: { size: 11, color: t.ink } },
          { xref: "paper", yref: "y", x: 0, y: b.base_single, xanchor: "left", yanchor: (Math.abs(b.sa_rl.value - b.base_single) < ymax * 0.06 ? "top" : "bottom"), xshift: 4, showarrow: false, text: "base model, single " + f1(b.base_single), font: { size: 11, color: t.ink2 } }
        ];
        var lay = V.layout(t, {
          height: 380, barmode: "overlay", shapes: shapes, annotations: ann,
          margin: { l: 52, r: 14, t: 16, b: 40 },
          xaxis: { type: "category", showgrid: false },
          yaxis: { title: { text: "Accuracy (%)" }, range: [0, ymax * 1.14] }
        });
        var legend = [
          { type: "rect", color: t.neutralBar, label: "Untrained workflow" },
          { type: "split", colors: WF.map(function (w) { return t.wf[w]; }), label: "IP, selected peak" },
          { type: "split", colors: WF.map(function (w) { return V.tint(t.wf[w], 0.42, t.surface); }), label: "SP, selected peak" },
          { type: "tick", horizontal: true, color: t.ink, label: "SA-RL, one response" },
          { type: "line", dash: "dot", color: t.axisInk, label: "Base model, single" }
        ];
        if (s.sa) legend.push({ type: "diamond", color: t.ink, label: "SA-RL in the untrained workflow" });
        var tblRows = rows.map(function (r, i) {
          var sr = saRows[i];
          return [r.workflow, f1(r.base_workflow), f1(r.IP.value) + " (" + r.IP.step + ")", f1(r.SP.value) + " (" + r.SP.step + ")",
            f1(sr.sa_rl_in_structure), sgn(r.IP.residual_vs_sa_rl_pp), sgn(r.SP.residual_vs_sa_rl_pp),
            sgn(r.IP.residual_vs_sa_in_structure_pp), sgn(r.SP.residual_vs_sa_in_structure_pp)];
        });
        tblRows.push({ sep: true, cells: ["Base model, single", f1(b.base_single), "", "", "", "", "", "", ""] });
        tblRows.push(["SA-RL, one response", "", f1(b.sa_rl.value) + " (" + b.sa_rl.step + ")", "", "", "", "", "", ""]);
        return {
          plots: [{ data: data, layout: lay }], legend: legend,
          table: {
            columns: ["Workflow", "Untrained", "IP (step)", "SP (step)", "SA-RL in structure", "IP − SA-RL", "SP − SA-RL", "IP − SA-RL in structure", "SP − SA-RL in structure"],
            numeric: [false, true, true, true, true, true, true, true, true], rows: tblRows
          }
        };
      }
    });
  }

  function m1b(el, D) {
    var rowsAll = D.gain_decomposition.rows;
    V.figure(el, {
      id: "fig-m1b", num: "Figure 1b",
      title: "Where the gain comes from: solving-role gain plus structural gain (Math)",
      controls: [{ type: "seg", key: "scale", label: "Scale", options: SCALE_OPTS_ALL }],
      defaults: { scale: "All" },
      caption: "Each bar splits a Math cell's gain over its untrained workflow, at the sampled checkpoint nearest its selected peak, into the solving role's own gain (gray; graded on its own answer) and the structural gain the workflow adds on top of it (color). Dot: the total workflow gain. Tick: SA-RL's gain over the untrained single model on the same problems. Only the structural part can carry a workflow past SA-RL.",
      render: function (s, t, W) {
        var rows = [];
        WF.forEach(function (w) {
          SC.forEach(function (sc) {
            if (s.scale !== "All" && s.scale !== sc) return;
            ["IP", "SP"].forEach(function (p) {
              var r = rowsAll.filter(function (x) { return x.workflow === w && x.scale === sc && x.policy === p; })[0];
              if (r) rows.push(r);
            });
          });
        });
        var ys = rows.map(function (r) { return r.workflow + " " + r.policy + " " + r.scale; });
        function hv(r) {
          return hoverLines([
            "<b>" + r.cell + "</b> (checkpoint step " + r.dump_step + ")",
            "Workflow gain: <b>" + sgn(r.workflow_gain_pp) + "</b> points (" + f1(r.workflow_acc_base) + " → " + f1(r.workflow_acc_trained) + "%)",
            "Solving-role gain: " + sgn(r.solving_role_gain_pp) + " (" + f1(r.solving_role_acc_base) + " → " + f1(r.solving_role_acc_trained) + "%)",
            "Structural gain: " + sgn(r.structural_gain_pp),
            "SA-RL gain over the untrained single model: " + sgn(r.sa_rl_gain_pp) + " (step " + r.sa_rl_step + ")"
          ]);
        }
        var hov = rows.map(hv);
        var data = [
          { type: "bar", orientation: "h", y: ys, x: rows.map(function (r) { return r.solving_role_gain_pp; }), marker: { color: t.neutralBar }, hoverinfo: "text", hovertext: hov },
          { type: "bar", orientation: "h", y: ys, x: rows.map(function (r) { return r.structural_gain_pp; }), marker: { color: rows.map(function (r) { return t.wf[r.workflow]; }) }, hoverinfo: "text", hovertext: hov },
          { type: "scatter", mode: "markers", y: ys, x: rows.map(function (r) { return r.workflow_gain_pp; }), marker: { symbol: "circle", size: 9, color: t.ink, line: { color: t.surface, width: 2 } }, hoverinfo: "text", hovertext: hov },
          { type: "scatter", mode: "markers", y: ys, x: rows.map(function (r) { return r.sa_rl_gain_pp; }), marker: { symbol: "line-ns-open", size: 18, color: t.ink, line: { color: t.ink, width: 2.5 } }, hoverinfo: "text", hovertext: hov }
        ];
        var lay = V.layout(t, {
          height: 70 + rows.length * (rows.length > 6 ? 27 : 38), barmode: "relative", bargap: 0.38,
          margin: { l: 120, r: 16, t: 8, b: 44 },
          xaxis: { title: { text: "Gain over the untrained workflow (points)" }, zeroline: true, zerolinecolor: t.axisInk, zerolinewidth: 1 },
          yaxis: { autorange: "reversed", type: "category", showgrid: false, ticks: "" }
        });
        return {
          plots: [{ data: data, layout: lay }],
          legend: [
            { type: "rect", color: t.neutralBar, label: "Solving-role gain" },
            { type: "split", colors: WF.map(function (w) { return t.wf[w]; }), label: "Structural gain (by workflow)" },
            { type: "dot", color: t.ink, label: "Workflow gain" },
            { type: "tick", color: t.ink, label: "SA-RL gain" }
          ],
          table: {
            columns: ["Cell", "Checkpoint step", "Workflow gain", "Solving-role gain", "Structural gain", "SA-RL gain (step)"],
            numeric: [false, true, true, true, true, true],
            rows: rows.map(function (r) { return [r.cell, r.dump_step, sgn(r.workflow_gain_pp), sgn(r.solving_role_gain_pp), sgn(r.structural_gain_pp), sgn(r.sa_rl_gain_pp) + " (" + r.sa_rl_step + ")"]; })
          }
        };
      }
    });
  }

  function m1c(el, D) {
    var S = D.solving_role_vs_sa;
    V.figure(el, {
      id: "fig-m1c", num: "Figure 1c",
      title: "Solving role minus SA-RL over training",
      controls: [
        { type: "seg", key: "task", label: "Task", options: TASK_OPTS },
        { type: "seg", key: "wf", label: "Workflow", options: WF_OPTS_ALL },
        { type: "seg", key: "scale", label: "Scale", options: SCALE_OPTS_ALL }
      ],
      defaults: { task: "Math", wf: "All", scale: "1.7B" },
      caption: "Each line follows one cell's solving role (the voters, the workers or the Eval-Opt first attempt), graded on its own answer, minus SA-RL at the nearest evaluated step on the same problems; solid: IP, dashed: SP. Open circle at step 0 (Math): the untrained solving role minus the untrained single model. On Code, Orch-Workers-IP-0.6B-Code is not drawn. The solving roles end at or below SA-RL on both tasks, by a smaller margin on Code.",
      render: function (s, t) {
        var series = (s.task === "Math" ? S.math_series : S.code_series).filter(function (c) {
          return (s.wf === "All" || c.workflow === s.wf) && (s.scale === "All" || c.scale === s.scale) && !c.excluded_fence_loss;
        });
        var data = [], tbl = [];
        series.forEach(function (c) {
          var pts = c.points.slice();
          var xs = pts.map(function (p) { return p.step; });
          var ys = pts.map(function (p) { return p.diff_pp; });
          var sym = pts.map(function () { return "circle"; });
          var hv = pts.map(function (p) {
            return hoverLines([
              "<b>" + sgn(p.diff_pp) + " points</b>  " + c.cell + ", step " + p.step,
              "Solving role " + f1(p.solving_role_acc) + "% vs SA-RL " + f1(p.sa_rl_acc) + "% (step " + p.sa_rl_step + ")"
            ]);
          });
          if (s.task === "Math") {
            var u = S.math_untrained_reference.filter(function (r) { return r.workflow === c.workflow && r.scale === c.scale; })[0];
            if (u) {
              xs.unshift(0); ys.unshift(u.diff_pp); sym.unshift("circle-open");
              hv.unshift(hoverLines(["<b>" + sgn(u.diff_pp) + " points</b>  untrained " + c.workflow + " " + c.scale, "Untrained solving role " + f1(u.untrained_solving_role_acc) + "% vs untrained single model " + f1(u.untrained_single_acc) + "%"]));
            }
          }
          data.push({
            type: "scatter", mode: "lines+markers", x: xs, y: ys, name: c.cell,
            line: { color: t.wf[c.workflow], width: 2, dash: c.policy === "SP" ? "dash" : "solid" },
            marker: { size: 7, symbol: sym, color: t.wf[c.workflow], line: { color: t.surface, width: 1.5 } },
            hoverinfo: "text", hovertext: hv
          });
          pts.forEach(function (p) { tbl.push([c.cell, p.step, f1(p.solving_role_acc), f1(p.sa_rl_acc) + " (" + p.sa_rl_step + ")", sgn(p.diff_pp)]); });
        });
        var lay = V.layout(t, {
          height: 380, shapes: [hline(0, t.ink2, 1.2)],
          margin: { l: 56, r: 14, t: 10, b: 44 },
          xaxis: { title: { text: "Training step" } },
          yaxis: { title: { text: "Solving role minus SA-RL (points)" } }
        });
        var legend = wfLegend(t, "line").filter(function (it) { return s.wf === "All" || it.label === s.wf; });
        legend.push({ type: "line", color: t.ink2, label: "IP" });
        legend.push({ type: "line", dash: "dash", color: t.ink2, label: "SP" });
        if (s.task === "Math") legend.push({ type: "dot", open: true, color: t.ink2, label: "Untrained (step 0)" });
        return {
          plots: [{ data: data, layout: lay }], legend: legend,
          table: { columns: ["Cell", "Step", "Solving role (%)", "SA-RL (%) (step)", "Difference"], numeric: [false, true, true, true, true], rows: tbl }
        };
      }
    });
  }

  function m1d(el, D) {
    var rows = D.decision_role_rules.rows;
    var spec = {
      "Voting": {
        role: "aggregator", rule: "Scores close to a plurality vote over the three answers, and slightly above it at 4B.",
        cols: ["Workflow − plurality vote (points)", "Lone correct voter picked"],
        vals: function (r) { return [sgn(r.workflow_minus_plurality_pp), kn(r.lone_correct_picked)]; }
      },
      "Orch-Workers": {
        role: "synthesizer", rule: "When it writes only an answer, it returns the workers' most common answer.",
        cols: ["Answer-only outputs", "…equal to the workers' plurality", "Selection loss", "Rescue"],
        vals: function (r) { return [V.pct(r.answer_only_share), kn(r.answer_only_equals_plurality), V.pct(r.loss_share), V.pct(r.rescue_share)]; }
      },
      "Eval-Opt": {
        role: "evaluator", rule: "Sends back most truncated first attempts.",
        cols: ["Accepts a truncated wrong first attempt", "Rejects a finished wrong first attempt"],
        vals: function (r) { return [kn(r.accepts_truncated_wrong), kn(r.rejects_finished_wrong)]; }
      }
    };
    V.figure(el, {
      id: "fig-m1d", num: "Figure 1d",
      title: "Decision-role rules after RL (Math)",
      controls: [],
      caption: "Latest sampled IP and SP checkpoint of each Math cell, on all validation problems; k/n: count and denominator. Selection loss: the workflow is wrong although an instance was right; rescue: the workflow is right although no instance was. Each decision role learns the rule its structure needs.",
      render: function (s, t) {
        var box = document.createElement("div");
        box.className = "rules";
        WF.forEach(function (w) {
          var sp = spec[w];
          var card = document.createElement("div");
          card.className = "rule-card";
          var h = document.createElement("h4");
          var dot = document.createElement("i");
          dot.style.background = t.wf[w];
          h.appendChild(dot);
          h.appendChild(document.createTextNode(w + " · " + sp.role));
          card.appendChild(h);
          var p = document.createElement("p");
          p.className = "rule";
          p.textContent = sp.rule;
          card.appendChild(p);
          var tb = document.createElement("div");
          var trs = [];
          SC.forEach(function (sc) {
            ["IP", "SP"].forEach(function (pol) {
              var r = rows.filter(function (x) { return x.workflow === w && x.scale === sc && x.policy === pol; })[0];
              if (r) trs.push([sc + " " + pol, r.step].concat(sp.vals(r)));
            });
          });
          V.renderTable(tb, { columns: ["Cell", "Step"].concat(sp.cols), numeric: [false, true].concat(sp.cols.map(function () { return true; })), rows: trs });
          card.appendChild(tb);
          box.appendChild(card);
        });
        return { plots: [], html: box };
      }
    });
  }

  /* ======================= Message 2 ======================= */

  function m2a(el, D) {
    var A = D.fig5.panel_a, LC = D.fig5.panel_a_sa_long_cap;
    V.figure(el, {
      id: "fig-m2a", num: "Figure 2a",
      title: "First attempt against final answer in Eval-Opt (Math)",
      controls: [{ type: "seg", key: "scale", label: "Scale", options: SCALE_OPTS_ALL }],
      defaults: { scale: "All" },
      caption: "First-attempt (muted) and final (solid) accuracy of each Eval-Opt cell at its latest sampled checkpoint, not its selected peak; gray: the untrained workflow; label: the loop's net contribution in points. Filled diamond: SA-RL's validation accuracy at the nearest step; open diamond: SA-RL with one longer response. Each generator turn is graded on its own answer. The first attempt alone sits below SA-RL, so the gain comes from the added turns.",
      render: function (s, t, W) {
        var scales = s.scale === "All" ? SC : [s.scale];
        var rows = [];
        scales.forEach(function (sc) {
          ["Untrained", "IP", "SP"].forEach(function (p) {
            var r = A.filter(function (x) { return x.scale === sc && x.policy === p; })[0];
            if (r) rows.push(r);
          });
        });
        var label = function (p) { return p === "Untrained" ? "Untrained" : p; };
        var xs0 = rows.map(function (r) { return r.scale; }), xs1 = rows.map(function (r) { return label(r.policy); });
        var nameOf = function (r) { return r.policy === "Untrained" ? "Untrained Eval-Opt, " + r.scale + ", Math" : r.cell + ", step " + r.step + " (latest sampled checkpoint)"; };
        var colFirst = rows.map(function (r) { return r.policy === "Untrained" ? V.tint(t.neutralBar, 0.45, t.surface) : V.tint(t.wf["Eval-Opt"], 0.42, t.surface); });
        var colFinal = rows.map(function (r) { return r.policy === "Untrained" ? t.neutralBar : t.wf["Eval-Opt"]; });
        var data = [
          {
            type: "bar", x: [xs0, xs1], y: rows.map(function (r) { return r.first; }), marker: { color: colFirst }, hoverinfo: "text",
            hovertext: rows.map(function (r) { return hoverLines(["First attempt <b>" + f1(r.first) + "%</b>", nameOf(r), f1(r.first_trunc) + "% of first attempts truncated"]); })
          },
          {
            type: "bar", x: [xs0, xs1], y: rows.map(function (r) { return r.final; }), marker: { color: colFinal }, hoverinfo: "text",
            text: rows.map(function (r) { return sgn(r.loop_net_pp); }), textposition: "outside", cliponaxis: false, constraintext: "none",
            textfont: { size: 10.5, color: t.ink2 },
            hovertext: rows.map(function (r) { return hoverLines(["Final answer <b>" + f1(r.final) + "%</b>", nameOf(r), "Loop net contribution " + sgn(r.loop_net_pp) + " points", "Recovered " + r.recovered + ", lost " + r.lost + " problems"]); })
          }
        ];
        var tr = rows.filter(function (r) { return r.policy !== "Untrained" && r.sa_acc != null; });
        data.push({
          type: "scatter", mode: "markers", x: [tr.map(function (r) { return r.scale; }), tr.map(function (r) { return label(r.policy); })], y: tr.map(function (r) { return r.sa_acc; }),
          marker: { symbol: "diamond", size: 12, color: t.ink, line: { color: t.surface, width: 2 } }, hoverinfo: "text",
          hovertext: tr.map(function (r) { return hoverLines(["SA-RL <b>" + f1(r.sa_acc) + "%</b>", "SA-RL-" + r.scale + "-Math at step " + r.sa_step + " (validation accuracy, one response)"]); })
        });
        var lc = scales.map(function (sc) { return LC[sc]; });
        data.push({
          type: "scatter", mode: "markers", x: [scales, scales.map(function () { return "SA-RL longer"; })], y: lc.map(function (o) { return o.acc; }),
          marker: { symbol: "diamond-open", size: 12, color: t.ink, line: { color: t.ink, width: 2 } }, hoverinfo: "text",
          hovertext: scales.map(function (sc) { var o = LC[sc]; return hoverLines(["SA-RL, one longer response <b>" + f1(o.acc) + "%</b>", "SA-RL-" + sc + "-Math at step " + o.sa_step + ", one response capped at " + o.cap.toLocaleString("en-US") + " tokens"]); })
        });
        var lay = V.layout(t, {
          height: 390, barmode: "group", bargap: 0.28, bargroupgap: 0.06,
          margin: { l: 52, r: 10, t: 18, b: 56 },
          xaxis: { type: "multicategory", showgrid: false, tickfont: { size: narrow(W) && scales.length > 1 ? 9 : 11 }, dividercolor: t.axis },
          yaxis: { title: { text: "Accuracy (%)" }, range: [0, 88] }
        });
        return {
          plots: [{ data: data, layout: lay }],
          legend: [
            { type: "rect", color: V.tint(t.wf["Eval-Opt"], 0.42, t.surface), label: "First attempt" },
            { type: "rect", color: t.wf["Eval-Opt"], label: "Final answer" },
            { type: "rect", color: t.neutralBar, label: "Untrained (muted: first, solid: final)" },
            { type: "diamond", color: t.ink, label: "SA-RL, nearest step" },
            { type: "diamond", open: true, color: t.ink, label: "SA-RL, one longer response" }
          ],
          table: {
            columns: ["Cell", "Step", "First attempt (%)", "Final (%)", "Loop net", "First attempts truncated (%)", "SA-RL (%) (step)"],
            numeric: [false, true, true, true, true, true, true],
            rows: rows.map(function (r) {
              return [r.policy === "Untrained" ? "Untrained Eval-Opt " + r.scale : r.cell, r.step, f1(r.first), f1(r.final), sgn(r.loop_net_pp), f1(r.first_trunc), r.sa_acc != null ? f1(r.sa_acc) + " (" + r.sa_step + ")" : "—"];
            }).concat(scales.map(function (sc) { var o = LC[sc]; return ["SA-RL-" + sc + "-Math, one response at " + o.cap.toLocaleString("en-US") + " tokens", o.sa_step, "", f1(o.acc), "", "", ""]; }))
          }
        };
      }
    });
  }

  function m2b(el, D) {
    var B = D.fig5.panel_b;
    V.figure(el, {
      id: "fig-m2b", num: "Figure 2b",
      title: "What the loop recovers and what it loses (Math)",
      controls: [{ type: "seg", key: "scale", label: "Scale", options: SCALE_OPTS_ALL }],
      defaults: { scale: "All" },
      caption: "Wrong first attempts of each trained Eval-Opt cell at its latest sampled checkpoint, split into truncated (T) and finished (F); solid: recovered by a later turn, muted: still wrong. Below zero: correct first attempts the loop lost. The loop recovers almost only truncated first attempts and almost never loses a correct one.",
      render: function (s, t, W) {
        var rows = B.filter(function (r) { return s.scale === "All" || r.scale === s.scale; });
        var x = [rows.map(function (r) { return r.scale; }), rows.map(function (r) { return r.policy; })];
        var eo = t.wf["Eval-Opt"];
        function h(r, what) { return hoverLines(["<b>" + r.cell + "</b>, step " + r.step + " (latest sampled checkpoint)", what]); }
        var data = [
          { type: "bar", x: x, y: rows.map(function (r) { return r.rec_trunc_wrong; }), base: rows.map(function () { return 0; }), offsetgroup: "T", marker: { color: eo }, hoverinfo: "text",
            hovertext: rows.map(function (r) { return h(r, "Truncated wrong first attempts recovered: <b>" + r.rec_trunc_wrong + "</b> of " + r.n_trunc_wrong); }) },
          { type: "bar", x: x, y: rows.map(function (r) { return r.n_trunc_wrong - r.rec_trunc_wrong; }), base: rows.map(function (r) { return r.rec_trunc_wrong; }), offsetgroup: "T", marker: { color: V.tint(eo, 0.38, t.surface) }, hoverinfo: "text",
            hovertext: rows.map(function (r) { return h(r, "Truncated wrong first attempts not recovered: <b>" + (r.n_trunc_wrong - r.rec_trunc_wrong) + "</b> of " + r.n_trunc_wrong); }) },
          { type: "bar", x: x, y: rows.map(function (r) { return r.rec_fin_wrong; }), base: rows.map(function () { return 0; }), offsetgroup: "F", marker: { color: t.neutralMark }, hoverinfo: "text",
            hovertext: rows.map(function (r) { return h(r, "Finished wrong first attempts recovered: <b>" + r.rec_fin_wrong + "</b> of " + r.n_fin_wrong); }) },
          { type: "bar", x: x, y: rows.map(function (r) { return r.n_fin_wrong - r.rec_fin_wrong; }), base: rows.map(function (r) { return r.rec_fin_wrong; }), offsetgroup: "F", marker: { color: V.tint(t.neutralBar, 0.5, t.surface) }, hoverinfo: "text",
            hovertext: rows.map(function (r) { return h(r, "Finished wrong first attempts not recovered: <b>" + (r.n_fin_wrong - r.rec_fin_wrong) + "</b> of " + r.n_fin_wrong); }) },
          { type: "bar", x: x, y: rows.map(function (r) { return -r.lost; }), base: rows.map(function () { return 0; }), offsetgroup: "L", marker: { color: t.c[8] }, hoverinfo: "text",
            text: rows.map(function (r) { return r.lost > 0 ? String(r.lost) : ""; }), textposition: "outside", cliponaxis: false, constraintext: "none", textfont: { size: 10.5, color: t.ink2 },
            hovertext: rows.map(function (r) { return h(r, "Correct first attempts lost: <b>" + r.lost + "</b>"); }) }
        ];
        var ymax = Math.max.apply(null, rows.map(function (r) { return Math.max(r.n_trunc_wrong, r.n_fin_wrong); }));
        var lay = V.layout(t, {
          height: 380, barmode: "group", bargap: 0.25, bargroupgap: 0.12,
          margin: { l: 56, r: 10, t: 10, b: 56 },
          xaxis: { type: "multicategory", showgrid: false, dividercolor: t.axis },
          yaxis: { title: { text: "Problems" }, range: [-ymax * 0.08, ymax * 1.06], zeroline: true, zerolinecolor: t.axisInk }
        });
        return {
          plots: [{ data: data, layout: lay }],
          legend: [
            { type: "rect", color: eo, label: "T: truncated wrong, recovered" },
            { type: "rect", color: V.tint(eo, 0.38, t.surface), label: "T: not recovered" },
            { type: "rect", color: t.neutralMark, label: "F: finished wrong, recovered" },
            { type: "rect", color: V.tint(t.neutralBar, 0.5, t.surface), label: "F: not recovered" },
            { type: "rect", color: t.c[8], label: "Correct first attempts lost" }
          ],
          table: {
            columns: ["Cell", "Step", "Truncated wrong", "…recovered", "Finished wrong", "…recovered", "Lost"],
            numeric: [false, true, true, true, true, true, true],
            rows: rows.map(function (r) { return [r.cell, r.step, r.n_trunc_wrong, r.rec_trunc_wrong, r.n_fin_wrong, r.rec_fin_wrong, r.lost]; })
          }
        };
      }
    });
  }

  function m2c(el, D) {
    var C = D.controllers;
    var ROWS = [
      { k: "sa_one_response_cap5120", l: "SA-RL, one response", who: "sa", ladder: 0 },
      { k: "sa_rule_b_retry_if_cut_off_no_evaluator", l: "SA-RL generator, retry if cut off", who: "sa", ladder: 1 },
      { k: "ip_rule_b_retry_if_cut_off_no_evaluator", l: "Eval-Opt-IP generator, retry if cut off", who: "ip", ladder: 2 },
      { k: "ip_trained", l: "Eval-Opt-IP, trained evaluator", who: "ip", ladder: 3 },
      { k: "ip_always_accept", l: "Eval-Opt-IP, always accept", who: "ip", ref: true },
      { k: "sa_one_response_cap10240", l: "SA-RL, one response at 10,240 tokens", who: "sa", ref: true },
      { k: "sa_one_response_cap15360", l: "SA-RL, one response at 15,360 tokens", who: "sa", ref: true },
      { k: "ip_rule_a_retry_if_cut_off_evaluator_text_kept", l: "Eval-Opt-IP, retry if cut off, evaluator text kept", who: "ip", ref: true, extra: true },
      { k: "ip_always_reject", l: "Eval-Opt-IP, always retry", who: "ip", ref: true, extra: true },
      { k: "sp_trained", l: "Eval-Opt-SP, trained evaluator", who: "sp", ref: true, extra: true },
      { k: "sp_rule_b_retry_if_cut_off_no_evaluator", l: "Eval-Opt-SP generator, retry if cut off", who: "sp", ref: true, extra: true },
      { k: "sp_rule_a_retry_if_cut_off_evaluator_text_kept", l: "Eval-Opt-SP, retry if cut off, evaluator text kept", who: "sp", ref: true, extra: true }
    ];
    var STEP = {
      A: { key: "A_retry_structure_sa_generator", what: "A: the retry rule, with the SA-RL generator" },
      B: { key: "B_eval_opt_ip_generator_minus_sa_generator_same_rule", what: "B: the Eval-Opt-IP generator against the SA-RL generator, same rule (same controller, not the same token budget)" },
      C: { key: "C_trained_evaluator_minus_rule_b", what: "C: the trained evaluator against the fixed rule" }
    };
    V.figure(el, {
      id: "fig-m2c", num: "Figure 2c",
      title: "From one SA-RL response to the trained loop: the controller ladder (Math)",
      controls: [
        { type: "seg", key: "scale", label: "Scale", options: SCALE_OPTS },
        { type: "check", key: "all", label: "Show every controller" }
      ],
      defaults: { scale: "1.7B", all: false },
      caption: "Accuracy at the selected checkpoints, up to three generator turns. Read the connected ladder top to bottom: A adds the fixed rule “retry only a cut-off attempt” to the SA-RL generator, B swaps in the Eval-Opt-IP generator under the same rule, and C replaces the rule by the trained evaluator. B compares the two generators under the same retry rule. Hollow marks are references: accepting every first attempt, and one longer SA-RL response. A fixed retry rule reproduces most of the loop at the larger scales; under it, the IP generator clearly beats the SA-RL generator at 1.7B.",
      render: function (s, t, W) {
        var L = C.levels[s.scale];
        var rows = ROWS.filter(function (r) { return !r.extra || s.all; });
        var ypos = [], y = 0;
        rows.forEach(function (r, i) {
          if (i > 0 && r.ref && !rows[i - 1].ref) y += 0.7;
          ypos.push(y); y += 1;
        });
        var nar = narrow(W);
        function lab(r) { return nar ? r.l.replace(", ", ",<br>") : r.l; }
        function col(r) { return r.who === "sa" ? t.ink : r.who === "ip" ? t.wf["Eval-Opt"] : V.tint(t.wf["Eval-Opt"], 0.55, t.surface); }
        function sym(r) {
          var base = r.who === "sa" ? "diamond" : "circle";
          return r.ref ? base + "-open" : base;
        }
        var hv = rows.map(function (r) {
          var o = L[r.k];
          return hoverLines([
            "<b>" + f1(o.acc) + "%</b>  " + r.l + " (" + s.scale + ")",
            "Checkpoint step " + o.step,
            "Generator attempts per episode " + f2(o.mean_generator_attempts) + "; model calls " + f2(o.mean_model_calls),
            "Completion tokens per episode " + f1(o.mean_completion_tokens_k) + "k",
            "First attempts truncated " + Math.round(o.first_attempt_trunc_pct) + "%"
          ]);
        });
        var lad = rows.map(function (r, i) { return { r: r, y: ypos[i] }; }).filter(function (o) { return o.r.ladder != null; });
        var data = [
          { type: "scatter", mode: "lines", x: lad.map(function (o) { return L[o.r.k].acc; }), y: lad.map(function (o) { return o.y; }), line: { color: t.axisInk, width: 1.5 }, hoverinfo: "skip" },
          {
            type: "scatter", mode: "markers", x: rows.map(function (r) { return L[r.k].acc; }), y: ypos,
            marker: { symbol: rows.map(sym), size: 13, color: rows.map(col), line: { color: rows.map(function (r) { return r.ref ? col(r) : t.surface; }), width: 2 } },
            hoverinfo: "text", hovertext: hv
          }
        ];
        var xs = rows.map(function (r) { return L[r.k].acc; });
        var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
        var ann = [];
        ["A", "B", "C"].forEach(function (k, i) {
          var d = C.differences_pp.margin_decomposition[STEP[k].key][s.scale];
          var a0 = L[lad[i].r.k].acc, a1 = L[lad[i + 1].r.k].acc;
          ann.push({
            x: (a0 + a1) / 2, y: (lad[i].y + lad[i + 1].y) / 2, xref: "x", yref: "y", showarrow: false,
            text: "<b>" + k + "</b> " + sgn(d.level), font: { size: 11.5, color: t.ink }, bgcolor: t.surface, borderpad: 2,
            xanchor: a1 >= a0 ? "left" : "right", xshift: a1 >= a0 ? 8 : -8,
            hovertext: STEP[k].what + "<br>" + sgn(d.level) + " points"
          });
        });
        var tot = C.differences_pp.margin_decomposition.total_margin_ip_trained_minus_sa_one_response[s.scale];
        ann.push({
          x: L.ip_trained.acc, y: lad[3].y, xref: "x", yref: "y", showarrow: false, xanchor: "left", yanchor: "top", xshift: 4, yshift: -8,
          text: "total " + sgn(tot.level), font: { size: 11, color: t.ink2 },
          hovertext: "Trained Eval-Opt-IP minus one SA-RL response: " + sgn(tot.level) + " points"
        });
        var lay = V.layout(t, {
          height: 70 + y * (nar ? 40 : 34), annotations: ann,
          margin: { l: 10, r: 18, t: 8, b: 44 },
          xaxis: { title: { text: "Accuracy (%)" }, range: [xmin - 4, xmax + 9] },
          yaxis: { tickvals: ypos, ticktext: rows.map(lab), autorange: "reversed", showgrid: false, ticks: "", tickfont: { size: nar ? 10 : 11.5, color: t.ink2 }, range: [y - 0.4, -0.6] }
        });
        lay.yaxis.autorange = false;
        lay.yaxis.range = [y - 0.4, -0.6];
        var md = C.differences_pp.margin_decomposition;
        return {
          plots: [{ data: data, layout: lay }],
          legend: [
            { type: "diamond", color: t.ink, label: "SA-RL generator" },
            { type: "dot", color: t.wf["Eval-Opt"], label: "Eval-Opt-IP generator" }
          ].concat(s.all ? [{ type: "dot", open: true, color: V.tint(t.wf["Eval-Opt"], 0.55, t.surface), label: "Eval-Opt-SP generator" }] : []).concat([
            { type: "line", color: t.axisInk, label: "Ladder A → B → C" },
            { type: "dot", open: true, color: t.ink2, label: "Hollow: reference" }
          ]),
          table: [
            {
              title: "Controllers at " + s.scale,
              columns: ["System", "Step", "Accuracy (%)", "Generator attempts", "Model calls", "Completion tokens (k)", "First attempts truncated (%)"],
              numeric: [false, true, true, true, true, true, true],
              rows: ROWS.map(function (r) { var o = L[r.k]; return [r.l, o.step, f1(o.acc), f2(o.mean_generator_attempts), f2(o.mean_model_calls), f1(o.mean_completion_tokens_k), Math.round(o.first_attempt_trunc_pct)]; })
            },
            {
              title: "Margin over one SA-RL response, decomposed (points)",
              columns: ["Term", "0.6B", "1.7B", "4B"],
              numeric: [false, true, true, true],
              rows: [["A: retry rule, SA-RL generator", "A_retry_structure_sa_generator"], ["B: Eval-Opt-IP generator vs SA-RL generator, same rule", "B_eval_opt_ip_generator_minus_sa_generator_same_rule"], ["C: trained evaluator vs the rule", "C_trained_evaluator_minus_rule_b"], ["Total: trained Eval-Opt-IP minus one SA-RL response", "total_margin_ip_trained_minus_sa_one_response"]].map(function (p) {
                return [p[0]].concat(SC.map(function (sc) { return sgn(md[p[1]][sc].level); }));
              })
            }
          ]
        };
      }
    });
  }

  function m2d(el, D) {
    var P = D.fig5.panel_c;
    V.figure(el, {
      id: "fig-m2d", num: "Figure 2d",
      title: "The IP generator learns when to finish its first attempt (Math)",
      controls: [{ type: "seg", key: "scale", label: "Scale", options: SCALE_OPTS }],
      defaults: { scale: "1.7B" },
      caption: "Share of first attempts that finish under the length cap (dashed) and accuracy of the finished ones (solid). SA-RL, trained alone, finishes more of its responses as training goes on; the Eval-Opt-IP generator finishes fewer first attempts, and the ones it finishes are mostly right.",
      render: function (s, t) {
        var p = P[s.scale], eo = t.wf["Eval-Opt"];
        function tr(pts, key, color, dash, name, who) {
          return {
            type: "scatter", mode: "lines+markers", x: pts.map(function (q) { return q.step; }), y: pts.map(function (q) { return q[key]; }),
            line: { color: color, width: 2, dash: dash }, marker: { size: 7, color: color, line: { color: t.surface, width: 1.5 } },
            hoverinfo: "text", hovertext: pts.map(function (q) { return hoverLines(["<b>" + f1(q[key]) + "%</b>  " + name, who + ", step " + q.step + (q.step === 0 ? " (untrained workflow)" : "")]); })
          };
        }
        var data = [
          tr(p.ip, "finished_share", eo, "dash", "first attempts finished", p.ip_cell),
          tr(p.ip, "finished_acc", eo, "solid", "accuracy of finished first attempts", p.ip_cell),
          tr(p.sa, "finished_share", t.ink, "dash", "responses finished", p.sa_cell),
          tr(p.sa, "finished_acc", t.ink, "solid", "accuracy of finished responses", p.sa_cell)
        ];
        var lay = V.layout(t, { height: 360, xaxis: { title: { text: "Training step" } }, yaxis: { title: { text: "Percent" }, range: [0, 102] } });
        var rows = p.ip.map(function (q) { return [p.ip_cell, q.step, f1(q.finished_share), f1(q.finished_acc)]; })
          .concat(p.sa.map(function (q) { return [p.sa_cell, q.step, f1(q.finished_share), f1(q.finished_acc)]; }));
        return {
          plots: [{ data: data, layout: lay }],
          legend: [
            { type: "line", color: eo, label: "Eval-Opt-IP: accuracy of finished first attempts" },
            { type: "line", dash: "dash", color: eo, label: "Eval-Opt-IP: share finished" },
            { type: "line", color: t.ink, label: "SA-RL: accuracy of finished responses" },
            { type: "line", dash: "dash", color: t.ink, label: "SA-RL: share finished" }
          ],
          table: { columns: ["Run", "Step", "Finished (%)", "Accuracy of finished (%)"], numeric: [false, true, true, true], rows: rows }
        };
      }
    });
  }

  /* ======================= Message 3 ======================= */

  var SIZE = { "0.6B": 12, "1.7B": 15.5, "4B": 20 };
  function pairScatter(el, D, opts) {
    V.figure(el, {
      id: opts.id, num: opts.num, title: opts.title,
      controls: [{ type: "seg", key: "task", label: "Task", options: [{ v: "All", l: "Both" }].concat(TASK_OPTS) }].concat(opts.controls || []),
      defaults: Object.assign({ task: "All" }, opts.defaults || {}),
      caption: opts.caption,
      render: function (s, t, W) {
        var pairs = D.pairs.filter(function (p) { return (s.task === "All" || p.task === s.task) && (!opts.keep || opts.keep(p)); });
        var data = [], all = [];
        WF.forEach(function (w) {
          ["Math", "Code"].forEach(function (task) {
            var ps = pairs.filter(function (p) { return p.workflow === w && p.task === task; });
            if (!ps.length) return;
            var xs = ps.map(function (p) { return opts.x(p, s); }), ys = ps.map(function (p) { return opts.y(p, s); });
            all = all.concat(xs, ys);
            data.push({
              type: "scatter", mode: "markers", x: xs, y: ys,
              marker: { opacity: 1,
                symbol: V.WF_SYMBOL[w] + (task === "Code" ? "-open" : ""), size: ps.map(function (p) { return SIZE[p.scale]; }),
                color: t.wf[w], line: { color: task === "Code" ? t.wf[w] : t.surface, width: task === "Code" ? 2 : 1 }
              },
              hoverinfo: "text", hovertext: ps.map(function (p) { return opts.hover(p, s); })
            });
          });
        });
        var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
        var pad = (hi - lo) * 0.08 + 1;
        lo = Math.max(0, lo - pad); hi = hi + pad;
        data.unshift({ type: "scatter", mode: "lines", x: [lo, hi], y: [lo, hi], line: { color: t.axisInk, width: 1.2 }, hoverinfo: "skip" });
        // square plot area set by an explicit x domain (scaleanchor can stretch the range after a control switch)
        var side = Math.min(W, 500), mg = { l: 56, r: 12, t: 10, b: 48 };
        var frac = Math.min(1, (side - mg.t - mg.b) / Math.max(1, W - mg.l - mg.r));
        var lay = V.layout(t, {
          height: side, margin: mg,
          xaxis: { title: { text: opts.xTitle(s) }, range: [lo, hi], domain: [(1 - frac) / 2, (1 + frac) / 2] },
          yaxis: { title: { text: opts.yTitle(s) }, range: [lo, hi] },
          annotations: [
            { xref: "x domain", yref: "y domain", x: 0.03, y: 0.97, xanchor: "left", yanchor: "top", showarrow: false, text: opts.upper, font: { size: 11, color: t.ink2 } },
            { xref: "x domain", yref: "y domain", x: 0.97, y: 0.03, xanchor: "right", yanchor: "bottom", showarrow: false, text: opts.lower, font: { size: 11, color: t.ink2 } }
          ]
        });
        var legend = WF.map(function (w) { return { type: w === "Voting" ? "dot" : w === "Orch-Workers" ? "square" : "triangle", color: t.wf[w], label: w }; });
        legend.push({ type: "dot", color: t.ink2, label: "filled: Math" });
        legend.push({ type: "dot", open: true, color: t.ink2, label: "open: Code" });
        legend.push({ type: "dot", color: t.ink2, label: "size: 0.6B < 1.7B < 4B" });
        return { plots: [{ data: data, layout: lay }], legend: legend, table: opts.table(pairs, s) };
      }
    });
  }

  function m3a(el, D) {
    pairScatter(el, D, {
      id: "fig-m3a", num: "Figure 3a", title: "Selected peak, IP against SP, for every matched pair",
      caption: "Each point is one matched pair (workflow, scale, task). Above the diagonal, IP peaks higher. Math: separate evaluation of each checkpoint; Code: re-evaluation of the checkpoint selected on the validation series logged during training. IP peaks higher in most pairs.",
      x: function (p) { return p.sp_peak; }, y: function (p) { return p.ip_peak; },
      xTitle: function () { return "SP selected peak (%)"; }, yTitle: function () { return "IP selected peak (%)"; },
      upper: "IP higher", lower: "SP higher",
      hover: function (p) {
        return hoverLines(["<b>" + p.pair + "</b>", "IP " + f1(p.ip_peak) + "% at step " + p.ip_peak_step, "SP " + f1(p.sp_peak) + "% at step " + p.sp_peak_step, "IP − SP " + sgn(p.ip_minus_sp_pp) + " points"]);
      },
      table: function (pairs) {
        return { columns: ["Pair", "IP peak (step)", "SP peak (step)", "IP − SP"], numeric: [false, true, true, true],
          rows: pairs.map(function (p) { return [p.pair, f1(p.ip_peak) + " (" + p.ip_peak_step + ")", f1(p.sp_peak) + " (" + p.sp_peak_step + ")", sgn(p.ip_minus_sp_pp)]; }) };
      }
    });
  }

  function m3b(el, D) {
    pairScatter(el, D, {
      id: "fig-m3b", num: "Figure 3b", title: "Training steps to reach the pair's common gain",
      keep: function (p) { return p.common_gain >= 2; },
      caption: "Steps until each run first reaches 90% of the gain both runs of the pair attain within their common horizon, on lightly smoothed validation series. Below the diagonal, IP is faster. Pairs whose common gain is under 2 points are not drawn. IP learns faster in nearly every pair.",
      x: function (p) { return p.t90_sp; }, y: function (p) { return p.t90_ip; },
      xTitle: function () { return "SP: steps to 90% of the common gain"; }, yTitle: function () { return "IP: steps to 90% of the common gain"; },
      upper: "SP faster", lower: "IP faster",
      hover: function (p) {
        return hoverLines(["<b>" + p.pair + "</b>", "IP reaches 90% of the common gain at step " + Math.round(p.t90_ip), "SP at step " + Math.round(p.t90_sp), "Common gain " + f1(p.common_gain) + " points; common horizon " + p.H + " steps"]);
      },
      table: function (pairs) {
        return { columns: ["Pair", "Common gain (points)", "IP: steps to 90%", "SP: steps to 90%"], numeric: [false, true, true, true],
          rows: pairs.map(function (p) { return [p.pair, f1(p.common_gain), Math.round(p.t90_ip), Math.round(p.t90_sp)]; }) };
      }
    });
  }

  function m3c(el, D) {
    V.figure(el, {
      id: "fig-m3c", num: "Figure 3c", title: "Validation accuracy of every matched pair over training",
      controls: [
        { type: "seg", key: "task", label: "Task", options: TASK_OPTS },
        { type: "check", key: "t1", label: "Reweighted SP (Voting 1.7B Math)" }
      ],
      defaults: { task: "Math", t1: true },
      caption: "One panel per workflow and scale. Solid: IP; dashed: SP; dots: selected peaks (filled IP, open SP; step ≤ 300, right of the thin vertical line is past that window); gray line: the untrained workflow. Dash-dot (Voting 1.7B Math): SP with each output's loss reweighted by its role's mean length. Math: separate evaluation of each checkpoint; Code: the validation series logged during training. Where IP declines, its lead shrinks or reverses.",
      render: function (s, t, W) {
        var cols = W < 640 ? 1 : 3, nrows = 9 / cols;
        var curves = D.curves.filter(function (c) { return c.task === s.task; });
        var data = [], shapes = [], ann = [], lay = {};
        var gapX = cols > 1 ? 0.06 : 0, gapY = cols > 1 ? 0.085 : 0.035;
        var cellH = cols > 1 ? 190 : 165;
        var tbl = [];
        WF.forEach(function (w, ri) {
          SC.forEach(function (sc, ci) {
            var c = curves.filter(function (x) { return x.workflow === w && x.scale === sc; })[0];
            if (!c) return;
            var idx = ri * 3 + ci, k = idx + 1;
            var gr = cols > 1 ? ri : idx, gc = cols > 1 ? ci : 0;
            var x0 = gc / cols + (gc > 0 ? gapX / 2 : 0), x1 = (gc + 1) / cols - (gc < cols - 1 ? gapX / 2 : 0);
            var yTop = 1 - gr / nrows - (gr > 0 ? gapY / 2 : 0.02), yBot = 1 - (gr + 1) / nrows + (gr < nrows - 1 ? gapY / 2 : 0);
            var xa = "x" + (k === 1 ? "" : k), ya = "y" + (k === 1 ? "" : k);
            lay["xaxis" + (k === 1 ? "" : k)] = { domain: [x0, x1], anchor: ya, tickfont: { size: 10, color: t.ink2 }, title: { text: "" } };
            lay["yaxis" + (k === 1 ? "" : k)] = { domain: [yBot, yTop], anchor: xa, tickfont: { size: 10, color: t.ink2 }, title: { text: "" } };
            ann.push({ xref: "paper", yref: "paper", x: x0, y: yTop + 0.004, xanchor: "left", yanchor: "bottom", showarrow: false, text: "<b>" + w + "</b> · " + sc, font: { size: 11.5, color: t.ink } });
            var col = t.wf[w];
            ["IP", "SP"].forEach(function (pol) {
              var r = c[pol];
              data.push({
                type: "scatter", mode: "lines", x: r.steps, y: r.acc, xaxis: xa, yaxis: ya,
                line: { color: col, width: 1.8, dash: pol === "SP" ? "dash" : "solid" }, hoverinfo: "text",
                hovertext: r.steps.map(function (st, i) { return "<b>" + f1(r.acc[i]) + "%</b>  " + r.cell + ", step " + st; })
              });
              data.push({
                type: "scatter", mode: "markers", x: [r.selected_peak_step], y: [r.selected_peak], xaxis: xa, yaxis: ya,
                marker: { size: 8, symbol: pol === "SP" ? "circle-open" : "circle", color: col, line: { color: pol === "SP" ? col : t.surface, width: pol === "SP" ? 2 : 1.5 } },
                hoverinfo: "text", hovertext: [r.cell + ": selected peak, step " + r.selected_peak_step]
              });
              tbl.push([r.cell, r.selected_peak_step, f1(r.last) + " (" + r.last_step + ")", r.steps[0] + "–" + r.steps[r.steps.length - 1]]);
            });
            if (s.t1 && s.task === "Math" && w === "Voting" && sc === "1.7B") {
              var T1 = D.t1_curve;
              data.push({
                type: "scatter", mode: "lines", x: T1.steps, y: T1.acc, xaxis: xa, yaxis: ya,
                line: { color: t.c[7], width: 1.8, dash: "dashdot" }, hoverinfo: "text",
                hovertext: T1.steps.map(function (st, i) { return "<b>" + f1(T1.acc[i]) + "%</b>  Voting-SP-1.7B-Math, loss reweighted by role length, step " + st; })
              });
            }
            if (c.base_workflow != null) shapes.push({ type: "line", xref: xa + " domain", yref: ya, x0: 0, x1: 1, y0: c.base_workflow, y1: c.base_workflow, line: { color: t.axisInk, width: 1 }, layer: "below" });
            var maxStep = Math.max(c.IP.steps[c.IP.steps.length - 1], c.SP.steps[c.SP.steps.length - 1]);
            if (maxStep > 300) shapes.push({ type: "line", xref: xa, yref: ya + " domain", x0: 300, x1: 300, y0: 0, y1: 1, line: { color: t.axis, width: 1 }, layer: "below" });
          });
        });
        var L = V.layout(t, Object.assign({
          height: nrows * cellH + 40, shapes: shapes, annotations: ann,
          margin: { l: 40, r: 10, t: 24, b: 30 }
        }, lay));
        var legend = [
          { type: "line", color: t.ink2, label: "IP" }, { type: "line", dash: "dash", color: t.ink2, label: "SP" },
          { type: "dot", color: t.ink2, label: "IP selected peak" }, { type: "dot", open: true, color: t.ink2, label: "SP selected peak" },
          { type: "line", color: t.axisInk, label: "Untrained workflow" }
        ];
        if (s.task === "Math" && s.t1) legend.push({ type: "line", dash: "dashdot", color: t.c[7], label: "SP, loss reweighted by role length" });
        return {
          plots: [{ data: data, layout: L, title: "Validation accuracy (%) against training step" }], legend: legend,
          table: { columns: ["Run", "Selected peak step", "Last (step)", "Validated steps"], numeric: [false, true, true, true], rows: tbl }
        };
      }
    });
  }

  function m3d(el, D) {
    var G = D.gradient_shares;
    V.figure(el, {
      id: "fig-m3d", num: "Figure 3d", title: "The decision role's share of the shared gradient under SP (Math)",
      controls: [],
      caption: "First panel: the decision role's share of the summed per-role gradient norm at probed checkpoints of four SP cells, with each output's loss averaged over its own tokens as in training (filled), over all tokens of the batch (open), and the uniform share, the role's fraction of calls (tick); bars span the two probe batches. Second panel: in four SP cells, the decision role's share of response tokens against its share of the gradient, averaged over probed checkpoints; bars span the checkpoints. The two panels come from different probe sets and are not one measurement. Under the per-output token mean, the decision role, with much shorter outputs, takes well over its uniform share.",
      render: function (s, t, W) {
        var P = G.fig3c.points;
        var yl = P.map(function (p) { return p.cell.replace("-Math", "") + " · step " + p.step; });
        var yi = P.map(function (p, i) { return i; });
        var wfOf = function (c) { return c.split("-SP-")[0]; };
        var hv = P.map(function (p) {
          return hoverLines(["<b>" + p.cell + "</b>, step " + p.step + " (" + p.decision_role + ")",
            "As trained: <b>" + f2(p.as_trained) + "</b> [" + f2(p.as_trained_min) + ", " + f2(p.as_trained_max) + "]",
            "All-token mean: " + f2(p.token_mean), "Uniform share: " + f2(p.uniform),
            "Share of response tokens: " + f2(p.token_fraction),
            "Mean output length: decision " + Math.round(p.mean_len_decision) + ", solving " + Math.round(p.mean_len_solving) + " tokens"]);
        });
        var d1 = [
          { type: "scatter", mode: "markers", x: P.map(function (p) { return p.uniform; }), y: yi, marker: { symbol: "line-ns-open", size: 16, color: t.ink, line: { color: t.ink, width: 2.5 } }, hoverinfo: "text", hovertext: hv },
          { type: "scatter", mode: "markers", x: P.map(function (p) { return p.token_mean; }), y: yi, marker: { symbol: "circle-open", size: 10, color: P.map(function (p) { return t.wf[wfOf(p.cell)]; }), line: { width: 2, color: P.map(function (p) { return t.wf[wfOf(p.cell)]; }) } }, hoverinfo: "text", hovertext: hv },
          {
            type: "scatter", mode: "markers", x: P.map(function (p) { return p.as_trained; }), y: yi,
            error_x: { type: "data", symmetric: false, array: P.map(function (p) { return p.as_trained_max - p.as_trained; }), arrayminus: P.map(function (p) { return p.as_trained - p.as_trained_min; }), color: t.ink2, thickness: 1.2, width: 3 },
            marker: { symbol: "circle", size: 10, color: P.map(function (p) { return t.wf[wfOf(p.cell)]; }), line: { color: t.surface, width: 1.5 } }, hoverinfo: "text", hovertext: hv
          }
        ];
        var nar = narrow(W);
        var l1 = V.layout(t, {
          height: 60 + P.length * 27, margin: { l: 10, r: 14, t: 6, b: 44 },
          xaxis: { title: { text: "Decision role's share of the gradient" }, range: [0, 1.02] },
          yaxis: { tickvals: yi, ticktext: yl, autorange: "reversed", showgrid: false, ticks: "", tickfont: { size: nar ? 9.5 : 11, color: t.ink2 } }
        });
        var R = G.fig1d.rows;
        var y2 = R.map(function (r, i) { return i; });
        var d2 = [];
        R.forEach(function (r, i) {
          d2.push({ type: "scatter", mode: "lines", x: [r.token_share, r.grad_share], y: [i, i], line: { color: t.axisInk, width: 1.5 }, hoverinfo: "skip" });
        });
        var hv2 = R.map(function (r) {
          return hoverLines(["<b>" + r.cell + "</b> (" + r.decision_role + ")", "Share of response tokens: " + f2(r.token_share),
            "Share of the gradient: <b>" + f2(r.grad_share) + "</b> [" + f2(r.grad_share_min) + ", " + f2(r.grad_share_max) + "]",
            "Solving outputs " + f1(r.len_ratio) + "× longer than the decision role's", "Probed checkpoints: " + r.probe_steps.join(", ")]);
        });
        d2.push({ type: "scatter", mode: "markers", x: R.map(function (r) { return r.token_share; }), y: y2, marker: { symbol: "circle-open", size: 10, color: t.ink2, line: { color: t.ink2, width: 2 } }, hoverinfo: "text", hovertext: hv2 });
        d2.push({
          type: "scatter", mode: "markers", x: R.map(function (r) { return r.grad_share; }), y: y2,
          error_x: { type: "data", symmetric: false, array: R.map(function (r) { return r.grad_share_max - r.grad_share; }), arrayminus: R.map(function (r) { return r.grad_share - r.grad_share_min; }), color: t.ink2, thickness: 1.2, width: 3 },
          marker: { symbol: "circle", size: 11, color: R.map(function (r) { return t.wf[wfOf(r.cell)]; }), line: { color: t.surface, width: 1.5 } }, hoverinfo: "text", hovertext: hv2
        });
        var l2 = V.layout(t, {
          height: Math.max(220, 60 + R.length * 40), margin: { l: 10, r: 14, t: 6, b: 44 },
          xaxis: { title: { text: "Decision role's share" }, range: [0, 1.02] },
          yaxis: { tickvals: y2, ticktext: R.map(function (r) { return r.cell.replace("-Math", ""); }), autorange: "reversed", showgrid: false, ticks: "", tickfont: { size: nar ? 9.5 : 11, color: t.ink2 } }
        });
        return {
          twoCol: true,
          plots: [
            { data: d1, layout: l1, title: "Probed checkpoints: as trained, all-token mean, uniform share" },
            { data: d2, layout: l2, title: "Four SP cells: share of tokens against share of the gradient (separate probe set)" }
          ],
          legend: [
            { type: "dot", color: t.wf["Voting"], label: "As trained (Voting)" },
            { type: "dot", color: t.wf["Orch-Workers"], label: "As trained (Orch-Workers)" },
            { type: "dot", color: t.wf["Eval-Opt"], label: "As trained (Eval-Opt)" },
            { type: "dot", open: true, color: t.ink2, label: "All-token mean (first panel) / share of tokens (second)" },
            { type: "tick", color: t.ink, label: "Uniform share" },
            { type: "errbar", color: t.ink2, label: "Range" }
          ],
          table: [
            { title: "Probed checkpoints", columns: ["Cell", "Step", "Decision role", "As trained", "Range", "All-token mean", "Uniform"], numeric: [false, true, false, true, true, true, true],
              rows: P.map(function (p) { return [p.cell, p.step, p.decision_role, f2(p.as_trained), f2(p.as_trained_min) + "–" + f2(p.as_trained_max), f2(p.token_mean), f2(p.uniform)]; }) },
            { title: "Share of tokens against share of the gradient (separate probe set)", columns: ["Cell", "Decision role", "Share of tokens", "Share of gradient", "Range"], numeric: [false, false, true, true, true],
              rows: R.map(function (r) { return [r.cell, r.decision_role, f2(r.token_share), f2(r.grad_share), f2(r.grad_share_min) + "–" + f2(r.grad_share_max)]; }) }
          ]
        };
      }
    });
  }

  function m3e(el, D) {
    var pair = D.curves.filter(function (c) { return c.pair === "Voting-1.7B-Math"; })[0];
    var T1 = D.t1_curve, TC = D.t1_control;
    V.figure(el, {
      id: "fig-m3e", num: "Figure 3e", title: "Reweighting the shared loss by role length (Voting, 1.7B, Math)",
      controls: [],
      caption: "Validation accuracy of Voting-IP-1.7B-Math, Voting-SP-1.7B-Math and the same SP run with each output's loss scaled by its role's mean length, which moves the shared gradient toward the voters; dots: selected peaks. Reweighted, the shared policy learns almost as fast as IP, peaks within noise of it, and declines after its peak like IP.",
      render: function (s, t) {
        var blue = t.wf["Voting"];
        function line(r, name, color, dash) {
          return {
            type: "scatter", mode: "lines", x: r.steps, y: r.acc, name: name,
            line: { color: color, width: 2, dash: dash }, hovertemplate: name + ": <b>%{y:.1f}%</b><extra></extra>"
          };
        }
        function peak(r, color, open, name) {
          return {
            type: "scatter", mode: "markers", x: [r.selected_peak_step], y: [r.selected_peak], name: name + " selected peak",
            marker: { size: 10, symbol: open ? "circle-open" : "circle", color: color, line: { color: open ? color : t.surface, width: 2 } },
            hovertemplate: name + " selected peak: <b>%{y:.1f}%</b> at step %{x}<extra></extra>"
          };
        }
        var data = [
          line(pair.IP, "Voting-IP-1.7B-Math", blue, "solid"),
          line(pair.SP, "Voting-SP-1.7B-Math", blue, "dash"),
          line(T1, "SP, reweighted by role length", t.c[7], "solid"),
          peak(pair.IP, blue, false, "IP"), peak(pair.SP, blue, true, "SP"), peak(T1, t.c[7], false, "Reweighted SP")
        ];
        var lay = V.layout(t, {
          height: 360, hovermode: "x unified", xaxis: { title: { text: "Training step" }, showspikes: true, spikecolor: t.axisInk, spikethickness: 1, spikedash: "solid", spikemode: "across" },
          yaxis: { title: { text: "Validation accuracy (%)" } },
          shapes: [hline(pair.base_workflow, t.axisInk, 1)],
          annotations: [{ xref: "paper", yref: "y", x: 1, y: pair.base_workflow, xanchor: "right", yanchor: "top", showarrow: false, text: "untrained workflow", font: { size: 10.5, color: t.ink2 } }]
        });
        var sp = TC.speed, vl = TC.voter_lead_over_sp_reference_pp, dc = TC.decline;
        return {
          plots: [{ data: data, layout: lay }],
          badges: [{ strong: "Reweighted SP: ", text: "peak at step " + dc.peak_step + ", last validated step " + dc.last_step + ", " + sgn(-dc.drop_pp) + " points" }],
          legend: [
            { type: "line", color: blue, label: "Voting-IP-1.7B-Math" },
            { type: "line", dash: "dash", color: blue, label: "Voting-SP-1.7B-Math" },
            { type: "line", color: t.c[7], label: "SP, loss reweighted by role length" }
          ],
          table: [
            { title: "Speed and peaks", columns: ["Run", "Steps to 90% of common gain", "Selected peak (step)", "Last (step)"], numeric: [false, true, true, true],
              rows: [
                ["Voting-IP-1.7B-Math", Math.round(sp.t90_ip), f1(pair.IP.selected_peak) + " (" + pair.IP.selected_peak_step + ")", f1(pair.IP.last) + " (" + pair.IP.last_step + ")"],
                ["Voting-SP-1.7B-Math", Math.round(sp.t90_sp), f1(pair.SP.selected_peak) + " (" + pair.SP.selected_peak_step + ")", f1(pair.SP.last) + " (" + pair.SP.last_step + ")"],
                ["SP, reweighted by role length", Math.round(sp.t90_t1), f1(T1.selected_peak) + " (" + T1.selected_peak_step + ")", f1(T1.last) + " (" + T1.last_step + ")"]
              ] },
            { title: "Voter accuracy lead over the SP reference (points)", columns: ["Run", "Early", "At the peaks"], numeric: [false, true, true],
              rows: [
                ["SP, reweighted by role length", sgn(vl.early), sgn(vl.at_peaks)]
              ] }
          ]
        };
      }
    });
  }

  /* ======================= Message 4 ======================= */

  function declineLabel(l) { return l ? l.charAt(0).toUpperCase() + l.slice(1) : ""; }

  function m4a(el, D) {
    var cells = D.cells;
    V.figure(el, {
      id: "fig-m4a", num: "Figure 4a", title: "Decline explorer: isolated-policy Math runs",
      controls: [{ type: "select", key: "cell", label: "Cell", options: cells.map(function (c) { return { v: c.cell, l: c.cell }; }) }],
      defaults: { cell: "Voting-IP-1.7B-Math" },
      caption: "Top: validation accuracy of the IP run (solid) and of the matched SP run (dashed), and IP training success (gray; a training-side signal, not accuracy). Bottom: the parallel role's logged entropy proxy over its early median, log scale (Eval-Opt: the generator's). Dotted line: selected peak (highest validation accuracy at step ≤ 300); shaded: second pass over the training data. In the declining Voting and Orch-Workers runs, validation accuracy and training success fall together while the parallel role's entropy proxy rises.",
      render: function (s, t, W) {
        var c = cells.filter(function (x) { return x.cell === s.cell; })[0];
        var col = t.wf[c.workflow], S = c.series;
        function xy(a) { return { x: a.map(function (p) { return p[0]; }), y: a.map(function (p) { return p[1]; }) }; }
        var ipv = xy(S.ip_validation), spv = xy(S.sp_validation), ts = xy(S.ip_train_success), ep = xy(S.entropy_proxy);
        var roleName = c.parallel_role ? (c.workflow === "Voting" ? "voters'" : "workers'") : "generator's";
        var data = [
          { type: "scatter", mode: "lines", x: ts.x, y: ts.y, xaxis: "x", yaxis: "y", line: { color: t.neutralBar, width: 1.5 }, hovertemplate: "IP training success: %{y:.1f}%<extra></extra>" },
          { type: "scatter", mode: "lines", x: spv.x, y: spv.y, xaxis: "x", yaxis: "y", line: { color: col, width: 2, dash: "dash" }, hovertemplate: c.matched_sp_cell + ": <b>%{y:.1f}%</b><extra></extra>" },
          { type: "scatter", mode: "lines+markers", x: ipv.x, y: ipv.y, xaxis: "x", yaxis: "y", line: { color: col, width: 2.2 }, marker: { size: 5, color: col }, hovertemplate: c.cell + ": <b>%{y:.1f}%</b><extra></extra>" },
          { type: "scatter", mode: "lines", x: ep.x, y: ep.y, xaxis: "x", yaxis: "y2", line: { color: t.ink, width: 1.6 }, hovertemplate: "Entropy proxy (" + roleName.replace("'", "").replace("’", "") + "): %{y:.2f}× early median<extra></extra>" }
        ];
        var horizon = c.horizon;
        var shapes = [
          { type: "line", xref: "x", yref: "paper", x0: c.selected_peak.step, x1: c.selected_peak.step, y0: 0, y1: 1, line: { color: t.ink2, width: 1.3, dash: "dot" } },
          { type: "line", xref: "paper", yref: "y2", x0: 0, x1: 1, y0: 1, y1: 1, line: { color: t.axisInk, width: 1 }, layer: "below" }
        ];
        if (c.second_pass_from_step != null) {
          shapes.push({ type: "rect", xref: "x", yref: "paper", x0: c.second_pass_from_step, x1: horizon, y0: 0, y1: 1, fillcolor: V.alpha(t.ink, 0.05), line: { width: 0 }, layer: "below" });
        }
        var ann = [{ xref: "x", yref: "paper", x: c.selected_peak.step, y: 1, xanchor: "right", yanchor: "top", xshift: -4, showarrow: false, text: "selected peak", font: { size: 10.5, color: t.ink2 } }];
        if (c.second_pass_from_step != null) ann.push({ xref: "x", yref: "paper", x: c.second_pass_from_step, y: 0.36, xanchor: "left", yanchor: "bottom", xshift: 4, showarrow: false, text: "second pass", font: { size: 10.5, color: t.ink2 } });
        var epMax = Math.max.apply(null, ep.y.filter(function (v) { return v > 0; })), epMin = Math.min.apply(null, ep.y.filter(function (v) { return v > 0; }));
        var eLo = Math.min(0.8, epMin * 0.9), eHi = Math.max(2, epMax * 1.15), eT = V.logTicks(eLo, eHi);
        var lay = V.layout(t, {
          height: 470, hovermode: "x unified", shapes: shapes, annotations: ann,
          margin: { l: 56, r: 14, t: 10, b: 44 },
          xaxis: { range: [0, horizon + 2], anchor: "y2", title: { text: "Training step" }, showspikes: true, spikecolor: t.axisInk, spikethickness: 1, spikedash: "solid", spikemode: "across" },
          yaxis: { domain: [0.42, 1], title: { text: "Percent" } },
          yaxis2: { domain: [0, 0.32], type: "log", title: { text: "Entropy proxy (×)" }, range: [Math.log10(eLo), Math.log10(eHi)], tickvals: eT.tickvals, ticktext: eT.ticktext }
        });
        return {
          plots: [{ data: data, layout: lay }],
          badges: [
            { strong: declineLabel(c.decline_label) },
            { text: "Selected peak " + f1(c.selected_peak.accuracy) + "% (step " + c.selected_peak.step + "), last validated " + f1(c.last_validated.accuracy) + "% (step " + c.last_validated.step + ")" }
          ],
          legend: [
            { type: "line", color: col, marker: true, label: c.cell + " validation" },
            { type: "line", dash: "dash", color: col, label: c.matched_sp_cell + " validation" },
            { type: "line", color: t.neutralBar, label: "IP training success" },
            { type: "line", color: t.ink, label: "Entropy proxy, " + roleName + " adapter" }
          ],
          table: {
            columns: ["Cell", "Selected peak (step)", "Last validated (step)", "Drop (points)", "Label"],
            numeric: [false, true, true, true, false],
            rows: cells.map(function (x) {
              return [x.cell, f1(x.selected_peak.accuracy) + " (" + x.selected_peak.step + ")", f1(x.last_validated.accuracy) + " (" + x.last_validated.step + ")", f1(x.drop_pp), declineLabel(x.decline_label)];
            })
          }
        };
      }
    });
  }

  function m4b(el, D) {
    var runs = D.localization.runs;
    var GROUP = { "Isolated policy, declining": "IP, declining", "Isolated policy, not declining": "IP, not declining", "Shared policy and single-agent RL": "SP or SA-RL", "Code": "Code" };
    function roleLabel(run, r) {
      if (r.parallel) return (r.role === "generator" ? "voters" : "workers") + " (parallel role)";
      if (run.cell.indexOf("SA-RL") === 0) return "generator";
      return r.role;
    }
    V.figure(el, {
      id: "fig-m4b", num: "Figure 4b", title: "Where the instability sits: per-role entropy and checkpoint change",
      controls: [{ type: "select", key: "run", label: "Run", options: runs.map(function (r) { return { v: r.cell, l: r.cell + " (" + (GROUP[r.group] || r.group) + ")" }; }) }],
      defaults: { run: "Voting-IP-1.7B-Math" },
      caption: "Per-token entropy and the KL between checkpoints ten steps apart, measured on fixed responses to the same validation problems and divided by each role's first probe (log scale). Dotted line: selected peak. In the declining isolated-policy runs, the parallel role's entropy and change between checkpoints are already high at the peak and keep rising, while the decision roles change only in brief jumps. In runs without a decline, the solving roles' change between checkpoints stays small.",
      render: function (s, t, W) {
        var run = runs.filter(function (r) { return r.cell === s.run; })[0];
        var wf = run.cell.split("-")[0] === "Orch" ? "Orch-Workers" : run.cell.split("-")[0];
        var isSA = run.cell.indexOf("SA-RL") === 0;
        function style(r) {
          if (r.parallel || isSA) return { color: isSA ? t.ink : t.wf[wf], width: 2.4, dash: "solid", size: 8 };
          if (r.role === "orchestrator") return { color: t.neutralBar, width: 1.8, dash: "dash", size: 6 };
          return { color: t.neutralMark, width: 1.8, dash: "solid", size: 6 };
        }
        function mk(key) {
          var data = [];
          run.roles.forEach(function (r) {
            var st = style(r);
            var tr = {
              type: "scatter", mode: "lines+markers", x: r.probes.map(function (p) { return p.step; }), y: r.probes.map(function (p) { return p[key]; }),
              line: { color: st.color, width: st.width, dash: st.dash }, marker: { size: st.size, color: st.color, line: { color: t.surface, width: 1.5 } },
              hoverinfo: "text",
              hovertext: r.probes.map(function (p) {
                return hoverLines(["<b>" + f2(p[key]) + "×</b>  " + roleLabel(run, r), "Probe at step " + p.step + " (next checkpoint " + p.step_next + ")"]);
              })
            };
            data.push(tr);
          });
          return data;
        }
        function lay(title, key) {
          var vals = [];
          run.roles.forEach(function (r) { r.probes.forEach(function (p) { vals.push(p[key]); }); });
          vals = vals.filter(function (v) { return v > 0; });
          var lo = Math.min.apply(null, vals) * 0.85, hi = Math.max.apply(null, vals) * 1.15, tk = V.logTicks(lo, hi);
          return V.layout(t, {
            height: 330, margin: { l: 52, r: 12, t: 8, b: 44 },
            shapes: [
              { type: "line", xref: "x", yref: "paper", x0: run.selected_peak_step, x1: run.selected_peak_step, y0: 0, y1: 1, line: { color: t.ink2, width: 1.3, dash: "dot" } },
              { type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 1, y1: 1, line: { color: t.axisInk, width: 1 }, layer: "below" }
            ],
            xaxis: { title: { text: "Probe step" } },
            yaxis: { type: "log", title: { text: title }, range: [Math.log10(lo), Math.log10(hi)], tickvals: tk.tickvals, ticktext: tk.ticktext }
          });
        }
        var legend = run.roles.map(function (r) { var st = style(r); return { type: "line", dash: st.dash === "dash" ? "dash" : null, marker: true, color: st.color, label: roleLabel(run, r) }; });
        legend.push({ type: "line", dash: "dot", color: t.ink2, label: "selected peak" });
        var rows = [];
        run.roles.forEach(function (r) { r.probes.forEach(function (p) { rows.push([roleLabel(run, r), p.step, f2(p.entropy_ratio), f2(p.kl_ratio)]); }); });
        return {
          twoCol: true,
          badges: [{ strong: run.cell + ": ", text: (GROUP[run.group] || run.group) + "; selected peak at step " + run.selected_peak_step + "; ratios to the first probe at step " + run.roles[0].base_probe_step }],
          plots: [
            { data: mk("entropy_ratio"), layout: lay("Entropy ratio (log)", "entropy_ratio"), title: "Per-token entropy" },
            { data: mk("kl_ratio"), layout: lay("Checkpoint KL ratio (log)", "kl_ratio"), title: "Change between checkpoints (KL)" }
          ],
          legend: legend,
          table: { columns: ["Role", "Probe step", "Entropy ratio", "KL ratio"], numeric: [false, true, true, true], rows: rows }
        };
      }
    });
  }

  function m4c(el, D) {
    var P = D.parameters;
    var CLASSES = ["IP parallel role", "IP single-instance role", "Eval-Opt IP role", "SP shared", "SA-RL generator"];
    var CLASS_LABEL = { "IP parallel role": "IP parallel role", "IP single-instance role": "IP single-instance role", "Eval-Opt IP role": "Eval-Opt IP role", "SP shared": "SP shared adapter", "SA-RL generator": "SA-RL generator" };
    var CLASS_SYM = { "IP parallel role": "circle", "IP single-instance role": "square", "Eval-Opt IP role": "triangle-up", "SP shared": "diamond", "SA-RL generator": "x" };
    V.figure(el, {
      id: "fig-m4c", num: "Figure 4c", title: "Parameter change per adapter against the AdamW noise model",
      controls: [{ type: "seg", key: "task", label: "Task", options: [{ v: "All", l: "Both" }].concat(TASK_OPTS) }],
      defaults: { task: "Math" },
      caption: "Each point is one adapter; color: N, its optimizer steps per training step; filled: Math, open: Code. Left: parameter change per training step against the pure-noise AdamW model with the same N (diagonal: the step count alone sets the distance). Right: excess cosine of consecutive 50-step changes over that model. Most adapters move about as far as their step count predicts. The parallel adapters of three of the four declining runs keep their direction far more than any other adapter.",
      render: function (s, t, W) {
        var A = P.all_adapters.filter(function (a) { return s.task === "All" || a.cell.slice(-4) === s.task || (s.task === "Code" && a.cell.slice(-4) === "Code"); });
        var cs = [[0, t.seqLo], [1, t.seqHi]];
        function hv(a) {
          return hoverLines(["<b>" + a.cell + "</b> · " + a.role, CLASS_LABEL[a.adapter_class] + "; N = " + f2(a.n_opt_steps_per_training_step),
            "Change per training step " + f2(a.rate) + " vs noise model " + f2(a.predicted_rate) + " (ratio " + f2(a.obs_over_pred) + ")",
            a.delta_cos != null ? "Δcos " + sgn(a.delta_cos, 3) + " over " + a.delta_cos_pairs + " pairs of 50-step changes" : "Δcos not measured"]);
        }
        var d1 = [], d2 = [], first = true;
        CLASSES.forEach(function (cl, ci) {
          ["Math", "Code"].forEach(function (task) {
            var g = A.filter(function (a) { return a.adapter_class === cl && a.cell.slice(-4) === task; });
            if (!g.length) return;
            var symbol = CLASS_SYM[cl] + (task === "Code" ? "-open" : "");
            var mk = {
              symbol: symbol, size: 10, color: g.map(function (a) { return a.n_opt_steps_per_training_step; }), colorscale: cs, cmin: 1, cmax: 5,
              line: task === "Code" ? { width: 2, color: g.map(function (a) { return a.n_opt_steps_per_training_step; }) } : { width: 1.2, color: t.surface }
            };
            if (task === "Code") { mk.line.colorscale = cs; mk.line.cmin = 1; mk.line.cmax = 5; }
            if (first) {
              mk.showscale = true;
              mk.colorbar = { title: { text: "N", font: { size: 11, color: t.ink2 } }, thickness: 10, len: 0.75, outlinewidth: 0, tickfont: { size: 10, color: t.ink2 }, tickvals: [1, 2, 3, 4, 5] };
              first = false;
            }
            d1.push({ type: "scatter", mode: "markers", x: g.map(function (a) { return a.predicted_rate; }), y: g.map(function (a) { return a.rate; }), marker: mk, hoverinfo: "text", hovertext: g.map(hv) });
            var gd = g.filter(function (a) { return a.delta_cos != null; });
            var mk2 = Object.assign({}, mk, { showscale: false, color: gd.map(function (a) { return a.n_opt_steps_per_training_step; }) });
            if (task === "Code") mk2.line = { width: 2, color: gd.map(function (a) { return a.n_opt_steps_per_training_step; }), colorscale: cs, cmin: 1, cmax: 5 };
            delete mk2.colorbar;
            d2.push({ type: "scatter", mode: "markers", x: gd.map(function (a, i) { return ci + V.jitter(i + ci * 31 + (task === "Code" ? 7 : 0), 0.22); }), y: gd.map(function (a) { return a.delta_cos; }), marker: mk2, hoverinfo: "text", hovertext: gd.map(hv) });
          });
        });
        var xs = A.map(function (a) { return a.predicted_rate; }).concat(A.map(function (a) { return a.rate; }));
        var lo = Math.max(0, Math.min.apply(null, xs) - 0.02), hi = Math.max.apply(null, xs) + 0.02;
        d1.unshift({ type: "scatter", mode: "lines", x: [lo, hi], y: [lo, hi], line: { color: t.axisInk, width: 1.2 }, hoverinfo: "skip" });
        var nar = narrow(W);
        var l1 = V.layout(t, {
          height: 380, margin: { l: 56, r: 10, t: 8, b: 48 },
          xaxis: { title: { text: "Noise-model change per training step" }, range: [lo, hi] },
          yaxis: { title: { text: "Observed change per training step" }, range: [lo, hi] },
          annotations: [{ xref: "x", yref: "y", x: hi, y: hi, xanchor: "right", yanchor: "top", xshift: -6, showarrow: false, text: "noise model", font: { size: 10.5, color: t.ink2 } }]
        });
        var l2 = V.layout(t, {
          height: 380, margin: { l: 56, r: 10, t: 8, b: 64 },
          shapes: [hline(0, t.ink2, 1.2)],
          annotations: [{ xref: "paper", yref: "y", x: 1, y: 0, xanchor: "right", yanchor: "top", showarrow: false, text: "noise model", font: { size: 10.5, color: t.ink2 } }],
          xaxis: { tickvals: [0, 1, 2, 3, 4], ticktext: CLASSES.map(function (c) { return CLASS_LABEL[c].replace(" ", "<br>"); }), range: [-0.6, 4.6], showgrid: false, tickfont: { size: nar ? 9 : 10.5, color: t.ink2 } },
          yaxis: { title: { text: "Δcos over the noise model" } }
        });
        var legend = CLASSES.map(function (c) {
          var ty = { circle: "dot", square: "square", "triangle-up": "triangle", diamond: "diamond", x: "tick" }[CLASS_SYM[c]];
          return { type: ty, color: t.ink2, label: CLASS_LABEL[c] };
        });
        legend.push({ type: "rect", color: t.seqLo, label: "N = 1" });
        legend.push({ type: "rect", color: t.seqHi, label: "N = 5" });
        return {
          twoCol: true,
          plots: [{ data: d1, layout: l1, title: "Distance: observed against noise model" }, { data: d2, layout: l2, title: "Direction persistence (Δcos)" }],
          legend: legend,
          table: {
            columns: ["Cell", "Role", "Adapter", "N", "Change", "Noise model", "Ratio", "Δcos (pairs)"], numeric: [false, false, false, true, true, true, true, true],
            rows: A.map(function (a) { return [a.cell, a.role, CLASS_LABEL[a.adapter_class], f2(a.n_opt_steps_per_training_step), f2(a.rate), f2(a.predicted_rate), f2(a.obs_over_pred), a.delta_cos != null ? sgn(a.delta_cos, 3) + " (" + a.delta_cos_pairs + ")" : "—"]; })
          }
        };
      }
    });
  }

  function m4d(el, D) {
    var C = D.controls;
    var SER = [
      { k: "reference", l: "Reference: Voting-IP-1.7B-Math", c: 1, dash: "solid", on: true },
      { k: "merged_step_T2", l: "Merged steps: one generator step per training step", c: 6, dash: "solid", on: true },
      { k: "single_role", l: "Single role: no workflow, no aggregator", c: 5, dash: "solid", on: true },
      { k: "shared_policy", l: "Shared policy: Voting-SP-1.7B-Math", c: 1, dash: "dash", on: true },
      { k: "per_voter_reward", l: "Per-voter reward", c: 4, dash: "solid", on: false },
      { k: "shared_policy_reweighted_T1", l: "Shared policy, loss reweighted by role length", c: 7, dash: "solid", on: false }
    ];
    var NOTE = {
      reference: "The parallel generator adapter takes one optimizer step per instance, three per training step.",
      per_voter_reward: "Aggregator rewarded on the final answer, each voter on its own answer.",
      single_role: "No workflow and no aggregator; one generator trained on its own answer, validated as a single agent.",
      merged_step_T2: "All of a training step's generator samples in one optimizer step; aggregator loss at one third. Stays near its peak over its validated horizon: it delays the decline at the same learning speed.",
      shared_policy: "Peak late in its validated horizon; entropy proxy pooled over roles.",
      shared_policy_reweighted_T1: "Each output's loss scaled by its role's mean length; declines like the isolated policies.",
      "sa_rl_0.6B": "Single-agent RL.", "sa_rl_1.7B": "Single-agent RL.", "sa_rl_4B": "Single-agent RL; peak at the last checkpoint."
    };
    var defaults = {};
    SER.forEach(function (x) { defaults[x.k] = x.on; });
    V.figure(el, {
      id: "fig-m4d", num: "Figure 4d", title: "Controls on Voting at 1.7B (Math)",
      controls: SER.map(function (x) { return { type: "check", key: x.k, label: x.l, swatch: true }; }),
      defaults: defaults,
      caption: "Validation accuracy of the reference run and its controls; dots: each run's peak over its validated horizon. The single-role and per-voter-reward controls also decline, so the decline needs neither the workflow nor the broadcast reward; the reweighted shared policy declines too, so the decline follows the gradient share. Merging the parallel role's optimizer steps into one per training step delays the decline at the same learning speed. The table lists the paper's control rows; change is last minus peak.",
      render: function (s, t) {
        var data = [];
        SER.forEach(function (x) {
          if (!s[x.k]) return;
          var vs = C.validation_series[x.k];
          var row = C.rows.filter(function (r) { return r.key === x.k; })[0];
          var color = t.c[x.c];
          data.push({
            type: "scatter", mode: "lines", x: vs.points.map(function (p) { return p[0]; }), y: vs.points.map(function (p) { return p[1]; }),
            line: { color: color, width: 2, dash: x.dash }, hovertemplate: x.l + ": <b>%{y:.1f}%</b><extra></extra>"
          });
          if (row) data.push({
            type: "scatter", mode: "markers", x: [row.peak_step], y: [row.peak],
            marker: { size: 10, color: color, line: { color: t.surface, width: 2 } }, hovertemplate: x.l + " peak: <b>%{y:.1f}%</b> at step %{x}<extra></extra>"
          });
        });
        var lay = V.layout(t, {
          height: 380, hovermode: "x unified",
          xaxis: { title: { text: "Training step" }, showspikes: true, spikecolor: t.axisInk, spikethickness: 1, spikedash: "solid", spikemode: "across" },
          yaxis: { title: { text: "Validation accuracy (%)" } }
        });
        var rows = C.rows.filter(function (r) { return r.in_paper_table; });
        var tbl = document.createElement("div");
        V.renderTable(tbl, {
          title: "Control rows",
          columns: ["Setting", "Reward", "N", "Peak (step)", "Last (step)", "Change", "Entropy proxy, late ratio", "Note"],
          numeric: [false, false, true, true, true, true, true, false], noteCol: 7, wrapCol: 0,
          rows: rows.map(function (r) {
            var ep = r.entropy_proxy_late_ratio;
            var epTxt = ep == null ? "not comparable" : Array.isArray(ep) ? ep.map(f1).join(", ") : f1(ep) + (r.entropy_proxy_pooled_over_roles ? " (pooled)" : "");
            return [r.setting, r.reward, r.N, f1(r.peak) + " (" + r.peak_step + ")", f1(r.last) + " (" + r.last_step + ")", r.change_display === "none" ? "none" : String(r.change_display).replace("-", "−"), epTxt, NOTE[r.key] || ""];
          })
        });
        return {
          plots: [{ data: data, layout: lay }],
          swatches: SER.reduce(function (o, x) { o[x.k] = { type: "line", dash: x.dash === "dash" ? "dash" : null, color: t.c[x.c] }; return o; }, {}),
          below: tbl
        };
      }
    });
  }

  window.FIGURES = {
    m1: function (el, D) { m1a(el, D); m1b(el, D); m1c(el, D); m1d(el, D); },
    m2: function (el, D) { m2a(el, D); m2b(el, D); m2c(el, D); m2d(el, D); },
    m3: function (el, D) { m3a(el, D); m3b(el, D); m3c(el, D); m3d(el, D); m3e(el, D); },
    m4: function (el, D) { m4a(el, D); m4b(el, D); m4c(el, D); m4d(el, D); }
  };
})();
