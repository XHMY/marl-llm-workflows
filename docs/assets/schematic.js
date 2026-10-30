/* Inline SVG schematic of the three workflows and of the two policy-sharing strategies. */
(function () {
  "use strict";
  var n = 0;

  function defs(id) {
    return '<defs>' +
      '<marker id="ah-' + id + '" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="ah" d="M0 0 L10 5 L0 10 z"/></marker>' +
      '<marker id="ahr-' + id + '" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="ah-reward" d="M0 0 L10 5 L0 10 z"/></marker>' +
      '<marker id="aha-' + id + '" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="ah-adapter" d="M0 0 L10 5 L0 10 z"/></marker>' +
      '</defs>';
  }
  function node(x, y, w, h, kind, label, fs) {
    return '<rect class="n-' + kind + '" x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="7" stroke-width="1.2"/>' +
      '<text class="t-' + kind + '" x="' + (x + w / 2) + '" y="' + (y + h / 2 + 4) + '" text-anchor="middle" font-size="' + (fs || 12) + '" font-weight="600">' + label + '</text>';
  }
  function stack(x, y, w, h, kind, label, fs) {
    var s = "";
    s += '<rect class="n-' + kind + '" x="' + (x + 10) + '" y="' + (y - 10) + '" width="' + w + '" height="' + h + '" rx="7" stroke-width="1.2"/>';
    s += '<rect class="n-' + kind + '" x="' + (x + 5) + '" y="' + (y - 5) + '" width="' + w + '" height="' + h + '" rx="7" stroke-width="1.2"/>';
    s += node(x, y, w, h, kind, label, fs);
    return s;
  }
  function pill(x, y, w, h, label, fs) {
    return '<rect class="pill" x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="' + h / 2 + '" stroke-width="1"/>' +
      '<text class="t-ink2" x="' + (x + w / 2) + '" y="' + (y + h / 2 + 4) + '" text-anchor="middle" font-size="' + (fs || 11.5) + '">' + label + '</text>';
  }
  function arrow(id, d, cls) {
    var m = cls === "edge-reward" ? "ahr-" : cls === "adapter-edge" ? "aha-" : "ah-";
    return '<path class="' + (cls || "edge") + '" d="' + d + '" marker-end="url(#' + m + id + ')"/>';
  }
  function reward(id, xs) {
    var s = '<rect class="reward-bar" x="8" y="198" width="344" height="30" rx="8" stroke-width="1.2"/>' +
      '<text class="t-ink" x="180" y="217" text-anchor="middle" font-size="11.5" font-weight="600">Episode outcome reward, broadcast to every role</text>';
    xs.forEach(function (p) { s += arrow(id, "M" + p[0] + " 197 L" + p[0] + " " + p[1], "edge-reward"); });
    return s;
  }
  function svg(id, body, label) {
    return '<svg viewBox="0 0 360 236" role="img" aria-label="' + label + '">' + defs(id) + body + "</svg>";
  }

  function voting() {
    var id = "s" + (++n), b = "";
    b += pill(8, 58, 62, 30, "Problem");
    b += arrow(id, "M71 73 L96 73");
    b += stack(100, 56, 92, 36, "solving", "Voter ×3");
    b += '<text class="t-muted" x="126" y="110" text-anchor="start" font-size="10.5">parallel role</text>';
    b += arrow(id, "M203 73 L226 73");
    b += node(230, 55, 104, 36, "decision", "Aggregator");
    b += arrow(id, "M282 92 L282 120");
    b += pill(236, 122, 84, 26, "Final answer");
    b += arrow(id, "M278 149 L278 196");
    b += '<text class="t-muted" x="272" y="176" text-anchor="end" font-size="10.5">graded</text>';
    b += reward(id, [[116, 94], [328, 93]]);
    return svg(id, b, "Voting: three voters answer in parallel, an aggregator selects the final answer, and the episode reward is broadcast to every role.");
  }
  function orch() {
    var id = "s" + (++n), b = "";
    b += pill(4, 58, 54, 30, "Problem", 10);
    b += arrow(id, "M58 73 L63 73");
    b += node(65, 55, 90, 36, "planning", "Orchestrator", 11);
    b += arrow(id, "M156 73 L165 73");
    b += stack(168, 56, 70, 36, "solving", "Worker ×3", 11);
    b += '<text class="t-muted" x="186" y="110" text-anchor="start" font-size="10.5">parallel role</text>';
    b += arrow(id, "M249 73 L256 73");
    b += node(259, 55, 95, 36, "decision", "Synthesizer", 11);
    b += arrow(id, "M296 92 L296 120");
    b += pill(264, 122, 64, 26, "Answer");
    b += arrow(id, "M292 149 L292 196");
    b += '<text class="t-muted" x="286" y="176" text-anchor="end" font-size="10.5">graded</text>';
    b += reward(id, [[110, 93], [180, 94], [346, 93]]);
    return svg(id, b, "Orch-Workers: an orchestrator plans, three workers write candidate solutions in parallel, a synthesizer writes the final answer, and the episode reward is broadcast to every role.");
  }
  function evalopt() {
    var id = "s" + (++n), b = "";
    b += pill(8, 58, 62, 30, "Problem");
    b += arrow(id, "M71 73 L94 73");
    b += node(98, 55, 100, 36, "solving", "Generator");
    b += arrow(id, "M199 73 L232 73");
    b += node(236, 55, 100, 36, "decision", "Evaluator");
    b += arrow(id, "M276 54 C 276 32, 158 32, 158 52");
    b += '<text class="t-ink2" x="217" y="13" text-anchor="middle" font-size="10.5">verdict not “correct”: revise</text>';
    b += '<text class="t-muted" x="217" y="26" text-anchor="middle" font-size="10">up to three generator turns on Math</text>';
    b += '<text class="t-muted" x="120" y="110" text-anchor="start" font-size="10.5">first attempt, then revisions</text>';
    b += arrow(id, "M286 92 L286 120");
    b += pill(244, 122, 84, 26, "Final answer");
    b += arrow(id, "M282 149 L282 196");
    b += '<text class="t-muted" x="276" y="176" text-anchor="end" font-size="10.5">graded</text>';
    b += reward(id, [[110, 93], [332, 93]]);
    return svg(id, b, "Eval-Opt: a generator answers, an evaluator returns a verdict and critique and sends the answer back for revision, and the episode reward is broadcast to both roles.");
  }

  function policyCard() {
    var id = "s" + (++n), b = "";
    b += defs(id);
    function half(x0, title, sub, shared) {
      var s = "";
      s += '<text class="t-ink" x="' + (x0 + 130) + '" y="18" text-anchor="middle" font-size="13" font-weight="700">' + title + "</text>";
      s += '<text class="t-muted" x="' + (x0 + 130) + '" y="34" text-anchor="middle" font-size="11">' + sub + "</text>";
      s += stack(x0 + 22, 118, 96, 34, "solving", "Voter ×3", 11.5);
      s += node(x0 + 150, 108, 96, 34, "decision", "Aggregator", 11.5);
      if (shared) {
        s += '<rect class="adapter" x="' + (x0 + 84) + '" y="50" width="96" height="28" rx="6" stroke-width="1.2"/>';
        s += '<text class="t-ink" x="' + (x0 + 132) + '" y="68" text-anchor="middle" font-size="11.5" font-weight="600">one adapter</text>';
        s += arrow(id, "M" + (x0 + 112) + " 79 L" + (x0 + 80) + " 104", "adapter-edge");
        s += arrow(id, "M" + (x0 + 152) + " 79 L" + (x0 + 190) + " 106", "adapter-edge");
      } else {
        s += '<rect class="adapter" x="' + (x0 + 18) + '" y="50" width="108" height="28" rx="6" stroke-width="1.2"/>';
        s += '<text class="t-ink" x="' + (x0 + 72) + '" y="68" text-anchor="middle" font-size="11" font-weight="600">voter adapter</text>';
        s += '<rect class="adapter" x="' + (x0 + 136) + '" y="50" width="124" height="28" rx="6" stroke-width="1.2"/>';
        s += '<text class="t-ink" x="' + (x0 + 198) + '" y="68" text-anchor="middle" font-size="11" font-weight="600">aggregator adapter</text>';
        s += arrow(id, "M" + (x0 + 72) + " 79 L" + (x0 + 72) + " 104", "adapter-edge");
        s += arrow(id, "M" + (x0 + 198) + " 79 L" + (x0 + 198) + " 106", "adapter-edge");
      }
      s += '<text class="t-ink2" x="' + (x0 + 130) + '" y="178" text-anchor="middle" font-size="11">' +
        (shared ? "every role trains the same weights" : "the three voter instances share one adapter") + "</text>";
      return s;
    }
    b += half(0, "Isolated policies (IP)", "one adapter per role type", false);
    b += '<line x1="270" y1="8" x2="270" y2="184" stroke="var(--grid)" stroke-width="1"/>';
    b += half(280, "Shared policy (SP)", "one adapter for all roles", true);
    return '<svg viewBox="0 0 548 190" role="img" aria-label="Isolated policies give each role type its own LoRA adapter, and the parallel voter instances share one; a shared policy trains one adapter for all roles. Voting shown.">' + b + "</svg>";
  }

  window.SCHEMATIC = { voting: voting, orch: orch, evalopt: evalopt, policyCard: policyCard };
})();
