# PLAN.md - fm-hybrid-site: a site presenting the fm-hybrid results

run_id: 20261010-site
tier: build - fresh repo, four features with separate gates, a shared data/gate layer, a design pass and a review
classification: fan-out - s0 gates -> s1 scaffold+export -> {s2 flow, s3 quant, s4a trainer -> s4b widget, s5 prose, s7 review} -> s6 design
routing basis: routing.md derived 2026-10-08; observed.md 55 entries; overrides: s6 opus high (user brief); explore-pick: s2-flow, s3-quant, s4a-trainer on haiku
compute: this laptop, CPU, /home/josh/miniconda3/envs/annoflow/bin/python (torch 2.4, playwright installed for gates), node 26, port 8766
sources: ~/repos/fm-hybrid (fm.py, results.jsonl, README.md); reuse from ~/repos/flow-matching-explainer/site/js and tools

## Goals
- A single page, plain HTML/CSS/vanilla JS, no framework, no external requests, that presents baseline flow matching against the best hybrid (note joint16_expect: head joint, decode expect, offset_target expect; bits 4, R 6).
- (1) prior-to-data animation for both datasets and both methods: draggable t slider, looping play button.
- (2) a view of the quantisation: 16 Gray-coded velocity buckets over [-R, R] per component, where true velocities land, and the offset correction on top of the expected centre.
- (3) in-browser training of both methods in lockstep, smooth (worker off the main thread), energy distance over steps plotted hybrid against baseline.
- (4) plain prose describing both methods and the honest result (the hybrid did not beat the baseline), passing ai_lint.
- Design: artistic, modern, sleek; strong decisions, animation allowed; reduced-motion respected.

## Constraints
- Priority: fewest files and lines, then cost. No README beyond a run line. One check script with subcommands plus one node gate.
- Feature widgets build their DOM inside their section container from JS, so parallel nodes never edit index.html; only s1, s5 and s6 edit index.html.
- All colours in canvases come from CSS custom properties (getComputedStyle), so the design node can retheme without touching maths.

## Success criteria (each machine-checkable)
- [ ] `python tools/check.py data` - exported data matches the data contract
- [ ] `python tools/check.py html` - page parses, ids present, relative refs resolve, scripts and worker scripts pass node --check
- [ ] `python tools/check.py prose` - ai_lint exit 0 on site/index.html; #methods text 350 to 700 words
- [ ] `python tools/check.py serve` - every page URL returns 200 on port 8766
- [ ] `python tools/check.py browser` - headless chromium: no console errors; the DOM contract of every section holds; canvases non-blank; slider, play, next-particle and training start behave
- [ ] `node tools/check_trainer.js` - JS trainer matches the Python fixture for both methods, passes a finite-difference gradient check, and learns
- [ ] `python tools/check.py review` - .pyramid/review.json has no blockers

## Run
| run | nodes | expected USD | note |
|---|---|---|---|
| 20261010-site | 9 nodes, one workflow | nodes 1.6, orchestrator ~5 | wall clock ~50 min |

| id | task_type | agent | model | effort | depends_on | explore | gate |
|---|---|---|---|---|---|---|---|
| s0-gates | implement | implementer | sonnet | medium | - | no | all subcommands fail cleanly (exit 1, no traceback) on the empty repo |
| s1-scaffold | implement | implementer | sonnet | medium | s0-gates | no | check.py data && html && serve |
| s2-flow | implement | implementer | haiku | medium | s1-scaffold | yes | check.py browser --section flow |
| s3-quant | implement | implementer | haiku | medium | s1-scaffold | yes | check.py browser --section quant |
| s4a-trainer | implement | implementer | haiku | medium | s1-scaffold | yes | node tools/check_trainer.js |
| s4b-widget | implement | implementer | sonnet | medium | s4a-trainer | no | check.py browser --section train |
| s5-prose | write | implementer | haiku | medium | s1-scaffold | no | check.py prose |
| s7-review | review | implementer | sonnet | medium | s5-prose, s4a-trainer | no | check.py review |
| s6-design | implement | implementer | opus | high | s2, s3, s4b, s5 | no | check.py all (data html prose serve browser) && node tools/check_trainer.js |

## Contracts

### Data (site/data/, produced by tools/export.py, checked by check.py data)
- manifest.json: {"frames":49,"n":1000,"bounds":[-3.5,3.5],"bits":4,"R":6,"datasets":["moons","gaussians8"],"methods":["fm","hybrid"],"traj":{ds:{method:file}},"data":{ds:file}}; unknown extra keys allowed everywhere.
- traj_{ds}_{method}.bin: little-endian Float32, frames*n*2 values, layout [frame][particle][xy]; frame 0 is the prior and identical for both methods of a dataset (same noise seed); frame k is the midpoint-integrator state after k of 48 steps; all finite, |x| <= 4.
- data_{ds}.bin: Float32 [1000][2], a real data sample from fm.data.
- quant_{ds}.json: {"bits":4,"R":6,"gray":[gray(i) for i in 0..15],"centres":[16],"n":512,"xt":[n][2],"t":[n],"v":[n][2],"bucket":[n][2] (true bucket index 0..15),"probs":[n][2][16] (softmax mass per bucket in bucket order, not Gray order),"expect":[n][2],"offset":[n][2],"vhat":[n][2]}; vhat = expect + offset and expect = sum(probs*centres) within 1e-4; bucket = clamp(floor((v+R)/(2R)*16), 0, 15); probs rows sum to 1 within 1e-4.
- fixture.json: {"hidden":8,"n":16,"x":[16][2],"t":[16],"v":[16][2],"fm":{"W1":[8][3],"b1":[8],"W2":[8][8],"b2":[8],"W3":[2][8],"b3":[2],"vhat":[16][2],"loss":f},"hybrid":{same with W3 [34][8], b3 [34], "vhat","loss","ce","off"}}. Network: input [x, y, t] -> Linear -> SiLU -> Linear -> SiLU -> Linear, PyTorch weight layout W[out][in]. Hybrid output: o[0:32] viewed [2][16] are logits per component where position j is the logit of Gray code j (bucket ungray(j)), o[32:34] is the offset; vhat = sum_j softmax(logits)_j * centre(ungray(j)) + offset; loss = ce + off, ce = cross-entropy of the 2n logit rows against gray(bucket), off = mse(offset, v - expect) with expect treated as a constant. fm: vhat = o, loss = mse(o, v). Fixture weights are random (seeded), not trained.
- results.json: {"best_note":"joint16_expect","cfg":{...},"datasets":{ds:{"baseline":{"mean","std","seeds":[3]},"hybrid":{"mean","std","seeds":[3]}}},"log":[{"note","cfg","moons","gaussians8"} per note in file order]}; std is population std (ddof 0), matching README.

### DOM (built by each widget's JS inside its section; checked by check.py browser)
- section#flow: select.flow-dataset (options moons, gaussians8); two canvas.flow-canvas[data-method=fm|hybrid]; input[type=range].t-slider min 0 max 1 step 0.001; .t-readout text /^t = [01]\.\d\d$/; button.play[aria-pressed]. Setting the slider to 0.5 (input event) updates the readout and redraws; clicking play sets aria-pressed true and t changes within 500 ms; the loop wraps from 1 to 0; both canvases non-blank and the t=1 image differs from t=0.
- section#quant: select.quant-dataset; one or more canvas.quant-canvas; two .bin-strip[data-component=0|1] each holding exactly 16 .bin-label whose text is a 4-char 0/1 string, consecutive labels differing in exactly one bit, in bucket order from -R to R; button.next-particle; .offset-readout whose text changes on click; the view shows the histogram of true v components over the buckets and, for the selected particle, the probability mass per bucket, the expected centre, the offset arrow and the true v.
- section#train: select.train-dataset; button.train-start, button.train-pause, button.train-reset; .step-readout (integer); .metric-readout; two canvas.train-dist[data-method=fm|hybrid]; canvas.train-metric. Clicking start increases .step-readout within 8 s and the metric canvas becomes non-blank once a metric message arrives (within 20 s).
- section#methods: prose between <!-- prose:methods --> and <!-- /prose:methods -->, 350 to 700 words; section#results: a table of baseline against hybrid per dataset and the iteration log.
- header#top hero; nav links to #flow #quant #train #methods #results. Required ids: top, flow, quant, train, methods, results.
- Nothing runs training or plays animation at load; prefers-reduced-motion disables autoplay and hero motion.

### JS trainer (site/js/trainer.js, UMD like the explainer's; checked by check_trainer.js)
- new FMTrainer(method, dataset, opts) with method "fm"|"hybrid", dataset "moons"|"gaussians8"; opts {seed, hidden=64, batch=128, lr=2e-3}; .step() -> loss; .steps; .losses; .sample(n, nSteps, seed) -> Float32Array [n][2] by midpoint integration; .setParams({W1,b1,W2,b2,W3,b3}) in fixture layout; .lossGrad(xt, t, v, n) -> {loss, vhat:Float64Array} and fills .grads (same shapes as .params, order W1,b1,W2,b2,W3,b3 flattened row-major) without an optimiser step; statics FMTrainer.getDataset(name, n, seed) matching fm.data (moons scale 2.4, noise 0.08; gaussians8 radius 2, std 0.2; clamp 3), FMTrainer.energyDistance(a, b) over Float32Array [n][2], FMTrainer.gray(i), FMTrainer.ungray(g, bits), FMTrainer.Rng.
- check_trainer.js: for each method, setParams from fixture, lossGrad on the fixture batch: |loss - fixture.loss| < 1e-5 and max |vhat - fixture.vhat| < 1e-5; finite-difference gradient check on hidden 8 (central differences, h 1e-5) with max relative error < 1e-4 over all parameters; 400 lockstep steps on moons seed 1 with hidden 64: mean of last 50 losses < 0.9 * first 50 for both methods; sample(500, 20) finite inside [-4, 4]; energyDistance(sample, data) after 400 steps < at step 0 for both.

### Worker (site/js/trainer-worker.js) messages
- in: {type:"init", dataset, seed, epoch} | {type:"run"} | {type:"pause"} | {type:"reset"}; epoch echoed on replies.
- out: {type:"data", buf} 1000 real points; {type:"progress", step, loss:{fm, hybrid}, done} about every 10 lockstep steps; {type:"metric", step, ed:{fm, hybrid}, samples:{fm, hybrid}} every 100 steps with 500 samples each (nfe 20) and energy distance against 500 data points; {type:"state", step, running, done}; {type:"error", message}. MAX_STEPS 3000.

## Nodes

#### s0-gates
- goal: tools/check.py (subcommands data, html, prose, serve, browser [--section], review, all) and tools/check_trainer.js implementing the contracts above; copy tools/ai_lint.py and tools/ai_signs.json from the explainer; tools/serve.sh (tmux http.server 8766 on site/).
- steps: read the explainer gates for reusable parsing; write check.py against the Data and DOM contracts; write check_trainer.js against the JS trainer contract; browser subcommand starts http.server on 8766 if nothing answers, uses playwright chromium headless, fails on any console error or pageerror, non-blank = at least 0.5% of canvas pixels differ from the top-left pixel; every subcommand exits 1 with a one-line reason when its inputs are missing, 2 only for usage errors.
- files: tools/check.py, tools/check_trainer.js, tools/ai_lint.py, tools/ai_signs.json, tools/serve.sh
- gate: `cd ~/repos/fm-hybrid-site && for s in data html prose serve review browser; do python tools/check.py $s; [ $? -eq 1 ] || exit 1; done; node tools/check_trainer.js; [ $? -eq 1 ] && python tools/check.py --help`
#### s1-scaffold
- goal: tools/export.py produces every file of the Data contract from ~/repos/fm-hybrid (import fm by path); site/index.html skeleton with the required ids, nav, empty widget sections and script tags for js/flow.js, js/quant.js, js/train-widget.js (deferred); site/css/style.css with the design tokens (CSS variables for bg, surface, text, muted, accent fm, accent hybrid, border, canvas-bg, radius, fonts; dark default) and base styles for .widget, .controls, .btn, .range, .readout.
- steps: train fm and hybrid (joint16_expect cfg) on both datasets, seed 0, 3000 steps, sequentially; record trajectories; compute quant and fixture with fm.predict_v and fm.loss_fn on 2-hidden-layer nets; write results.json from results.jsonl; write the skeleton and tokens.
- read: ~/repos/fm-hybrid/fm.py, ~/repos/fm-hybrid/results.jsonl, ~/repos/flow-matching-explainer/site/index.html (structure only)
- files: tools/export.py, site/data/*, site/index.html, site/css/style.css, .gitignore
- gate: `python tools/check.py data && python tools/check.py html && python tools/check.py serve`
#### s2-flow
- goal: site/js/flow.js, the prior-to-data player per the DOM contract, side by side fm against hybrid, particles coloured by final position hue, linear interpolation between frames, loop of about 6 s per cycle via requestAnimationFrame.
- read: ~/repos/flow-matching-explainer/site/js/playground.js (slider, play, interpolation patterns), site/data/manifest.json, site/index.html, site/css/style.css
- files: site/js/flow.js
- gate: `python tools/check.py browser --section flow`
#### s3-quant
- goal: site/js/quant.js per the DOM contract: bucket strips with Gray labels, histogram of true v per component, selected particle with probability mass, expected centre, offset arrow, true v and vhat.
- read: site/data/quant_moons.json (shape), site/index.html, site/css/style.css
- files: site/js/quant.js
- gate: `python tools/check.py browser --section quant`
#### s4a-trainer
- goal: site/js/trainer.js and site/js/trainer-worker.js per the JS trainer and Worker contracts, extended from the explainer's files.
- read: ~/repos/flow-matching-explainer/site/js/trainer.js, ~/repos/flow-matching-explainer/site/js/trainer-worker.js, ~/repos/fm-hybrid/fm.py, site/data/fixture.json, tools/check_trainer.js
- files: site/js/trainer.js, site/js/trainer-worker.js
- gate: `node tools/check_trainer.js`
#### s4b-widget
- goal: site/js/train-widget.js per the DOM contract: controls, two distribution canvases drawn from metric samples over the real data, a metric chart with both energy-distance curves against steps on a log axis, smooth drawing via requestAnimationFrame.
- read: ~/repos/flow-matching-explainer/site/js/train-widget.js, site/js/trainer-worker.js (message header), site/index.html, site/css/style.css
- files: site/js/train-widget.js
- gate: `python tools/check.py browser --section train`
#### s5-prose
- goal: the #methods prose (350 to 700 words) and the #results table in site/index.html: what flow matching learns and how it samples; how the hybrid quantises velocity into Gray-coded buckets, predicts a distribution over them, decodes the expectation and adds an offset trained against the remaining gap; what the numbers say, including that the hybrid did not beat the baseline and the criterion was not met.
- read: ~/repos/fm-hybrid/fm.py, ~/repos/fm-hybrid/README.md, site/index.html, tools/ai_signs.json
- files: site/index.html (only inside #methods and #results)
- gate: `python tools/check.py prose`
#### s7-review
- goal: one lens, fidelity to the source: do the #methods prose, the #results table and trainer.js's loss and decode match fm.py and results.jsonl; write .pyramid/review.json {blockers:[], majors:[], minors:[]}.
- read: `git add -N . && git diff HEAD -- site/index.html site/js/trainer.js`, ~/repos/fm-hybrid/fm.py, ~/repos/fm-hybrid/results.jsonl
- files: .pyramid/review.json
- gate: `python tools/check.py review`
#### s6-design
- goal: the design pass: typography, palette, hero with an animated canvas, section rhythm, control styling, responsive layout, motion with reduced-motion fallback; may restructure index.html and restyle widget DOM, keeping every DOM-contract hook and id.
- read: site/index.html, site/css/style.css, site/js/*.js
- files: site/index.html, site/css/style.css, site/js/*.js (style and layout only), site/assets/*
- gate: `python tools/check.py all && node tools/check_trainer.js`

## Stop conditions
Gate failing after escalation; unforeseen blocker; contract problem; node cost past 2.4 USD.

## Approval
- [x] approved by user at 2026-10-10T08:19:59Z ("Approve as planned"); approval covers the whole run

## Run log
| run | started | passed/total | retries | escalations | nodes USD | orchestrator USD | wall clock | notes |
|---|---|---|---|---|---|---|---|---|
| 20261010-site | 08:20Z | 8/9 | 1 (s7-review retry) | 0 | 1.44 | 2.91 (at record) | 17 min | all gates pass; review found 3 blockers the gates could not see (offset detach, moons noise twice, joint-head sentence) plus a worker that could not pause; fixed directly |
| 20261010-site-review2 | 08:38Z | 0/1 | 0 | 0 | 0.15 | - | 1 min | earlier blockers confirmed fixed; one new prose blocker (soft decoding and gradients) fixed directly, review.json marks it resolved; final gates pass; committed 085774b |
