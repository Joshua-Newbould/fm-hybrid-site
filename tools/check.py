#!/usr/bin/env python3
"""check.py - gates for the fm-hybrid site.

usage: check.py {data|html|prose|serve|browser [--section flow|quant|train]|review|all}
Prints FAIL lines and exits 1 on any failure, 0 on pass, 2 on usage error.
"""
import glob
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urlsplit

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SITE = os.path.join(ROOT, "site")
INDEX = os.path.join(SITE, "index.html")
DATA = os.path.join(SITE, "data")
PORT = 8766
BASE = f"http://127.0.0.1:{PORT}/"
IDS = ["top", "flow", "quant", "train", "methods", "results"]
SKIP = ("http:", "https:", "//", "#", "data:", "mailto:", "tel:", "javascript:")
DS = ["moons", "gaussians8"]
MS = ["fm", "hybrid"]

LOADER_RE = re.compile(r"(?:\bnew\s+Worker|\bimportScripts)\s*\(\s*([^)]*?)\s*\)")
JS_LIT_RE = re.compile(r"[\"']([\w./-]+\.js)[\"']")
IDENT_RE = re.compile(r"^[A-Za-z_$][\w$.]*$")


def worker_scripts():
    out = set()
    for js in sorted(glob.glob(os.path.join(SITE, "js", "*.js"))):
        text = open(js, encoding="utf-8").read()
        for m in LOADER_RE.finditer(text):
            arg = m.group(1)
            lits = JS_LIT_RE.findall(arg)
            if not lits and IDENT_RE.match(arg):
                name = re.escape(arg.split(".")[-1])
                for am in re.finditer(r"\b" + name + r"\s*=\s*([^;]*);", text):
                    lits.extend(JS_LIT_RE.findall(am.group(1)))
            for lit in lits:
                base = SITE if "/" in lit else os.path.dirname(js)
                disk = os.path.normpath(os.path.join(base, lit.lstrip("/")))
                out.add((os.path.relpath(disk, SITE).replace(os.sep, "/"), disk))
    return sorted(out)


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs, self.scripts, self.ids, self.tags = [], [], [], set()

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        d = dict(attrs)
        for k, v in attrs:
            if k in ("src", "href", "poster", "data") and v:
                self.refs.append((tag, v.strip()))
        if tag == "script" and d.get("src"):
            self.scripts.append(d["src"].strip())
        if d.get("id"):
            self.ids.append(d["id"])

    handle_startendtag = handle_starttag


def parse_index():
    p = Page()
    p.feed(open(INDEX, encoding="utf-8").read())
    p.close()
    return p


def rel_refs(p):
    return [(t, u) for t, u in p.refs if not u.lower().startswith(SKIP) and urlsplit(u).path]


def finite(*vals):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)


def flat(a):
    for e in a:
        if isinstance(e, list):
            yield from flat(e)
        else:
            yield e


def shape(a):
    s = []
    while isinstance(a, list):
        s.append(len(a))
        a = a[0] if a else None
    return s


def load_json(path, fails):
    try:
        return json.load(open(path))
    except Exception as e:
        fails.append(f"{os.path.relpath(path, ROOT)}: {'missing' if not os.path.exists(path) else e}")


def check_data():
    fails = []
    mpath = os.path.join(DATA, "manifest.json")
    man = load_json(mpath, fails)
    if man is None:
        return fails
    for k, want in (("frames", 49), ("n", 1000), ("bounds", [-3.5, 3.5]), ("bits", 4), ("R", 6)):
        if man.get(k) != want:
            fails.append(f"manifest.{k} = {man.get(k)!r}, want {want!r}")
    if sorted(man.get("datasets", [])) != sorted(DS) or sorted(man.get("methods", [])) != sorted(MS):
        fails.append("manifest datasets/methods wrong")
    fr, n = 49, 1000
    for ds in DS:
        first = {}
        for m in MS:
            try:
                f = man["traj"][ds][m]
                raw = open(os.path.join(DATA, f), "rb").read()
            except Exception as e:
                fails.append(f"traj {ds}/{m}: {e!r}")
                continue
            if len(raw) != fr * n * 2 * 4:
                fails.append(f"{f}: {len(raw)} bytes, want {fr * n * 8}")
                continue
            a = struct.unpack(f"<{fr * n * 2}f", raw)
            if not all(math.isfinite(q) and abs(q) <= 4 for q in a):
                fails.append(f"{f}: non-finite value or |x| > 4")
            first[m] = a[: n * 2]
        if len(first) == 2 and first["fm"] != first["hybrid"]:
            fails.append(f"{ds}: frame 0 differs between fm and hybrid")
        try:
            f = man["data"][ds]
            raw = open(os.path.join(DATA, f), "rb").read()
            if len(raw) != 1000 * 2 * 4:
                fails.append(f"{f}: {len(raw)} bytes, want 8000")
        except Exception as e:
            fails.append(f"data {ds}: {e!r}")
        q = load_json(os.path.join(DATA, f"quant_{ds}.json"), fails)
        if q:
            fails += quant_fails(ds, q)
    fx = load_json(os.path.join(DATA, "fixture.json"), fails)
    if fx:
        fails += fixture_fails(fx)
    rs = load_json(os.path.join(DATA, "results.json"), fails)
    if rs:
        try:
            assert "best_note" in rs and "cfg" in rs
            for ds in DS:
                for k in ("baseline", "hybrid"):
                    e = rs["datasets"][ds][k]
                    assert finite(e["mean"], e["std"]) and len(e["seeds"]) == 3 and finite(*e["seeds"])
            assert rs["log"] and all(finite(r[d]) for r in rs["log"] for d in DS)
            assert all("note" in r and "cfg" in r for r in rs["log"])
        except Exception as e:
            fails.append(f"results.json invalid: {e!r}")
    return fails


def quant_fails(ds, q):
    f = []
    tag = f"quant_{ds}.json"
    try:
        R, nn = q["R"], q["n"]
        if q["bits"] != 4 or R != 6 or nn != 512:
            f.append(f"{tag}: bits/R/n wrong")
        if len(q["gray"]) != 16 or len(q["centres"]) != 16:
            f.append(f"{tag}: gray/centres length")
        for k, s in (("xt", [nn, 2]), ("t", [nn]), ("v", [nn, 2]), ("bucket", [nn, 2]),
                     ("probs", [nn, 2, 16]), ("expect", None), ("offset", None), ("vhat", [nn, 2])):
            if s and shape(q[k]) != s:
                f.append(f"{tag}: {k} shape {shape(q[k])}, want {s}")
        if f:
            return f
        c = q["centres"]
        for i in range(nn):
            for j in range(2):
                p = q["probs"][i][j]
                if abs(sum(p) - 1) > 1e-4:
                    f.append(f"{tag}: probs[{i}][{j}] sums to {sum(p)}"); return f
                b = min(max(math.floor((q["v"][i][j] + R) / (2 * R) * 16), 0), 15)
                if q["bucket"][i][j] != b:
                    f.append(f"{tag}: bucket[{i}][{j}] = {q['bucket'][i][j]}, want {b}"); return f
                e = sum(a * b2 for a, b2 in zip(p, c))
                ex = q["expect"][i][j]
                if abs(ex - e) > 1e-4:
                    f.append(f"{tag}: expect[{i}][{j}] off by {abs(ex - e)}"); return f
                if abs(q["vhat"][i][j] - (ex + q["offset"][i][j])) > 1e-4:
                    f.append(f"{tag}: vhat != expect + offset at [{i}][{j}]"); return f
    except Exception as e:
        f.append(f"{tag}: {e!r}")
    return f


def fixture_fails(fx):
    f = []
    try:
        h, n = fx["hidden"], fx["n"]
        if (h, n) != (8, 16):
            f.append("fixture hidden/n wrong")
        for k, s in (("x", [n, 2]), ("t", [n]), ("v", [n, 2])):
            if shape(fx[k]) != s:
                f.append(f"fixture.{k} shape {shape(fx[k])}")
        for m, out in (("fm", 2), ("hybrid", 34)):
            d = fx[m]
            want = {"W1": [h, 3], "b1": [h], "W2": [h, h], "b2": [h], "W3": [out, h], "b3": [out], "vhat": [n, 2]}
            for k, s in want.items():
                if shape(d[k]) != s:
                    f.append(f"fixture.{m}.{k} shape {shape(d[k])}, want {s}")
            keys = ["loss"] + (["ce", "off"] if m == "hybrid" else [])
            if not finite(*[d[k] for k in keys]) or not all(finite(*flat([d[k]])) for k in want):
                f.append(f"fixture.{m}: non-finite")
    except Exception as e:
        f.append(f"fixture.json: {e!r}")
    return f


def node_check(path):
    r = subprocess.run(["node", "--check", path], capture_output=True, text=True)
    return None if r.returncode == 0 else r.stderr.strip()


def check_html():
    if not os.path.isfile(INDEX):
        return ["site/index.html missing"]
    fails = []
    try:
        p = parse_index()
    except Exception as e:
        return [f"index.html parse error: {e}"]
    for i in IDS:
        if i not in p.ids:
            fails.append(f"required id {i!r} missing")
    for i in sorted({i for i in p.ids if p.ids.count(i) > 1}):
        fails.append(f"duplicate id {i!r}")
    for t, u in rel_refs(p):
        if not os.path.isfile(os.path.join(SITE, urlsplit(u).path)):
            fails.append(f"<{t}> {u} does not resolve")
    if not shutil.which("node"):
        fails.append("node not found")
        return fails
    for s in p.scripts:
        if s.lower().startswith(SKIP):
            fails.append(f"external script {s}")
            continue
        d = os.path.join(SITE, urlsplit(s).path)
        if os.path.isfile(d) and (err := node_check(d)):
            fails.append(f"node --check {s}: {err}")
    for url, disk in worker_scripts():
        if not os.path.isfile(disk):
            fails.append(f"worker script {url} missing")
        elif err := node_check(disk):
            fails.append(f"node --check {url}: {err}")
    for fpath in [INDEX] + glob.glob(os.path.join(SITE, "**", "*.css"), recursive=True) + glob.glob(os.path.join(SITE, "js", "*.js")):
        if re.search(r"https?://", open(fpath, encoding="utf-8").read()):
            fails.append(f"{os.path.relpath(fpath, ROOT)}: contains an http(s):// URL")
    return fails


class Words(HTMLParser):
    def __init__(self):
        super().__init__()
        self.on = False
        self.words = 0
        self.skip = 0

    def handle_comment(self, data):
        d = data.strip()
        if d == "prose:methods":
            self.on = True
        elif d == "/prose:methods":
            self.on = False

    def handle_starttag(self, tag, attrs):
        self.skip += tag in ("script", "style")

    def handle_endtag(self, tag):
        self.skip -= tag in ("script", "style")

    def handle_data(self, data):
        if self.on and not self.skip:
            self.words += len(data.split())


def check_prose():
    if not os.path.isfile(INDEX):
        return ["site/index.html missing"]
    fails = []
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "ai_lint.py"), INDEX], capture_output=True, text=True)
    if r.returncode != 0:
        fails.append(f"ai_lint exit {r.returncode}: " + (r.stdout + r.stderr).strip().replace("\n", " | ")[:600])
    w = Words()
    w.feed(open(INDEX, encoding="utf-8").read())
    if not 350 <= w.words <= 700:
        fails.append(f"methods prose has {w.words} words, want 350 to 700")
    return fails


def up():
    try:
        urllib.request.urlopen(BASE, timeout=2).read()
        return True
    except Exception:
        return False


def ensure_server():
    if up():
        return None
    proc = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1", "--directory", SITE],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        if up():
            return proc
        time.sleep(0.2)
    proc.kill()
    return False


def check_serve():
    if not os.path.isfile(INDEX):
        return ["site/index.html missing"]
    srv = ensure_server()
    if srv is False:
        return [f"could not start server on {PORT}"]
    try:
        urls = [""] + sorted({urlsplit(u).path for _, u in rel_refs(parse_index())}) + [u for u, _ in worker_scripts()]
        fails = []
        for u in dict.fromkeys(urls):
            try:
                code = urllib.request.urlopen(BASE + u, timeout=5).status
            except Exception as e:
                code = getattr(e, "code", repr(e))
            if code != 200:
                fails.append(f"GET /{u} -> {code}")
        return fails
    finally:
        if srv:
            srv.kill()


def check_review():
    p = os.path.join(ROOT, ".pyramid", "review.json")
    fails = []
    r = load_json(p, fails)
    if r is None:
        return fails
    for k in ("blockers", "majors", "minors"):
        if not isinstance(r.get(k), list):
            fails.append(f"review.json: {k} is not a list")
    if r.get("blockers"):
        fails.append(f"review.json: {len(r['blockers'])} blocker(s)")
    return fails


BLANK_JS = """el => { const w = el.width, h = el.height; if (!w || !h) return false;
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const g = c.getContext('2d'); g.drawImage(el, 0, 0);
  const d = g.getImageData(0, 0, w, h).data; let n = 0;
  for (let i = 4; i < d.length; i += 4) if (d[i]!==d[0]||d[i+1]!==d[1]||d[i+2]!==d[2]||d[i+3]!==d[3]) n++;
  return n >= 0.005 * w * h; }"""
SNAP_JS = "el => el.toDataURL()"


def browser_section(page, sec, fails):
    def has(sel):
        return page.locator(sel).count() > 0

    def must(sel, label=None):
        if not has(sel):
            fails.append(f"{sec}: missing {label or sel}")
            return False
        return True

    def nonblank(sel, wait=0):
        end = time.time() + wait
        while True:
            ok = page.locator(sel).first.evaluate(BLANK_JS)
            if ok or time.time() >= end:
                return ok
            page.wait_for_timeout(500)

    if sec == "flow":
        if not must("select.flow-dataset"):
            return
        opts = page.locator("select.flow-dataset option").evaluate_all("o => o.map(x => x.value)")
        if not {"moons", "gaussians8"} <= set(opts):
            fails.append(f"flow: dataset options {opts}")
        for m in MS:
            if must(f"canvas.flow-canvas[data-method={m}]") and not nonblank(f"canvas.flow-canvas[data-method={m}]", 5):
                fails.append(f"flow: {m} canvas blank")
        if not must("input[type=range].t-slider"):
            return
        sl = page.locator("input[type=range].t-slider").first
        if sl.get_attribute("min") not in ("0", "0.0") or sl.get_attribute("max") not in ("1", "1.0"):
            fails.append("flow: slider min/max not 0 and 1")
        ro = page.locator(".t-readout").first
        if not re.fullmatch(r"t = [01]\.\d\d", ro.inner_text().strip()):
            fails.append(f"flow: readout {ro.inner_text()!r}")
        cv = page.locator("canvas.flow-canvas[data-method=hybrid]").first

        def at(v):
            sl.evaluate(f"el => {{ el.value = {v}; el.dispatchEvent(new Event('input', {{bubbles: true}})); }}")
            page.wait_for_timeout(400)
            return cv.evaluate(SNAP_JS)
        a = at(0)
        b = at(0.5)
        if ro.inner_text().strip() != "t = 0.50":
            fails.append(f"flow: readout after 0.5 is {ro.inner_text()!r}")
        c = at(1)
        if a == c:
            fails.append("flow: canvas identical at t=0 and t=1")
        at(0)
        if must("button.play[aria-pressed]"):
            before = sl.input_value()
            page.locator("button.play").first.click()
            page.wait_for_timeout(500)
            if page.locator("button.play").first.get_attribute("aria-pressed") != "true":
                fails.append("flow: play did not set aria-pressed true")
            if sl.input_value() == before:
                fails.append("flow: slider did not move within 500 ms of play")
            page.locator("button.play").first.click()
    elif sec == "quant":
        must("select.quant-dataset")
        if must("canvas.quant-canvas") and not nonblank("canvas.quant-canvas", 5):
            fails.append("quant: canvas blank")
        for c in (0, 1):
            sel = f".bin-strip[data-component='{c}']"
            if not must(sel):
                continue
            labs = page.locator(sel + " .bin-label").all_inner_texts()
            labs = [x.strip() for x in labs]
            if len(labs) != 16 or not all(re.fullmatch(r"[01]{4}", x) for x in labs):
                fails.append(f"quant: strip {c} labels {labs}")
            elif any(sum(p != q for p, q in zip(a, b)) != 1 for a, b in zip(labs, labs[1:])):
                fails.append(f"quant: strip {c} labels are not Gray-coded")
        if must("button.next-particle") and must(".offset-readout"):
            ro = page.locator(".offset-readout").first
            seen = {ro.inner_text()}
            for _ in range(5):
                page.locator("button.next-particle").first.click()
                page.wait_for_timeout(100)
                seen.add(ro.inner_text())
            if len(seen) < 2:
                fails.append("quant: next-particle did not change .offset-readout")
    elif sec == "train":
        must("select.train-dataset")
        ok = all([must(".train-start"), must(".train-pause"), must(".train-reset"), must(".step-readout"),
                  must("canvas.train-dist[data-method=fm]"), must("canvas.train-dist[data-method=hybrid]"),
                  must("canvas.train-metric")])
        if not ok:
            return
        ro = page.locator(".step-readout").first
        s0 = int(re.sub(r"\D", "", ro.inner_text()) or 0)
        page.locator(".train-start").first.click()
        end = time.time() + 8
        grew = False
        while time.time() < end and not grew:
            page.wait_for_timeout(300)
            m = re.search(r"\d+", ro.inner_text().replace(",", ""))
            grew = bool(m) and int(m.group()) > s0
        if not grew:
            fails.append("train: step readout did not increase within 8 s")
        if not nonblank("canvas.train-metric", 25):
            fails.append("train: metric canvas blank after 25 s")
        page.locator(".train-pause").first.click()


def check_browser(section=None):
    if not os.path.isfile(INDEX):
        return ["site/index.html missing"]
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        return [f"playwright unavailable: {e}"]
    srv = ensure_server()
    if srv is False:
        return [f"could not start server on {PORT}"]
    fails = []
    try:
        with sync_playwright() as pw:
            br = pw.chromium.launch(headless=True)
            page = br.new_page(viewport={"width": 1280, "height": 900})
            errs = []
            page.on("console", lambda m: errs.append(f"console.error: {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
            page.goto(BASE, wait_until="load")
            page.wait_for_timeout(500)
            for sec in ("flow", "quant", "train"):
                if section and section != sec:
                    continue
                if page.locator(f"#{sec} *").count() == 0:
                    continue
                try:
                    browser_section(page, sec, fails)
                except Exception as e:
                    fails.append(f"{sec}: {type(e).__name__}: {str(e).splitlines()[0]}")
            fails += errs
            br.close()
    except Exception as e:
        fails.append(f"browser: {type(e).__name__}: {str(e).splitlines()[0]}")
    finally:
        if srv:
            srv.kill()
    return fails


def run(name, fn, *a):
    try:
        fails = fn(*a)
    except Exception as e:
        fails = [f"{type(e).__name__}: {e}"]
    for f in fails:
        print(f"FAIL [{name}]: {f}")
    if not fails:
        print(f"OK [{name}]")
    return not fails


def main(argv):
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    cmds = ["data", "html", "prose", "serve", "browser", "review", "all"]
    if not argv or argv[0] not in cmds:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    section = None
    if rest:
        if cmd != "browser" or len(rest) != 2 or rest[0] != "--section" or rest[1] not in ("flow", "quant", "train"):
            print(__doc__.strip(), file=sys.stderr)
            return 2
        section = rest[1]
    fns = {"data": check_data, "html": check_html, "prose": check_prose, "serve": check_serve, "review": check_review}
    if cmd == "browser":
        return 0 if run("browser", check_browser, section) else 1
    if cmd == "all":
        ok = True
        for name in ("data", "html", "prose", "serve"):
            ok = run(name, fns[name]) and ok
        return 0 if run("browser", check_browser, None) and ok else 1
    return 0 if run(cmd, fns[cmd]) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
