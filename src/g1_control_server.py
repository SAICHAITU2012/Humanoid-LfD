from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "outputs/g1_control_spec.json"
STATE_PATH = ROOT / "outputs/g1_control_state.json"
HAND_CLOSED = {"right_hand_thumb_0_joint": -0.55, "right_hand_thumb_1_joint": -0.35,
               "right_hand_thumb_2_joint": -1.10, "right_hand_index_0_joint": 1.20,
               "right_hand_index_1_joint": 1.35, "right_hand_middle_0_joint": 1.20,
               "right_hand_middle_1_joint": 1.35}

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>G1 joint controls</title>
<style>body{font-family:-apple-system,sans-serif;background:#14181f;color:#e8eaed;max-width:430px;margin:24px auto;padding:0 18px}
h1{font-size:18px}h2{font-size:13px;color:#9aa0a6;margin-top:22px}
.row{margin:9px 0}label{display:flex;justify-content:space-between;font-size:12px;color:#bdc1c6}
input[type=range]{width:100%;accent-color:#4c8bf5}
.btns{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}
button{background:#2b6cb0;color:#fff;border:0;border-radius:6px;padding:9px 14px;font-size:14px;cursor:pointer}
button.alt{background:#333a45}</style></head><body>
<h1>Unitree G1 — live joint controls</h1>
<div class="btns">
<button onclick="cmd('open_hand')">Open hand</button>
<button onclick="cmd('close_hand')">Close hand</button>
<button onclick="cmd('reset')" class="alt">Reset stand pose</button>
</div>
<h2>Right arm</h2><div id="arm"></div>
<h2>Right hand</h2><div id="hand"></div>
<script>
const ACT = __ACTUATORS__;
function slider(a, host) {
  const row = document.createElement('div'); row.className = 'row';
  row.innerHTML = `<label><span>${a.name}</span><span id="v_${a.name}">${a.value.toFixed(2)}</span></label>` +
    `<input type="range" min="${a.lo}" max="${a.hi}" step="0.01" value="${a.value}">`;
  row.querySelector('input').addEventListener('input', (ev) => {
    document.getElementById('v_' + a.name).textContent = (+ev.target.value).toFixed(2);
    fetch('/set?j=' + a.name + '&v=' + ev.target.value);
  });
  document.getElementById(host).appendChild(row);
}
ACT.forEach(a => slider(a, a.hand ? 'hand' : 'arm'));
function cmd(n) { fetch('/cmd?name=' + n).then(() => refresh()); }
function refresh() {
  fetch('/sync').then(r => r.json()).then(vals => {
    ACT.forEach(a => {
      const inp = document.querySelectorAll('#arm input,#hand input')[ACT.indexOf(a)];
      if (inp && a.name in vals) {
        inp.value = vals[a.name];
        document.getElementById('v_' + a.name).textContent = vals[a.name].toFixed(2);
      }
    });
  });
}
refresh();
</script></body></html>
"""


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


class ControlState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.spec = read_json(SPEC_PATH, {"actuators": []})
        self.state = read_json(STATE_PATH, {"targets": {}, "reset": False, "seq": 0})

    def save(self) -> None:
        write_json(STATE_PATH, self.state)

    def set_joint(self, name: str, value: float) -> bool:
        limits = {a["name"]: (a["lo"], a["hi"]) for a in self.spec["actuators"]}
        if name not in limits:
            return False
        lo, hi = limits[name]
        with self.lock:
            self.state["targets"][name] = min(hi, max(lo, value))
            self.state["seq"] += 1
            self.save()
        return True

    def command(self, name: str) -> bool:
        with self.lock:
            if name == "reset":
                self.state["reset"] = True
            elif name == "open_hand":
                for joint in HAND_CLOSED:
                    self.state["targets"][joint] = 0.0
            elif name == "close_hand":
                self.state["targets"].update(HAND_CLOSED)
            else:
                return False
            self.state["seq"] += 1
            self.save()
        return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Browser control page for the live G1 viewer (writes commands to outputs/g1_control_state.json)."
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not SPEC_PATH.exists():
        raise SystemExit(f"Missing {SPEC_PATH.name}; start src/view_g1_web.py first (it publishes the actuator spec).")

    control = ControlState()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str = "text/html; charset=utf-8") -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                page = PAGE.replace("__ACTUATORS__", json.dumps(control.spec["actuators"]))
                self._send(page.encode())
            elif parsed.path == "/set":
                try:
                    value = float(query.get("v", ["0"])[0])
                except ValueError:
                    self._send(b"bad value", "text/plain")
                    return
                ok = control.set_joint(query.get("j", [""])[0], value)
                self._send(b"ok" if ok else b"unknown joint", "text/plain")
            elif parsed.path == "/cmd":
                ok = control.command(query.get("name", [""])[0])
                self._send(b"ok" if ok else b"unknown command", "text/plain")
            elif parsed.path == "/sync":
                with control.lock:
                    body = json.dumps(control.state["targets"]).encode()
                self._send(body, "application/json")
            else:
                self.send_error(404)

        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Controls page: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
