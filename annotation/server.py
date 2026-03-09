#!/usr/bin/env python3
"""Annotation server for human-LLM agreement on inoculation detection.

Usage: python annotation/server.py [--port 8080] [--data annotation/completions.jsonl]
"""

import json
import os
import sys
import re
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

DATA_FILE = os.path.join(os.path.dirname(__file__), "completions.jsonl")
ANNOTATIONS_FILE = os.path.join(os.path.dirname(__file__), "annotations.jsonl")

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inoculation Annotation</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e0e0e0; padding: 20px; }
.container { max-width: 900px; margin: 0 auto; }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.progress-bar { width: 100%; height: 8px; background: #2a2d35; border-radius: 4px; overflow: hidden; margin-bottom: 8px; }
.progress-fill { height: 100%; background: #4f8cff; transition: width 0.3s; }
.progress-text { font-size: 13px; color: #888; margin-bottom: 16px; }
.meta { font-size: 12px; color: #666; margin-bottom: 12px; }
.meta span { margin-right: 16px; }
.prompt-box { background: #1a1d27; border: 1px solid #2a2d35; border-radius: 8px; padding: 16px; margin-bottom: 12px; white-space: pre-wrap; word-wrap: break-word; font-size: 14px; line-height: 1.5; max-height: 200px; overflow-y: auto; }
.think-box { background: #2a2010; border: 1px solid #4a3a20; border-radius: 8px; padding: 16px; margin-bottom: 12px; white-space: pre-wrap; word-wrap: break-word; font-size: 13px; line-height: 1.5; max-height: 400px; overflow-y: auto; color: #d4a854; }
.think-label { font-size: 11px; font-weight: 600; color: #b08830; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
.response-box { background: #1a1d27; border: 1px solid #2a2d35; border-radius: 8px; padding: 16px; margin-bottom: 20px; white-space: pre-wrap; word-wrap: break-word; font-size: 14px; line-height: 1.5; max-height: 400px; overflow-y: auto; }
.response-label { font-size: 11px; font-weight: 600; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
.buttons { display: flex; gap: 12px; margin-bottom: 16px; }
.btn { flex: 1; padding: 14px; border: 2px solid #333; border-radius: 8px; background: #1a1d27; color: #e0e0e0; font-size: 15px; font-weight: 600; cursor: pointer; text-align: center; transition: all 0.15s; }
.btn:hover { border-color: #4f8cff; background: #1e2433; }
.btn-inoc { border-color: #c0392b; }
.btn-inoc:hover, .btn-inoc.active { background: #3a1515; border-color: #e74c3c; color: #e74c3c; }
.btn-not { border-color: #27ae60; }
.btn-not:hover, .btn-not.active { background: #152a1a; border-color: #2ecc71; color: #2ecc71; }
.btn-skip { border-color: #666; }
.btn-skip:hover, .btn-skip.active { background: #2a2a2a; border-color: #999; color: #999; }
.kbd { display: inline-block; background: #2a2d35; border-radius: 3px; padding: 1px 6px; font-size: 12px; font-family: monospace; margin-left: 6px; }
.llm-toggle { font-size: 13px; color: #666; cursor: pointer; user-select: none; margin-bottom: 12px; }
.llm-toggle:hover { color: #888; }
.llm-info { background: #1a1d27; border: 1px solid #2a2d35; border-radius: 6px; padding: 10px 14px; font-size: 13px; margin-bottom: 12px; display: none; }
.llm-info.visible { display: block; }
.done { text-align: center; padding: 60px 20px; }
.done h2 { color: #4f8cff; margin-bottom: 12px; }
.loading { text-align: center; padding: 40px; color: #666; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h2>Inoculation Annotation</h2>
    <span class="progress-text" id="progress-text">Loading...</span>
  </div>
  <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
  <div id="content"><div class="loading">Loading...</div></div>
</div>
<script>
let current = null;
let llmVisible = false;

async function loadNext() {
  const res = await fetch('/api/next');
  const data = await res.json();
  updateProgress(data);
  if (data.done) {
    document.getElementById('content').innerHTML =
      '<div class="done"><h2>All done!</h2><p>Run <code>python annotation/analyze.py</code> to see results.</p></div>';
    return;
  }
  current = data;
  render(data);
}

function updateProgress(data) {
  const pct = data.total > 0 ? (data.annotated / data.total * 100) : 0;
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-text').textContent =
    `${data.annotated} / ${data.total} annotated (${pct.toFixed(1)}%)`;
}

function splitThinkResponse(completion) {
  // Split on </think> tag
  const m = completion.match(/^<think>([\s\S]*?)<\/think>\s*([\s\S]*)$/);
  if (m) return { think: m[1].trim(), response: m[2].trim() };
  // If starts with <think> but no closing tag, it's all thinking
  if (completion.startsWith('<think>')) return { think: completion.slice(7).trim(), response: '' };
  return { think: '', response: completion };
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function render(data) {
  const { think, response } = splitThinkResponse(data.completion);
  const llmLabel = data.llm_label > 0.5 ? 'INOCULATING' : 'NOT INOCULATING';
  const llmColor = data.llm_label > 0.5 ? '#e74c3c' : '#2ecc71';

  let html = `
    <div class="meta">
      <span>Step: <b>${data.gen_step}</b></span>
      <span>ID: ${data.id}</span>
      <span>Advantage: ${data.advantages?.toFixed(3) ?? 'N/A'}</span>
    </div>
    <div class="prompt-box">${escapeHtml(data.prompt)}</div>`;

  if (think) {
    html += `<div class="think-box"><div class="think-label">Thinking</div>${escapeHtml(think)}</div>`;
  }
  if (response) {
    html += `<div class="response-box"><div class="response-label">Response</div>${escapeHtml(response)}</div>`;
  }

  html += `
    <div class="buttons">
      <button class="btn btn-inoc" onclick="annotate('inoculating')">
        Inoculating <span class="kbd">1</span>
      </button>
      <button class="btn btn-not" onclick="annotate('not_inoculating')">
        Not Inoculating <span class="kbd">2</span>
      </button>
      <button class="btn btn-skip" onclick="annotate('skip')">
        Skip <span class="kbd">3</span>
      </button>
    </div>
    <div class="llm-toggle" onclick="toggleLLM()">
      &#9654; Show LLM label (click to reveal)
    </div>
    <div class="llm-info" id="llm-info">
      LLM verdict: <b style="color:${llmColor}">${llmLabel}</b>
      (score: ${data.llm_label?.toFixed(2)})
      &nbsp;|&nbsp; Judge reward: ${data.llm_judge_reward?.toFixed(2)}
      &nbsp;|&nbsp; Eval awareness: ${data.eval_awareness?.toFixed(2)}
    </div>`;

  document.getElementById('content').innerHTML = html;
  llmVisible = false;
}

function toggleLLM() {
  llmVisible = !llmVisible;
  const el = document.getElementById('llm-info');
  el.classList.toggle('visible', llmVisible);
  el.previousElementSibling.innerHTML = (llmVisible ? '&#9660;' : '&#9654;') + ' Show LLM label';
}

async function annotate(label) {
  if (!current) return;
  await fetch('/api/annotate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ id: current.id, label })
  });
  loadNext();
}

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === '1') annotate('inoculating');
  else if (e.key === '2') annotate('not_inoculating');
  else if (e.key === '3') annotate('skip');
});

loadNext();
</script>
</body>
</html>
"""


class AnnotationHandler(BaseHTTPRequestHandler):
    completions = []
    annotated_ids = set()

    @classmethod
    def load_data(cls, data_file, annotations_file):
        cls.data_file = data_file
        cls.annotations_file = annotations_file
        # Load completions
        with open(data_file) as f:
            cls.completions = [json.loads(line) for line in f if line.strip()]
        # Load existing annotations (resume support)
        cls.annotated_ids = set()
        if os.path.exists(annotations_file):
            with open(annotations_file) as f:
                for line in f:
                    if line.strip():
                        ann = json.loads(line)
                        cls.annotated_ids.add(ann["id"])
        print(f"Loaded {len(cls.completions)} completions, {len(cls.annotated_ids)} already annotated")

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._respond(200, "text/html", HTML_PAGE.encode())
        elif parsed.path == "/api/next":
            self._serve_next()
        elif parsed.path == "/api/progress":
            self._serve_progress()
        else:
            self._respond(404, "application/json", b'{"error":"not found"}')

    def do_POST(self):
        if self.path == "/api/annotate":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            self._save_annotation(body)
            self._respond(200, "application/json", b'{"ok":true}')
        else:
            self._respond(404, "application/json", b'{"error":"not found"}')

    def _serve_next(self):
        for item in self.completions:
            if item["id"] not in self.annotated_ids:
                resp = {**item, "annotated": len(self.annotated_ids), "total": len(self.completions), "done": False}
                self._respond(200, "application/json", json.dumps(resp).encode())
                return
        resp = {"done": True, "annotated": len(self.annotated_ids), "total": len(self.completions)}
        self._respond(200, "application/json", json.dumps(resp).encode())

    def _serve_progress(self):
        resp = {"annotated": len(self.annotated_ids), "total": len(self.completions)}
        self._respond(200, "application/json", json.dumps(resp).encode())

    def _save_annotation(self, body):
        ann = {
            "id": body["id"],
            "label": body["label"],
        }
        # Find the original item to include LLM label for analysis
        for item in self.completions:
            if item["id"] == body["id"]:
                ann["llm_label"] = item.get("llm_label", 0)
                ann["gen_step"] = item.get("gen_step", 0)
                break
        with open(self.annotations_file, "a") as f:
            f.write(json.dumps(ann) + "\n")
        self.annotated_ids.add(body["id"])

    def _respond(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--data", default=DATA_FILE)
    args = parser.parse_args()

    annotations_file = os.path.join(os.path.dirname(args.data), "annotations.jsonl")
    AnnotationHandler.load_data(args.data, annotations_file)

    # Bind to all interfaces, display the public IP
    public_ip = "87.120.211.197"
    server = HTTPServer(("0.0.0.0", args.port), AnnotationHandler)
    print(f"Annotation server running at http://{public_ip}:{args.port}")
    print("Keyboard shortcuts: 1=Inoculating, 2=Not Inoculating, 3=Skip")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nStopped. {len(AnnotationHandler.annotated_ids)} annotations saved.")


if __name__ == "__main__":
    main()
