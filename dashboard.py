"""DogVision Dashboard — Web UI for viewing results and processing new videos.

Features:
  - Displays all evaluation run summaries from out/*/summary.json
  - Shows event logs (bite alerts + access violations) per run
  - Upload a video → runs pipeline → returns annotated video + logs
  - Serves output videos for playback in browser (auto-transcodes to H.264)

Usage:
    python dashboard.py
    python dashboard.py --port 8080
    Open http://localhost:5000 in your browser
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import (
    Flask, render_template_string, request, jsonify,
    send_from_directory, redirect, url_for, Response,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max upload

OUT_DIR = Path("out")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Track processing jobs
jobs: dict[str, dict] = {}


def _transcode_h264(src: Path) -> Path:
    """Transcode mp4v video to H.264 for browser playback. Caches result."""
    h264_path = src.with_suffix(".h264.mp4")
    if h264_path.exists():
        return h264_path
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264", "-preset", "fast",
             "-crf", "23", "-movflags", "+faststart", "-an", str(h264_path)],
            capture_output=True, timeout=300,
        )
        if h264_path.exists():
            return h264_path
    except Exception:
        pass
    return src


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DogVision Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg-primary:#080b12;--bg-secondary:#0f1219;--bg-card:#141820;--bg-elevated:#1a1f2b;
  --bg-hover:#1e2433;--bg-input:#111621;
  --border:#1e2535;--border-hover:#2a3548;
  --text-primary:#e8eaf0;--text-secondary:#8892a4;--text-muted:#5a6478;
  --accent:#3b82f6;--accent-hover:#2563eb;--accent-glow:#3b82f620;
  --green:#10b981;--green-bg:#10b98115;--green-border:#10b98130;
  --red:#ef4444;--red-bg:#ef444415;--red-border:#ef444430;
  --orange:#f59e0b;--orange-bg:#f59e0b15;--orange-border:#f59e0b30;
  --cyan:#06b6d4;--cyan-bg:#06b6d415;
  --radius:8px;--radius-lg:12px;--radius-xl:16px;
  --shadow:0 1px 3px rgba(0,0,0,.3),0 4px 12px rgba(0,0,0,.2);
  --shadow-lg:0 4px 16px rgba(0,0,0,.4),0 8px 32px rgba(0,0,0,.3);
  --transition:150ms ease;
}
body{
  font-family:'Inter',system-ui,-apple-system,sans-serif;
  background:var(--bg-primary);color:var(--text-primary);
  min-height:100vh;line-height:1.5;-webkit-font-smoothing:antialiased;
}

/* Header */
.header{
  background:var(--bg-secondary);border-bottom:1px solid var(--border);
  padding:0 32px;height:64px;display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:100;backdrop-filter:blur(12px);
}
.logo{display:flex;align-items:center;gap:12px}
.logo-icon{
  width:36px;height:36px;border-radius:8px;
  background:linear-gradient(135deg,var(--accent),#8b5cf6);
  display:flex;align-items:center;justify-content:center;
  font-weight:800;font-size:16px;color:white;
}
.logo-text{font-size:18px;font-weight:700;letter-spacing:-0.3px}
.logo-text span{color:var(--text-secondary);font-weight:400;font-size:13px;margin-left:8px}

.nav{display:flex;gap:2px;background:var(--bg-primary);border-radius:8px;padding:3px}
.nav button{
  background:transparent;color:var(--text-secondary);border:none;
  padding:7px 18px;border-radius:6px;cursor:pointer;font-size:13px;
  font-weight:500;transition:var(--transition);font-family:inherit;
}
.nav button:hover{color:var(--text-primary);background:var(--bg-hover)}
.nav button.active{background:var(--accent);color:white}

/* Container */
.container{max-width:1360px;margin:0 auto;padding:28px 32px}

/* Stats */
.stats-row{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:28px}
.stat-card{
  background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-lg);
  padding:20px;text-align:center;transition:var(--transition);
}
.stat-card:hover{border-color:var(--border-hover);transform:translateY(-1px)}
.stat-value{font-size:32px;font-weight:800;letter-spacing:-1px;font-feature-settings:'tnum'}
.stat-value.blue{color:var(--accent)}.stat-value.green{color:var(--green)}
.stat-value.cyan{color:var(--cyan)}.stat-value.red{color:var(--red)}
.stat-value.orange{color:var(--orange)}
.stat-label{font-size:11px;color:var(--text-muted);margin-top:4px;text-transform:uppercase;letter-spacing:1.2px;font-weight:600}

/* Section header */
.section-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
.section-title{font-size:16px;font-weight:600;color:var(--text-primary)}
.section-count{font-size:12px;color:var(--text-muted);background:var(--bg-elevated);padding:3px 10px;border-radius:20px}

/* Run cards */
.runs-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:16px;margin-bottom:32px}
.run-card{
  background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-lg);
  overflow:hidden;transition:var(--transition);
}
.run-card:hover{border-color:var(--border-hover);box-shadow:var(--shadow)}
.card-head{
  padding:14px 18px;border-bottom:1px solid var(--border);
  display:flex;justify-content:space-between;align-items:center;
}
.card-name{font-size:14px;font-weight:600;color:var(--text-primary)}
.badges{display:flex;gap:6px;flex-wrap:wrap}
.badge{
  padding:2px 8px;border-radius:20px;font-size:11px;font-weight:600;
  letter-spacing:0.3px;
}
.badge-bite{background:var(--red-bg);color:var(--red);border:1px solid var(--red-border)}
.badge-clean{background:var(--green-bg);color:var(--green);border:1px solid var(--green-border)}
.badge-access{background:var(--orange-bg);color:var(--orange);border:1px solid var(--orange-border)}

.card-body{padding:14px 18px}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.metric{
  text-align:center;padding:10px 6px;background:var(--bg-primary);
  border-radius:var(--radius);border:1px solid transparent;
}
.metric-val{font-size:18px;font-weight:700;font-feature-settings:'tnum'}
.metric-lbl{font-size:10px;color:var(--text-muted);margin-top:2px;text-transform:uppercase;letter-spacing:0.8px;font-weight:500}
.metric-val.g{color:var(--green)}.metric-val.c{color:var(--cyan)}
.metric-val.r{color:var(--red)}.metric-val.o{color:var(--orange)}
.metric-val.m{color:var(--text-secondary)}

.card-foot{
  padding:10px 18px;border-top:1px solid var(--border);
  display:flex;gap:6px;
}
.btn{
  padding:6px 14px;border-radius:6px;border:1px solid var(--border);
  background:var(--bg-elevated);color:var(--text-secondary);font-size:12px;
  cursor:pointer;transition:var(--transition);font-family:inherit;font-weight:500;
  text-decoration:none;display:inline-flex;align-items:center;gap:4px;
}
.btn:hover{background:var(--bg-hover);color:var(--text-primary);border-color:var(--border-hover)}
.btn-primary{background:var(--accent);border-color:var(--accent);color:white}
.btn-primary:hover{background:var(--accent-hover)}
.btn svg{width:14px;height:14px}

/* Upload */
.upload-zone{
  background:var(--bg-card);border:2px dashed var(--border);border-radius:var(--radius-xl);
  padding:56px 40px;text-align:center;transition:var(--transition);cursor:pointer;
  margin-bottom:24px;
}
.upload-zone:hover,.upload-zone.dragover{border-color:var(--accent);background:var(--accent-glow)}
.upload-icon{
  width:56px;height:56px;border-radius:14px;
  background:linear-gradient(135deg,var(--accent),#8b5cf6);
  display:flex;align-items:center;justify-content:center;margin:0 auto 16px;
}
.upload-icon svg{width:28px;height:28px;color:white}
.upload-zone h2{font-size:18px;font-weight:600;margin-bottom:6px}
.upload-zone p{color:var(--text-secondary);font-size:14px;margin-bottom:20px}
.upload-zone input[type="file"]{display:none}
.upload-btn{
  display:inline-block;padding:10px 28px;background:var(--accent);color:white;
  border-radius:8px;cursor:pointer;font-size:14px;font-weight:600;
  transition:var(--transition);border:none;
}
.upload-btn:hover{background:var(--accent-hover)}
.upload-opts{display:flex;gap:16px;justify-content:center;margin-top:18px;flex-wrap:wrap}
.upload-opts label{
  display:flex;align-items:center;gap:6px;font-size:13px;color:var(--text-secondary);
  cursor:pointer;padding:6px 14px;background:var(--bg-elevated);border-radius:6px;
  border:1px solid var(--border);transition:var(--transition);
}
.upload-opts label:hover{border-color:var(--border-hover)}
.upload-opts input[type="checkbox"]{accent-color:var(--accent)}

/* Progress */
.progress-wrap{display:none;margin-top:20px;background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-lg);padding:20px 24px}
.progress-header{display:flex;justify-content:space-between;margin-bottom:10px}
.progress-title{font-weight:600;font-size:14px}.progress-pct{color:var(--text-muted);font-size:13px;font-feature-settings:'tnum'}
.progress-bar{height:4px;background:var(--bg-primary);border-radius:2px;overflow:hidden}
.progress-fill{height:100%;background:linear-gradient(90deg,var(--accent),#8b5cf6);border-radius:2px;transition:width .4s ease;width:0}
.progress-msg{margin-top:8px;font-size:12px;color:var(--text-muted);font-family:'JetBrains Mono',monospace}

/* Events */
.events-controls{display:flex;gap:12px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
.select-run{
  background:var(--bg-input);color:var(--text-primary);border:1px solid var(--border);
  padding:8px 14px;border-radius:var(--radius);font-size:13px;font-family:inherit;
  min-width:200px;cursor:pointer;
}
.select-run:focus{border-color:var(--accent);outline:none;box-shadow:0 0 0 3px var(--accent-glow)}
.event-filters{display:flex;gap:6px}
.filter-btn{
  padding:5px 12px;border-radius:20px;border:1px solid var(--border);
  background:var(--bg-elevated);color:var(--text-secondary);font-size:12px;
  cursor:pointer;transition:var(--transition);font-family:inherit;font-weight:500;
}
.filter-btn:hover,.filter-btn.active{background:var(--accent);border-color:var(--accent);color:white}

table.events{
  width:100%;border-collapse:separate;border-spacing:0;
  background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-lg);overflow:hidden;
}
table.events th{
  background:var(--bg-elevated);padding:10px 14px;text-align:left;
  font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--text-muted);
  font-weight:600;border-bottom:1px solid var(--border);position:sticky;top:0;
}
table.events td{
  padding:8px 14px;border-bottom:1px solid var(--border);font-size:13px;
  font-feature-settings:'tnum';
}
table.events tr:last-child td{border-bottom:none}
table.events tr:hover td{background:var(--bg-hover)}
.ev-bite{color:var(--red);font-weight:600}
.ev-access{color:var(--orange);font-weight:600}
.ev-score{
  display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;
  font-weight:600;font-family:'JetBrains Mono',monospace;
}
.ev-score.high{background:var(--red-bg);color:var(--red)}
.ev-score.med{background:var(--orange-bg);color:var(--orange)}
.ev-score.low{background:var(--green-bg);color:var(--green)}

/* Video modal */
.modal-overlay{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:1000;
  justify-content:center;align-items:center;backdrop-filter:blur(4px);
}
.modal-overlay.show{display:flex}
.modal-content{position:relative;max-width:90vw;max-height:90vh}
.modal-content video{max-width:100%;max-height:85vh;border-radius:var(--radius-lg);box-shadow:var(--shadow-lg)}
.modal-close{
  position:absolute;top:-12px;right:-12px;width:32px;height:32px;
  border-radius:50%;background:var(--bg-card);border:1px solid var(--border);
  color:var(--text-primary);font-size:18px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;transition:var(--transition);
}
.modal-close:hover{background:var(--red);border-color:var(--red);color:white}

/* JSON modal */
.json-overlay{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:1000;
  justify-content:center;align-items:center;backdrop-filter:blur(4px);
}
.json-overlay.show{display:flex}
.json-panel{
  background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-xl);
  width:90vw;max-width:640px;max-height:80vh;overflow:hidden;display:flex;flex-direction:column;
}
.json-header{
  padding:16px 20px;border-bottom:1px solid var(--border);
  display:flex;justify-content:space-between;align-items:center;
}
.json-header h3{font-size:15px;font-weight:600}
.json-body{
  padding:16px 20px;overflow:auto;flex:1;
  font-family:'JetBrains Mono',monospace;font-size:12px;line-height:1.6;
  color:var(--text-secondary);white-space:pre-wrap;word-break:break-all;
}

/* Result card */
.result-card{
  background:var(--bg-card);border:1px solid var(--green-border);
  border-radius:var(--radius-xl);padding:24px;margin-top:20px;
}
.result-card h3{font-size:16px;font-weight:700;color:var(--green);margin-bottom:16px;display:flex;align-items:center;gap:8px}

/* Tab content */
.tab{display:none}.tab.active{display:block}

/* Empty state */
.empty{text-align:center;padding:48px;color:var(--text-muted);font-size:14px}

/* Responsive */
@media(max-width:768px){
  .stats-row{grid-template-columns:repeat(3,1fr)}
  .runs-grid{grid-template-columns:1fr}
  .header{padding:0 16px}
  .container{padding:20px 16px}
  .nav button{padding:6px 12px;font-size:12px}
}
</style>
</head>
<body>

<div class="header">
  <div class="logo">
    <div class="logo-icon">DV</div>
    <div class="logo-text">DogVision<span>Dashboard</span></div>
  </div>
  <div class="nav" id="nav">
    <button class="active" data-tab="overview">Overview</button>
    <button data-tab="upload">Upload & Analyze</button>
    <button data-tab="events">Event Logs</button>
  </div>
</div>

<div class="container">
  <!-- OVERVIEW -->
  <div id="tab-overview" class="tab active">
    <div class="stats-row" id="global-stats"></div>
    <div class="section-header">
      <div class="section-title">Evaluation Runs</div>
      <div class="section-count" id="run-count">0 runs</div>
    </div>
    <div class="runs-grid" id="runs-grid"></div>
  </div>

  <!-- UPLOAD -->
  <div id="tab-upload" class="tab">
    <div class="upload-zone" id="upload-zone">
      <div class="upload-icon">
        <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
          <path stroke-linecap="round" stroke-linejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
        </svg>
      </div>
      <h2>Upload Video for Analysis</h2>
      <p>Drag & drop a video file or click to browse. Supports .mp4, .mkv, .avi</p>
      <input type="file" id="video-input" accept="video/*">
      <label class="upload-btn" for="video-input">Choose Video File</label>
      <div class="upload-opts">
        <label><input type="checkbox" id="opt-gpu"> GPU Mode (TensorRT FP16)</label>
        <label><input type="checkbox" id="opt-restricted" checked> Restricted Access</label>
        <label><input type="checkbox" id="opt-nodisplay" checked> Headless</label>
      </div>
      <div id="gpu-status" style="margin-top:10px;font-size:12px;color:var(--text-muted)"></div>
    </div>
    <div class="progress-wrap" id="progress-wrap">
      <div class="progress-header">
        <span class="progress-title" id="p-title">Processing...</span>
        <span class="progress-pct" id="p-pct">0%</span>
      </div>
      <div class="progress-bar"><div class="progress-fill" id="p-fill"></div></div>
      <div class="progress-msg" id="p-msg">Uploading video...</div>
    </div>
    <div id="upload-result"></div>
  </div>

  <!-- EVENTS -->
  <div id="tab-events" class="tab">
    <div class="events-controls">
      <select class="select-run" id="event-select">
        <option value="">Select a run...</option>
      </select>
      <div class="event-filters">
        <button class="filter-btn active" data-filter="all">All</button>
        <button class="filter-btn" data-filter="bite_risk">Bite Alerts</button>
        <button class="filter-btn" data-filter="access_violation">Access Violations</button>
      </div>
    </div>
    <div id="events-container"><div class="empty">Select a run to view event logs.</div></div>
  </div>
</div>

<!-- Video Modal -->
<div class="modal-overlay" id="video-modal">
  <div class="modal-content">
    <button class="modal-close" onclick="closeVideo()">&#215;</button>
    <video controls id="modal-video"></video>
  </div>
</div>

<!-- JSON Modal -->
<div class="json-overlay" id="json-modal">
  <div class="json-panel">
    <div class="json-header">
      <h3 id="json-title">Summary JSON</h3>
      <button class="modal-close" onclick="closeJson()" style="position:static">&#215;</button>
    </div>
    <div class="json-body" id="json-body"></div>
  </div>
</div>

<script>
let runsData=[];
let currentFilter='all';

// Tab switching
document.getElementById('nav').addEventListener('click',e=>{
  const btn=e.target.closest('button');
  if(!btn)return;
  const tab=btn.dataset.tab;
  document.querySelectorAll('.nav button').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('tab-'+tab).classList.add('active');
});

function switchToTab(tab){
  document.querySelectorAll('.nav button').forEach(b=>{
    b.classList.toggle('active',b.dataset.tab===tab);
  });
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('tab-'+tab).classList.add('active');
}

// Check GPU
async function checkGpu(){
  const r=await fetch('/api/gpu-status');
  const d=await r.json();
  const el=document.getElementById('gpu-status');
  if(d.available){
    el.innerHTML='<span style="color:var(--green)">&#9679;</span> GPU: '+d.name+' (CUDA '+d.cuda+')';
    document.getElementById('opt-gpu').checked=true;
  }else{
    el.innerHTML='<span style="color:var(--text-muted)">&#9679;</span> No GPU detected — CPU mode only';
    document.getElementById('opt-gpu').disabled=true;
  }
}

// Load data
async function loadDashboard(){
  const r=await fetch('/api/runs');
  runsData=await r.json();
  renderStats();renderRuns();renderEventSelect();
  checkGpu();
}

function renderStats(){
  let tf=0,td=0,tp=0,tb=0,ta=0;
  runsData.forEach(r=>{
    const s=r.summary;
    tf+=s.frames||0;td+=s.unique_dogs||0;tp+=s.unique_persons||0;
    tb+=s.bite_alerts||s.bite_risk_alerts||0;
    ta+=s.access_violations||0;
  });
  const n=runsData.length;
  document.getElementById('run-count').textContent=n+' runs';
  document.getElementById('global-stats').innerHTML=`
    <div class="stat-card"><div class="stat-value blue">${n}</div><div class="stat-label">Total Runs</div></div>
    <div class="stat-card"><div class="stat-value">${tf.toLocaleString()}</div><div class="stat-label">Frames</div></div>
    <div class="stat-card"><div class="stat-value green">${td}</div><div class="stat-label">Dogs</div></div>
    <div class="stat-card"><div class="stat-value cyan">${tp}</div><div class="stat-label">Persons</div></div>
    <div class="stat-card"><div class="stat-value red">${tb}</div><div class="stat-label">Bite Alerts</div></div>
    <div class="stat-card"><div class="stat-value orange">${ta}</div><div class="stat-label">Violations</div></div>`;
}

function renderRuns(){
  const g=document.getElementById('runs-grid');
  g.innerHTML=runsData.map(r=>{
    const s=r.summary;
    const bites=s.bite_alerts||s.bite_risk_alerts||0;
    const access=s.access_violations||0;
    let badges='';
    if(bites>0)badges+=`<span class="badge badge-bite">${bites} bites</span>`;
    if(access>0)badges+=`<span class="badge badge-access">${access} violations</span>`;
    if(!bites&&!access)badges='<span class="badge badge-clean">clean</span>';
    const vBtn=r.has_video?`<button class="btn btn-primary" onclick="playVideo('${r.video_url}')"><svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>Video</button>`:'';
    return `<div class="run-card">
      <div class="card-head"><div class="card-name">${r.name}</div><div class="badges">${badges}</div></div>
      <div class="card-body"><div class="metrics">
        <div class="metric"><div class="metric-val g">${s.unique_dogs||0}</div><div class="metric-lbl">Dogs</div></div>
        <div class="metric"><div class="metric-val c">${s.unique_persons||0}</div><div class="metric-lbl">Persons</div></div>
        <div class="metric"><div class="metric-val">${(s.frames||0).toLocaleString()}</div><div class="metric-lbl">Frames</div></div>
        <div class="metric"><div class="metric-val r">${bites}</div><div class="metric-lbl">Bites</div></div>
        <div class="metric"><div class="metric-val o">${access}</div><div class="metric-lbl">Violations</div></div>
        <div class="metric"><div class="metric-val m">${s.avg_fps||0}</div><div class="metric-lbl">FPS</div></div>
      </div></div>
      <div class="card-foot">
        ${vBtn}
        <button class="btn" onclick="viewEvents('${r.name}')">Events</button>
        <button class="btn" onclick="viewJson('${r.name}')">JSON</button>
      </div>
    </div>`;
  }).join('');
}

function renderEventSelect(){
  const sel=document.getElementById('event-select');
  sel.innerHTML='<option value="">Select a run...</option>'+
    runsData.map(r=>`<option value="${r.name}">${r.name}</option>`).join('');
}

// Events
document.getElementById('event-select').addEventListener('change',function(){loadEvents(this.value)});
document.querySelector('.event-filters').addEventListener('click',e=>{
  const btn=e.target.closest('.filter-btn');if(!btn)return;
  currentFilter=btn.dataset.filter;
  document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  const run=document.getElementById('event-select').value;
  if(run)loadEvents(run);
});

async function loadEvents(name){
  if(!name){document.getElementById('events-container').innerHTML='<div class="empty">Select a run to view event logs.</div>';return}
  const r=await fetch('/api/events/'+name);
  let events=await r.json();
  if(currentFilter!=='all')events=events.filter(e=>e.event_type===currentFilter);
  if(!events.length){
    document.getElementById('events-container').innerHTML='<div class="empty">No events found for this run'+(currentFilter!=='all'?' with current filter':'')+'.</div>';
    return;
  }
  let h='<table class="events"><thead><tr><th>Type</th><th>Frame</th><th>Stream</th><th>Score / Time</th><th>Reason</th><th>Details</th></tr></thead><tbody>';
  events.forEach(e=>{
    const isBite=e.event_type==='bite_risk';
    const cls=isBite?'ev-bite':'ev-access';
    const label=isBite?'BITE RISK':'UNAUTHORIZED';
    let score='';
    if(isBite){
      const pct=Math.round((e.risk_score||0)*100);
      const lvl=pct>=60?'high':pct>=45?'med':'low';
      score=`<span class="ev-score ${lvl}">${pct}%</span>`;
    }else{score=e.current_time||'-'}
    const reason=(e.reason||'-').replace(/_/g,' ');
    const details=isBite?`Dog #${e.dog_track_id} → Person #${e.person_track_id}`
      :`Person #${e.person_track_id}`;
    h+=`<tr><td class="${cls}">${label}</td><td>${e.frame_idx}</td><td>${e.stream_id}</td><td>${score}</td><td>${reason}</td><td>${details}</td></tr>`;
  });
  h+='</tbody></table>';
  document.getElementById('events-container').innerHTML=h;
}

function viewEvents(name){
  document.getElementById('event-select').value=name;
  loadEvents(name);switchToTab('events');
}

// JSON viewer
async function viewJson(name){
  const r=await fetch('/api/summary/'+name);
  const data=await r.json();
  document.getElementById('json-title').textContent=name+' — summary.json';
  document.getElementById('json-body').textContent=JSON.stringify(data,null,2);
  document.getElementById('json-modal').classList.add('show');
}
function closeJson(){document.getElementById('json-modal').classList.remove('show')}
document.getElementById('json-modal').addEventListener('click',function(e){if(e.target===this)closeJson()});

// Video player
function playVideo(url){
  const m=document.getElementById('video-modal');
  const v=document.getElementById('modal-video');
  v.src=url;m.classList.add('show');v.play();
}
function closeVideo(){
  const m=document.getElementById('video-modal');
  const v=document.getElementById('modal-video');
  v.pause();v.src='';m.classList.remove('show');
}
document.getElementById('video-modal').addEventListener('click',function(e){
  if(e.target===this)closeVideo();
});

// Upload
const zone=document.getElementById('upload-zone');
const vinput=document.getElementById('video-input');
['dragenter','dragover'].forEach(ev=>zone.addEventListener(ev,e=>{e.preventDefault();zone.classList.add('dragover')}));
['dragleave','drop'].forEach(ev=>zone.addEventListener(ev,e=>{e.preventDefault();zone.classList.remove('dragover')}));
zone.addEventListener('drop',e=>{if(e.dataTransfer.files.length)uploadVideo(e.dataTransfer.files[0])});
vinput.addEventListener('change',()=>{if(vinput.files.length)uploadVideo(vinput.files[0])});

async function uploadVideo(file){
  const fd=new FormData();
  fd.append('video',file);
  fd.append('restricted',document.getElementById('opt-restricted').checked);
  fd.append('gpu',document.getElementById('opt-gpu').checked);
  const pw=document.getElementById('progress-wrap');
  const fill=document.getElementById('p-fill');
  const pct=document.getElementById('p-pct');
  const msg=document.getElementById('p-msg');
  const title=document.getElementById('p-title');
  pw.style.display='block';
  title.textContent='Processing: '+file.name;
  msg.textContent='Uploading...';fill.style.width='10%';pct.textContent='10%';
  fill.style.background='';
  try{
    const resp=await fetch('/api/upload',{method:'POST',body:fd});
    const data=await resp.json();
    if(data.error){msg.textContent='Error: '+data.error;fill.style.width='100%';fill.style.background='var(--red)';return}
    const jobId=data.job_id;
    msg.textContent='Running pipeline...';fill.style.width='30%';pct.textContent='30%';
    const poll=setInterval(async()=>{
      const sr=await fetch('/api/job/'+jobId);
      const st=await sr.json();
      if(st.status==='running'){
        const p=Math.min(90,30+(st.progress||0)*60);
        fill.style.width=p+'%';pct.textContent=Math.round(p)+'%';
        msg.textContent=st.message||'Processing...';
      }else if(st.status==='done'){
        clearInterval(poll);fill.style.width='100%';pct.textContent='100%';
        fill.style.background='var(--green)';msg.textContent='Complete!';
        showResult(st.result);loadDashboard();
      }else if(st.status==='error'){
        clearInterval(poll);fill.style.width='100%';
        fill.style.background='var(--red)';msg.textContent='Error: '+st.message;
      }
    },2000);
  }catch(err){msg.textContent='Upload failed: '+err.message;fill.style.background='var(--red)'}
}

function showResult(result){
  const s=result.summary;
  const bites=s.bite_alerts||s.bite_risk_alerts||0;
  const access=s.access_violations||0;
  const vBtn=result.has_video?`<button class="btn btn-primary" onclick="playVideo('${result.video_url}')"><svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>Watch Video</button>`:'';
  document.getElementById('upload-result').innerHTML=`
  <div class="result-card">
    <h3><svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="var(--green)" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>Analysis Complete</h3>
    <div class="metrics" style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px">
      <div class="metric"><div class="metric-val">${(s.frames||0).toLocaleString()}</div><div class="metric-lbl">Frames</div></div>
      <div class="metric"><div class="metric-val g">${s.unique_dogs||0}</div><div class="metric-lbl">Dogs</div></div>
      <div class="metric"><div class="metric-val c">${s.unique_persons||0}</div><div class="metric-lbl">Persons</div></div>
      <div class="metric"><div class="metric-val r">${bites}</div><div class="metric-lbl">Bites</div></div>
      <div class="metric"><div class="metric-val o">${access}</div><div class="metric-lbl">Violations</div></div>
      <div class="metric"><div class="metric-val m">${s.avg_fps||0}</div><div class="metric-lbl">FPS</div></div>
    </div>
    <div style="margin-top:16px;display:flex;gap:8px">
      ${vBtn}
      <button class="btn" onclick="viewEvents('${result.name}')">View Events</button>
    </div>
  </div>`;
}

loadDashboard();
</script>
</body>
</html>"""


def scan_runs() -> list[dict]:
    runs = []
    if not OUT_DIR.exists():
        return runs
    for sub in sorted(OUT_DIR.iterdir()):
        if not sub.is_dir():
            continue
        summary_path = sub / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        video_file = None
        for vname in ["dogvision_output.mp4", "annotated.mp4", "multi_stream_output.mp4"]:
            if (sub / vname).exists():
                video_file = vname
                break
        runs.append({
            "name": sub.name,
            "summary": summary,
            "has_video": video_file is not None,
            "video_url": f"/video/{sub.name}/{video_file}" if video_file else None,
        })
    # Root out/summary.json
    root_summary = OUT_DIR / "summary.json"
    if root_summary.exists():
        try:
            summary = json.loads(root_summary.read_text())
            video_file = None
            for vname in ["dogvision_output.mp4", "multi_stream_output.mp4", "annotated.mp4"]:
                if (OUT_DIR / vname).exists():
                    video_file = vname
                    break
            runs.append({
                "name": "default",
                "summary": summary,
                "has_video": video_file is not None,
                "video_url": f"/video/default/{video_file}" if video_file else None,
            })
        except (json.JSONDecodeError, OSError):
            pass
    return runs


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/runs")
def api_runs():
    return jsonify(scan_runs())


@app.route("/api/summary/<run_name>")
def api_summary(run_name):
    path = OUT_DIR / "summary.json" if run_name == "default" else OUT_DIR / run_name / "summary.json"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    return jsonify(json.loads(path.read_text()))


@app.route("/api/events/<run_name>")
def api_events(run_name):
    path = OUT_DIR / "events.json" if run_name == "default" else OUT_DIR / run_name / "events.json"
    if not path.exists():
        return jsonify([])
    try:
        data = json.loads(path.read_text())
        return jsonify(data if isinstance(data, list) else [])
    except (json.JSONDecodeError, OSError):
        return jsonify([])


@app.route("/video/<run_name>/<filename>")
def serve_video(run_name, filename):
    directory = OUT_DIR if run_name == "default" else OUT_DIR / run_name
    src = directory / filename
    if not src.exists():
        return "Not found", 404
    # Transcode to H.264 for browser compatibility
    h264 = _transcode_h264(src)
    return send_from_directory(str(h264.parent.resolve()), h264.name, mimetype="video/mp4")


@app.route("/api/gpu-status")
def api_gpu_status():
    try:
        import torch
        if torch.cuda.is_available():
            return jsonify({
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda or "unknown",
            })
    except ImportError:
        pass
    return jsonify({"available": False})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    video = request.files["video"]
    if video.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    restricted = request.form.get("restricted", "false") == "true"
    use_gpu = request.form.get("gpu", "false") == "true"
    job_id = str(uuid.uuid4())[:8]
    video_path = UPLOAD_DIR / f"upload_{job_id}.mp4"
    video.save(str(video_path))
    out_name = f"upload_{job_id}"
    out_path = OUT_DIR / out_name
    jobs[job_id] = {"status": "running", "progress": 0, "message": "Starting pipeline..."}
    thread = threading.Thread(
        target=_process_video,
        args=(job_id, str(video_path), str(out_path), restricted, use_gpu),
        daemon=True,
    )
    thread.start()
    return jsonify({"job_id": job_id, "status": "started"})


def _process_video(job_id, video_path, out_path, restricted, use_gpu=False):
    try:
        script = "run_demo_gpu.py" if use_gpu else "run_demo_cpu.py"
        mode = "GPU (TensorRT FP16)" if use_gpu else "CPU"
        jobs[job_id]["message"] = f"Running {mode} pipeline..."
        jobs[job_id]["progress"] = 0.1
        cmd = [
            sys.executable, script,
            "--source", video_path, "--no-display", "--out", out_path,
        ]
        if restricted:
            cmd += ["--access-config", "configs/access_schedule_restricted.yaml"]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(Path(__file__).parent),
        )
        for line in proc.stdout:
            line = line.strip()
            if "[mvp] frame" in line:
                for p in line.split():
                    if "/" in p and p[0].isdigit():
                        try:
                            cur, tot = p.split("/")
                            jobs[job_id]["progress"] = int(cur) / max(int(tot), 1)
                            jobs[job_id]["message"] = line
                        except ValueError:
                            pass
        proc.wait()
        summary_path = Path(out_path) / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"frames": 0}
        video_file = None
        for vname in ["dogvision_output.mp4", "annotated.mp4"]:
            if (Path(out_path) / vname).exists():
                video_file = vname
                break
        run_name = Path(out_path).name
        jobs[job_id] = {
            "status": "done", "progress": 1.0, "message": "Complete",
            "result": {
                "name": run_name, "summary": summary,
                "has_video": video_file is not None,
                "video_url": f"/video/{run_name}/{video_file}" if video_file else None,
            },
        }
    except Exception as e:
        jobs[job_id] = {"status": "error", "progress": 1.0, "message": str(e)}


@app.route("/api/job/<job_id>")
def api_job(job_id):
    if job_id not in jobs:
        return jsonify({"error": "job not found"}), 404
    return jsonify(jobs[job_id])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("dogvision-dashboard")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"\n  DogVision Dashboard")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Runs found: {len(scan_runs())}\n")
    app.run(host=args.host, port=args.port, debug=False)
