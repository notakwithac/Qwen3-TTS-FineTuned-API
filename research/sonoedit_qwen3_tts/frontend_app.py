"""Temporary SonoEdit experiment UI for Qwen3-TTS pronunciation edits."""

from __future__ import annotations

import json
import shutil
import subprocess
import traceback
import uuid
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse


ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs" / "sonoedit_frontend"
JOBS_DIR = ROOT / "jobs"
DOCKER_API_BASE = "http://127.0.0.1:8000"
DOCKER_CONTAINER = "qwen3-tts-api-1"

app = FastAPI(title="SonoEdit Lab", version="0.1.0")


def _json_error(status_code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail)


def _map_container_path(path: str | Path | None) -> Path | None:
    if not path:
        return None
    raw = str(path)
    if raw.startswith("/app/") or raw == "/app":
        return ROOT / raw.removeprefix("/app/").replace("/", "\\")
    return Path(raw)


def _to_container_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"path is outside workspace and cannot be mapped into Docker: {resolved}") from exc
    return "/app/" + relative.as_posix()


def _run(command: list[str], *, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)


def _ensure_sonoedit_in_container() -> None:
    source = ROOT / "research" / "sonoedit_qwen3_tts"
    setup = _run(["docker", "exec", DOCKER_CONTAINER, "bash", "-lc", "mkdir -p /app/research"], timeout=60)
    if setup.returncode != 0:
        raise RuntimeError(setup.stderr or setup.stdout or "failed to prepare /app/research in Docker")
    copied = _run(["docker", "cp", str(source), f"{DOCKER_CONTAINER}:/app/research/"], timeout=120)
    if copied.returncode != 0:
        raise RuntimeError(copied.stderr or copied.stdout or "failed to copy SonoEdit code into Docker")


def _read_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for job_json in JOBS_DIR.glob("*/job.json"):
        try:
            data = json.loads(job_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        checkpoint = _map_container_path(data.get("checkpoint_path"))
        data["job_json_path"] = str(job_json)
        data["local_checkpoint_path"] = str(checkpoint) if checkpoint else ""
        data["local_checkpoint_exists"] = bool(checkpoint and checkpoint.exists())
        jobs.append(data)
    return sorted(jobs, key=lambda item: item.get("last_accessed_at") or item.get("created_at") or "", reverse=True)


def _job_summary(job: dict[str, Any]) -> dict[str, Any]:
    config = job.get("config") or {}
    return {
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "speaker_name": job.get("speaker_name"),
        "character_id": config.get("character_id"),
        "checkpoint_path": job.get("checkpoint_path"),
        "local_checkpoint_path": job.get("local_checkpoint_path"),
        "local_checkpoint_exists": job.get("local_checkpoint_exists"),
        "available_checkpoint_epochs": job.get("available_checkpoint_epochs") or [],
    }


def _session_dir() -> Path:
    path = RUNS_DIR / uuid.uuid4().hex[:12]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_request(
    *,
    work_dir: Path,
    checkpoint_path: Path,
    output_checkpoint_path: Path,
    target_term: str,
    source_sentence: str,
    exemplar_audio_path: Path,
    preservation_sentences: str,
    layer: str,
    codec_start: int | None,
    codec_end: int | None,
) -> dict[str, Any]:
    preservation = [{"sentence": line.strip()} for line in preservation_sentences.splitlines() if line.strip()]
    if not preservation:
        preservation = [{"sentence": source_sentence}]

    span = (codec_start, codec_end) if codec_start is not None and codec_end is not None else None
    request = {
        "target_term": target_term,
        "source_sentence": source_sentence,
        "desired_pronunciation": {
            "audio_path": _to_container_path(exemplar_audio_path),
            "transcript": source_sentence,
            "codec0_frame_span": list(span) if span else None,
        },
        "preservation_manifest": preservation,
        "model_checkpoint_path": _to_container_path(checkpoint_path),
        "output_checkpoint_path": _to_container_path(output_checkpoint_path),
        "selected_edit_layers": [layer],
        "target_frame_span": list(span) if span else None,
    }
    request_path = work_dir / "sonoedit-request.json"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    return request


def _save_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)


def _uploaded_audio_path(work_dir: Path, upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
        suffix = ".wav"
    return work_dir / f"correct-pronunciation{suffix}"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/api/jobs")
def list_jobs(character_id: str | None = None) -> dict[str, Any]:
    jobs = [_job_summary(job) for job in _read_jobs()]
    if character_id:
        jobs = [job for job in jobs if job.get("character_id") == character_id]
    return {"jobs": jobs}


@app.post("/api/generate")
def generate(
    checkpoint_path: str = Form(...),
    speaker_name: str = Form(...),
    text: str = Form(...),
    language: str = Form("English"),
    instruct: str = Form(""),
    job_id: str = Form(""),
    max_new_tokens: int | None = Form(None),
) -> dict[str, Any]:
    checkpoint = _map_container_path(checkpoint_path)
    if checkpoint is None or not checkpoint.exists():
        raise _json_error(404, f"checkpoint path does not exist: {checkpoint_path}")
    if not text.strip():
        raise _json_error(400, "text is required")

    try:
        if job_id.strip():
            payload: dict[str, Any] = {
                "text": text.strip(),
                "language": language.strip() or "English",
                "instruct": instruct.strip(),
                "upload_to_s3": False,
            }
            if max_new_tokens is not None:
                payload["max_new_tokens"] = max_new_tokens
            response = requests.post(
                f"{DOCKER_API_BASE}/infer/{job_id.strip()}",
                json=payload,
                timeout=300,
            )
            if response.status_code >= 400:
                try:
                    detail = response.json()
                except Exception:
                    detail = response.text
                raise RuntimeError(f"Docker inference failed ({response.status_code}): {detail}")
            wav_bytes = response.content
            sample_rate = int(response.headers.get("X-Sample-Rate") or 24000)
        else:
            _ensure_sonoedit_in_container()
            work_dir = _session_dir()
            output = work_dir / "generated.wav"
            command = [
                "docker",
                "exec",
                DOCKER_CONTAINER,
                "python",
                "-m",
                "research.sonoedit_qwen3_tts.generate_audio",
                "--checkpoint-path",
                _to_container_path(checkpoint),
                "--output-wav",
                _to_container_path(output),
                "--text",
                text.strip(),
                "--speaker",
                speaker_name.strip(),
                "--language",
                language.strip() or "English",
                "--attn-implementation",
                "eager",
            ]
            if instruct.strip():
                command.extend(["--instruct", instruct.strip()])
            if max_new_tokens is not None:
                command.extend(["--max-new-tokens", str(max_new_tokens)])
            generated = _run(command, timeout=900)
            if generated.returncode != 0:
                raise RuntimeError(generated.stderr or generated.stdout or "Docker checkpoint generation failed")
            return {
                "audio_url": f"/api/artifacts/{work_dir.name}/{output.name}",
                "sample_rate": 24000,
                "checkpoint_path": str(checkpoint),
                "job_id": "",
                "backend": "docker-checkpoint",
                "stdout": generated.stdout,
            }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"generation failed: {exc}",
                "checkpoint_path": str(checkpoint),
                "speaker_name": speaker_name.strip(),
                "traceback_tail": traceback.format_exc(limit=8).strip().splitlines()[-16:],
            },
        )
    work_dir = _session_dir()
    output = work_dir / "generated.wav"
    output.write_bytes(wav_bytes)
    return {
        "audio_url": f"/api/artifacts/{work_dir.name}/{output.name}",
        "sample_rate": sample_rate,
        "checkpoint_path": str(checkpoint),
        "job_id": job_id.strip(),
        "backend": "docker-api" if job_id.strip() else "local-checkpoint",
    }


@app.post("/api/build-edit")
def build_edit(
    correct_audio: UploadFile = File(...),
    checkpoint_path: str = Form(...),
    speaker_name: str = Form(...),
    target_term: str = Form(...),
    source_sentence: str = Form(...),
    language: str = Form("English"),
    instruct: str = Form(""),
    layer: str = Form("talker.model.layers.8"),
    preservation_sentences: str = Form(""),
    codec_start: int | None = Form(None),
    codec_end: int | None = Form(None),
    output_delta_scale: float = Form(1.0),
    semantic_codebook_weight: float = Form(1.0),
    residual_codebook_weight: float = Form(0.15),
    max_relative_delta: float = Form(0.001),
) -> dict[str, Any]:
    checkpoint = _map_container_path(checkpoint_path)
    if checkpoint is None or not checkpoint.exists():
        raise _json_error(404, f"checkpoint path does not exist: {checkpoint_path}")
    if not target_term.strip() or not source_sentence.strip():
        raise _json_error(400, "target word and sentence are required")
    if (codec_start is None) != (codec_end is None):
        raise _json_error(400, "codec start and end must be provided together")

    work_dir = _session_dir()
    exemplar = _uploaded_audio_path(work_dir, correct_audio)
    _save_upload(correct_audio, exemplar)
    output_checkpoint = work_dir / "edited-checkpoint"
    delta_path = work_dir / "sonoedit-delta.pt"
    request = _write_request(
        work_dir=work_dir,
        checkpoint_path=checkpoint,
        output_checkpoint_path=output_checkpoint,
        target_term=target_term.strip(),
        source_sentence=source_sentence.strip(),
        exemplar_audio_path=exemplar,
        preservation_sentences=preservation_sentences,
        layer=layer.strip(),
        codec_start=codec_start,
        codec_end=codec_end,
    )
    try:
        _ensure_sonoedit_in_container()
        build_command = [
            "docker",
            "exec",
            DOCKER_CONTAINER,
            "python",
            "-m",
            "research.sonoedit_qwen3_tts.build_delta",
            "--request-json",
            _to_container_path(work_dir / "sonoedit-request.json"),
            "--output-delta",
            _to_container_path(delta_path),
            "--layer",
            layer.strip(),
            "--speaker",
            speaker_name.strip(),
            "--language",
            language.strip() or "English",
            "--attn-implementation",
            "eager",
            "--output-delta-scale",
            str(output_delta_scale),
            "--semantic-codebook-weight",
            str(semantic_codebook_weight),
            "--residual-codebook-weight",
            str(residual_codebook_weight),
            "--max-relative-delta",
            str(max_relative_delta),
        ]
        if instruct.strip():
            build_command.extend(["--instruct", instruct.strip()])
        build_proc = _run(build_command, timeout=1800)
        if build_proc.returncode != 0:
            raise RuntimeError(build_proc.stderr or build_proc.stdout or "Docker delta build failed")
        apply_proc = _run(
            [
                "docker",
                "exec",
                DOCKER_CONTAINER,
                "python",
                "-m",
                "research.sonoedit_qwen3_tts.apply_edit",
                "--request-json",
                _to_container_path(work_dir / "sonoedit-request.json"),
                "--delta-file",
                _to_container_path(delta_path),
            ],
            timeout=900,
        )
        if apply_proc.returncode != 0:
            raise RuntimeError(apply_proc.stderr or apply_proc.stdout or "Docker delta apply failed")
        build_result = json.loads(build_proc.stdout[build_proc.stdout.find("{") :])
        apply_result = json.loads(apply_proc.stdout[apply_proc.stdout.find("{") :])
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"build/edit failed: {exc}",
                "work_dir": str(work_dir),
                "traceback_tail": traceback.format_exc(limit=8).strip().splitlines()[-16:],
            },
        )
    return {
        "work_dir": str(work_dir),
        "request_json": str(work_dir / "sonoedit-request.json"),
        "delta_path": str(delta_path),
        "delta_metadata_path": str(delta_path.with_suffix(delta_path.suffix + ".json")),
        "edited_checkpoint_path": str(output_checkpoint),
        "build": build_result,
        "apply": apply_result,
    }


@app.get("/api/artifacts/{session_id}/{filename}")
def artifact(session_id: str, filename: str) -> FileResponse:
    path = RUNS_DIR / session_id / filename
    if not path.exists() or not path.is_file():
        raise _json_error(404, "artifact not found")
    media_type = "audio/wav" if path.suffix.lower() == ".wav" else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=filename)


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SonoEdit Lab</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #20241d;
      --muted: #687060;
      --paper: #f8f6ef;
      --line: #d9d2c2;
      --field: #fffdf7;
      --accent: #0f766e;
      --accent-ink: #f4fffc;
      --warn: #9a3412;
      --ok: #166534;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: ui-serif, Georgia, Cambria, "Times New Roman", serif;
    }
    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 56px;
    }
    header {
      display: grid;
      grid-template-columns: 1.15fr .85fr;
      gap: 28px;
      align-items: end;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 22px;
    }
    h1 {
      margin: 0;
      font-size: clamp(34px, 5vw, 68px);
      line-height: .93;
      font-weight: 700;
      letter-spacing: 0;
    }
    .subtitle {
      margin: 0;
      color: var(--muted);
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 22px;
      align-items: start;
    }
    section, aside {
      border-top: 1px solid var(--line);
      padding-top: 18px;
    }
    h2 {
      margin: 0 0 14px;
      font-size: 22px;
      line-height: 1.1;
    }
    label {
      display: grid;
      gap: 7px;
      margin-bottom: 13px;
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 13px;
      font-weight: 650;
    }
    input, textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      background: var(--field);
      color: var(--ink);
      border-radius: 6px;
      padding: 10px 11px;
      font: 14px/1.35 ui-sans-serif, system-ui, sans-serif;
    }
    textarea { min-height: 84px; resize: vertical; }
    input:focus, textarea:focus, select:focus {
      outline: 2px solid color-mix(in srgb, var(--accent), transparent 65%);
      border-color: var(--accent);
    }
    .row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .triple {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 12px;
    }
    button {
      border: 1px solid var(--accent);
      background: var(--accent);
      color: var(--accent-ink);
      border-radius: 6px;
      padding: 11px 14px;
      font: 700 14px/1 ui-sans-serif, system-ui, sans-serif;
      cursor: pointer;
    }
    button.secondary {
      background: transparent;
      color: var(--accent);
    }
    button:disabled { opacity: .55; cursor: wait; }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 16px 0 22px;
    }
    .audio-pair {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-top: 18px;
    }
    .audio-box {
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }
    .audio-box h3 {
      margin: 0 0 10px;
      font: 700 13px/1 ui-sans-serif, system-ui, sans-serif;
      text-transform: uppercase;
      letter-spacing: 0;
      color: var(--muted);
    }
    audio { width: 100%; }
    pre {
      min-height: 180px;
      max-height: 420px;
      overflow: auto;
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fffcf2;
      padding: 12px;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Consolas, monospace;
    }
    .status {
      margin: 0 0 14px;
      color: var(--muted);
      font: 13px/1.45 ui-sans-serif, system-ui, sans-serif;
    }
    .ok { color: var(--ok); }
    .warn { color: var(--warn); }
    @media (max-width: 880px) {
      header, .grid, .audio-pair { grid-template-columns: 1fr; }
      main { width: min(100vw - 24px, 720px); }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>SonoEdit Lab</h1>
      <p class="subtitle">Temporary pronunciation edit console for one checkpoint at a time. Build small deltas, compare baseline and edited audio, keep the generated checkpoint under <code>runs/sonoedit_frontend</code>.</p>
    </header>

    <div class="grid">
      <section>
        <h2>Experiment</h2>
        <div class="row">
          <label>Character ID
            <input id="characterId" value="44846261-4d39-47f9-89be-f311652352c4" />
          </label>
          <label>Job
            <select id="jobSelect"></select>
          </label>
        </div>
        <input id="jobId" type="hidden" />
        <div class="row">
          <label>Checkpoint path
            <input id="checkpointPath" />
          </label>
          <label>Speaker
            <input id="speakerName" />
          </label>
        </div>
        <label>Sentence
          <textarea id="sentence">Put the target word in a natural sentence.</textarea>
        </label>
        <div class="row">
          <label>Target word
            <input id="targetWord" placeholder="word to edit" />
          </label>
          <label>Language
            <input id="language" value="English" />
          </label>
        </div>
        <label>Correct pronunciation audio
          <input id="correctAudio" type="file" accept="audio/*" />
        </label>
        <label>Preservation sentences
          <textarea id="preservation" placeholder="One sentence per line. Leave blank to preserve the source sentence."></textarea>
        </label>
        <details>
          <summary>Advanced</summary>
          <div class="triple">
            <label>Layer
              <input id="layer" value="talker.model.layers.8" />
            </label>
            <label>Codec start
              <input id="codecStart" type="number" min="0" />
            </label>
            <label>Codec end
              <input id="codecEnd" type="number" min="1" />
            </label>
          </div>
          <div class="row">
            <label>Delta scale
              <input id="deltaScale" type="number" step="0.1" value="1.0" />
            </label>
            <label>Max relative delta
              <input id="maxRelativeDelta" type="number" step="0.0001" value="0.001" />
            </label>
          </div>
          <div class="row">
            <label>Codec 0 weight
              <input id="semanticWeight" type="number" step="0.1" value="1.0" />
            </label>
            <label>Residual codec weight
              <input id="residualWeight" type="number" step="0.05" value="0.15" />
            </label>
          </div>
          <label>Instruct
            <input id="instruct" />
          </label>
        </details>

        <div class="actions">
          <button id="loadJobs" class="secondary">Find Job</button>
          <button id="baseline">Generate Baseline</button>
          <button id="build">Build Edit</button>
          <button id="edited" class="secondary">Generate Edited</button>
        </div>

        <div class="audio-pair">
          <div class="audio-box">
            <h3>Baseline</h3>
            <audio id="baselineAudio" controls></audio>
          </div>
          <div class="audio-box">
            <h3>Edited</h3>
            <audio id="editedAudio" controls></audio>
          </div>
        </div>
      </section>

      <aside>
        <h2>Run Log</h2>
        <p id="status" class="status">Ready.</p>
        <pre id="log"></pre>
      </aside>
    </div>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    let editedCheckpoint = "";

    function log(value) {
      $("log").textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    }

    function status(text, kind = "") {
      $("status").className = "status " + kind;
      $("status").textContent = text;
    }

    async function api(url, options) {
      const response = await fetch(url, options);
      const type = response.headers.get("content-type") || "";
      const body = type.includes("application/json") ? await response.json() : await response.text();
      if (!response.ok) throw new Error(body.detail || body);
      return body;
    }

    function setBusy(isBusy) {
      for (const id of ["loadJobs", "baseline", "build", "edited"]) $(id).disabled = isBusy;
    }

    async function loadJobs() {
      setBusy(true);
      try {
        status("Looking up local jobs...");
        const data = await api("/api/jobs?character_id=" + encodeURIComponent($("characterId").value.trim()));
        $("jobSelect").innerHTML = "";
        for (const job of data.jobs) {
          const opt = document.createElement("option");
          opt.value = JSON.stringify(job);
          opt.textContent = `${job.job_id} · ${job.status} · ${job.speaker_name}`;
          $("jobSelect").appendChild(opt);
        }
        if (data.jobs[0]) applyJob(data.jobs[0]);
        status(data.jobs.length ? "Job loaded." : "No matching local job found.", data.jobs.length ? "ok" : "warn");
        log(data);
      } catch (error) {
        status(error.message, "warn");
      } finally {
        setBusy(false);
      }
    }

    function applyJob(job) {
      $("jobId").value = job.job_id || "";
      $("checkpointPath").value = job.local_checkpoint_path || job.checkpoint_path || "";
      $("speakerName").value = job.speaker_name || "";
    }

    $("jobSelect").addEventListener("change", () => {
      if ($("jobSelect").value) applyJob(JSON.parse($("jobSelect").value));
    });

    async function generate(target) {
      const form = new FormData();
      form.set("checkpoint_path", target === "edited" ? editedCheckpoint : $("checkpointPath").value);
      form.set("speaker_name", $("speakerName").value);
      form.set("text", $("sentence").value);
      form.set("language", $("language").value);
      form.set("instruct", $("instruct").value);
      form.set("job_id", target === "edited" ? "" : $("jobId").value);
      setBusy(true);
      try {
        status(target === "edited" ? "Generating edited audio..." : "Generating baseline audio...");
        const data = await api("/api/generate", { method: "POST", body: form });
        $(target === "edited" ? "editedAudio" : "baselineAudio").src = data.audio_url + "?t=" + Date.now();
        status("Audio generated.", "ok");
        log(data);
      } catch (error) {
        status(error.message, "warn");
      } finally {
        setBusy(false);
      }
    }

    async function buildEdit() {
      if (!$("correctAudio").files[0]) {
        status("Upload the correct pronunciation audio first.", "warn");
        return;
      }
      const form = new FormData();
      for (const [key, id] of [
        ["checkpoint_path", "checkpointPath"], ["speaker_name", "speakerName"],
        ["target_term", "targetWord"], ["source_sentence", "sentence"],
        ["language", "language"], ["instruct", "instruct"], ["layer", "layer"],
        ["preservation_sentences", "preservation"], ["output_delta_scale", "deltaScale"],
        ["max_relative_delta", "maxRelativeDelta"], ["semantic_codebook_weight", "semanticWeight"],
        ["residual_codebook_weight", "residualWeight"]
      ]) form.set(key, $(id).value);
      if ($("codecStart").value !== "") form.set("codec_start", $("codecStart").value);
      if ($("codecEnd").value !== "") form.set("codec_end", $("codecEnd").value);
      form.set("correct_audio", $("correctAudio").files[0]);
      setBusy(true);
      try {
        status("Building and applying delta. This can take a while...");
        const data = await api("/api/build-edit", { method: "POST", body: form });
        editedCheckpoint = data.edited_checkpoint_path;
        status("Edited checkpoint created.", "ok");
        log(data);
      } catch (error) {
        status(error.message, "warn");
      } finally {
        setBusy(false);
      }
    }

    $("loadJobs").addEventListener("click", loadJobs);
    $("baseline").addEventListener("click", () => generate("baseline"));
    $("edited").addEventListener("click", () => {
      if (!editedCheckpoint) {
        status("Build an edit first.", "warn");
        return;
      }
      generate("edited");
    });
    $("build").addEventListener("click", buildEdit);
    loadJobs();
  </script>
</body>
</html>
"""
