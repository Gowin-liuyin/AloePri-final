"""Inventory local checkpoints and run a resumable sequential obfuscation queue."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SUPPORTED = {"llama", "qwen2", "bert"}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if os.name != "nt" or attempt == 5:
                raise
            time.sleep(0.02 * 2**attempt)


def source_fingerprint(folder: Path):
    config = folder / "config.json"
    weights = sorted(list(folder.glob("*.safetensors")) + list(folder.glob("pytorch_model*.bin")))
    return {"config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "weights": [{"name": file.name, "bytes": file.stat().st_size,
                         "mtime_ns": file.stat().st_mtime_ns} for file in weights]}


def create_plan(models_root: Path, run_dir: Path, *, threads=8):
    if threads < 1:
        raise ValueError("threads must be positive")
    models_root, run_dir = models_root.resolve(), run_dir.resolve()
    if not models_root.is_dir():
        raise ValueError(f"Models root does not exist: {models_root}")
    if models_root == run_dir or models_root in run_dir.parents or run_dir in models_root.parents:
        raise ValueError("Models root and experiment output must not overlap")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Queue output must be empty: {run_dir}")
    jobs = []
    for config_file in sorted(models_root.rglob("config.json")):
        relative = config_file.parent.relative_to(models_root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        folder = config_file.parent.resolve()
        if folder != models_root and models_root not in folder.parents:
            raise ValueError(f"Model symlink points outside the specified root: {folder}")
        name = relative.as_posix() if relative.parts else models_root.name
        identifier = re.sub(r"[^A-Za-z0-9_.-]+", "_", name) + "_" + hashlib.sha256(name.encode()).hexdigest()[:8]
        job = {"id": identifier, "name": name, "source": str(folder), "attempts": [], "status": "pending"}
        try:
            config = json.loads(config_file.read_text(encoding="utf-8"))
            job.update(model_type=config.get("model_type"), hidden_size=config.get("hidden_size"),
                       layers=config.get("num_hidden_layers"), architectures=config.get("architectures"),
                       requires_custom_runtime=config.get("model_type") == "bert",
                       source_fingerprint=source_fingerprint(folder))
            if not job["source_fingerprint"]["weights"]:
                raise ValueError("No local weight files")
            if config.get("quantization_config"):
                raise ValueError("Quantized source needs a separate verified conversion")
            if job["model_type"] not in SUPPORTED:
                raise ValueError(f"Unsupported architecture: {job['model_type']}")
            job["implementation_variant"] = ("square_transform_with_layernorm_bridge"
                                                if job["model_type"] == "bert" else "square_monomial")
            job["export_dtype"] = "float32" if job["model_type"] == "bert" else "bfloat16"
        except Exception as error:
            job.update(status="blocked", reason=str(error))
        jobs.append(job)
    if not jobs:
        raise ValueError("No model configurations found")
    plan = {"format": "aloepri_model_queue_v1", "created_at": now(), "models_root": str(models_root),
            "run_dir": str(run_dir), "project_root": str(ROOT), "status": "planned",
            "parameters": {"alpha_e": 0.0, "alpha_h": 0.0, "export_device": "cpu",
                           "verification_device": "cuda", "verification_dtype": "float32",
                           "correctness_atol": 2e-4, "max_new_tokens": 8, "threads": threads,
                           "seed": None, "input_reference": "provided_local_checkpoint"}, "jobs": jobs}
    write_json(run_dir / "queue.json", plan)
    return plan


def export_command(job, output, python):
    family = job["model_type"]
    script = {"llama": "export_llama_checkpoint.py", "qwen2": "export_qwen_checkpoint.py",
              "bert": "export_bert_checkpoint.py"}[family]
    command = [python, str(ROOT / "scripts/offline_obfuscation" / script),
               "--model-dir", job["source"], "--export-dir", str(output),
               "--dtype", job["export_dtype"], "--device", "cpu", "--alpha-e", "0"]
    if family != "bert":
        command += ["--alpha-h", "0"]
    return command


def verify_command(job, output, result_dir, parameters, python):
    if job["model_type"] == "bert":
        return [python, str(Path(__file__).resolve()), "verify-bert", "--checkpoint", str(output),
                "--source", job["source"], "--output", str(result_dir / "default.json")]
    return [python, str(ROOT / "scripts/evaluation/run_llama_correctness.py"),
            "--bundle-dir", str(output), "--baseline-model-dir", job["source"],
            "--output-dir", str(result_dir), "--profiles", "default", "--dtype", "float32",
            "--device", parameters["verification_device"], "--max-new-tokens", str(parameters["max_new_tokens"]),
            "--correctness-atol", str(parameters["correctness_atol"]), "--isolate-model-processes"]


def write_checkpoint_view(output: Path, job):
    # A catalog view references the existing checkpoint, without duplicating its weight files.
    write_json(output / "catalog.json", {"format": "aloepri_experiment_checkpoint_view_v1",
        "upload_whole_bundle_to_server": False, "recommended_profile": "default", "reference_profile": "default",
        "profiles": [{"name": "default", "server_dir": "server", "client_secret": "client/client_secret.pt",
                      "requires_custom_runtime": job["requires_custom_runtime"]}]})


def run_step(command, log_file: Path, environment):
    with log_file.open("ab") as log:
        log.write(("COMMAND " + json.dumps(command) + "\n").encode())
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        return process.wait()


def run_queue(run_dir: Path, python: str):
    run_dir = run_dir.resolve()
    plan_file, lock = run_dir / "queue.json", run_dir / "queue.lock"
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if Path(plan["run_dir"]).resolve() != run_dir:
        raise ValueError("Queue output path does not match its plan")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = str(plan["parameters"]["threads"])
    environment.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false",
                       PYTHONUNBUFFERED="1", HF_DEACTIVATE_ASYNC_LOAD="1")
    plan.update(status="running", pid=os.getpid(), started_at=now(), python=python)
    write_json(plan_file, plan)
    try:
        for job in plan["jobs"]:
            if job["status"] in {"complete", "blocked"}:
                continue
            if source_fingerprint(Path(job["source"])) != job["source_fingerprint"]:
                job.update(status="blocked", reason="Input checkpoint changed after the queue was planned")
                write_json(plan_file, plan)
                continue
            attempt = {"number": len(job["attempts"]) + 1, "started_at": now(), "status": "exporting"}
            attempt_dir = run_dir / "jobs" / job["id"] / f"attempt_{attempt['number']:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            output = attempt_dir / "checkpoint"
            attempt.update(directory=str(attempt_dir), checkpoint=str(output))
            job["attempts"].append(attempt)
            job.update(status="running", current_attempt=attempt["number"])
            write_json(plan_file, plan)
            print(f"[{now()}] Exporting {job['name']}", flush=True)
            command = export_command(job, output, python)
            attempt["export_command"] = command
            code = run_step(command, attempt_dir / "export.log", environment)
            attempt["export_exit_code"] = code
            if code:
                job["status"] = attempt["status"] = "export_failed"
            else:
                write_checkpoint_view(output, job)
                attempt["status"] = "verifying"
                write_json(plan_file, plan)
                result_dir = attempt_dir / "correctness"
                result_dir.mkdir()
                command = verify_command(job, output, result_dir, plan["parameters"], python)
                attempt["verification_command"] = command
                print(f"[{now()}] Verifying {job['name']}", flush=True)
                code = run_step(command, attempt_dir / "verify.log", environment)
                attempt["verification_exit_code"] = code
                report_file = result_dir / "default.json"
                if report_file.is_file():
                    report = json.loads(report_file.read_text(encoding="utf-8"))
                    attempt["verification_summary"] = report.get("summary", report)
                passed = code == 0 and attempt.get("verification_summary", {}).get("status") == "pass"
                job["status"] = attempt["status"] = "complete" if passed else "exported_verification_failed"
            attempt["finished_at"] = now()
            attempt["source_unchanged"] = source_fingerprint(Path(job["source"])) == job["source_fingerprint"]
            if not attempt["source_unchanged"]:
                job["status"] = attempt["status"] = "input_changed"
            write_json(plan_file, plan)
            print(f"[{now()}] {job['name']}: {job['status']}", flush=True)
        plan["finished_at"] = now()
        plan["status"] = "complete" if all(job["status"] == "complete" for job in plan["jobs"]) else "finished_with_issues"
        write_json(plan_file, plan)
    finally:
        lock.unlink(missing_ok=True)
    return plan


def verify_bert(checkpoint: Path, source: Path, output: Path):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.aloepri._implementation.bert.bert_correctness import run_stage_k_bert_profile_correctness
    report = run_stage_k_bert_profile_correctness(release_dir=checkpoint, profile="default",
                                                baseline_model_dir=source, dtype="float32", device="cpu")
    write_json(output, report)
    return report["status"] == "pass"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--models-root", type=Path, required=True)
    plan.add_argument("--run-dir", type=Path, required=True)
    plan.add_argument("--threads", type=int, default=8)
    run = sub.add_parser("run")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--python", default=sys.executable)
    status = sub.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    bert = sub.add_parser("verify-bert")
    bert.add_argument("--checkpoint", type=Path, required=True)
    bert.add_argument("--source", type=Path, required=True)
    bert.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "plan":
        result = create_plan(args.models_root, args.run_dir, threads=args.threads)
    elif args.action == "run":
        result = run_queue(args.run_dir, args.python)
    elif args.action == "verify-bert":
        raise SystemExit(0 if verify_bert(args.checkpoint, args.source, args.output) else 1)
    else:
        result = json.loads((args.run_dir / "queue.json").read_text(encoding="utf-8"))
    print(json.dumps({"status": result["status"], "jobs": [{"name": job["name"], "status": job["status"]}
                                                              for job in result["jobs"]]}, indent=2))


if __name__ == "__main__":
    main()
