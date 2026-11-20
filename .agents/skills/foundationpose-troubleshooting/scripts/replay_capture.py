#!/usr/bin/env python3
"""Replay ordered NPZ captures through the library and save numeric evidence."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from numeric import default_repo, file_identity, inspect_frame, load_frame, validate_frame, validate_pose, write_json


def command_output(argv):
    try:
        result = subprocess.run(argv, text=True, capture_output=True, timeout=15, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"unavailable": str(exc)}


def read_manifest(path):
    manifest = json.loads(path.read_text())
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("manifest requires a nonempty frames list")
    ids = set()
    for index, entry in enumerate(frames):
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
            raise ValueError(f"frame {index}: provide a nonempty string id")
        if entry["id"] in ids:
            raise ValueError(f"duplicate frame id: {entry['id']}")
        ids.add(entry["id"])
        if entry.get("action") not in ("register", "track"):
            raise ValueError(f"frame {index}: action must be register or track")
        if index == 0 and entry["action"] != "register":
            raise ValueError("first frame must register; tracking requires initialization")
        entry["path"] = (path.parent / entry["path"]).resolve()
        frame = load_frame(entry["path"])
        validate_frame(frame, require_mask=entry["action"] == "register")
    return manifest


def run_sequence(estimator, frame_type, entries, output, register_hypotheses=None):
    """Stop on the first failure; never silently track from an invented pose."""
    records = []
    for index, entry in enumerate(entries):
        record = {"index": index, "id": entry["id"], "action": entry["action"],
                  "capture_metadata": {k: v for k, v in entry.items() if k != "path"}}
        try:
            data = load_frame(entry["path"])
            frame = frame_type(data["rgb"], data["depth_m"], data["K"],
                               data.get("mask") if entry["action"] == "register" else None)
            kwargs = ({"n_hypotheses": register_hypotheses}
                      if entry["action"] == "register" and register_hypotheses is not None else {})
            result = getattr(estimator, entry["action"])(frame, **kwargs)
            # Save the exact returned pose before checking it, including invalid outputs.
            np.save(output / f"{index:06d}_pose.npy", result.pose, allow_pickle=False)
            validate_pose(result.pose)
            if not np.isfinite([result.score, result.elapsed_s]).all():
                raise ValueError("library returned a nonfinite score or duration")
            record.update(ok=True, pose_file=f"{index:06d}_pose.npy",
                          score=float(result.score), elapsed_s=float(result.elapsed_s),
                          score_meaning="raw scorer output" if entry["action"] == "register"
                          else "constant placeholder (1), not confidence")
        except Exception as exc:
            record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            records.append(record)
            write_json(output / "records.json", records)
            raise
        records.append(record)
    write_json(output / "records.json", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    parser.add_argument("--validate-only", action="store_true", help="CPU input checks, no library load")
    parser.add_argument("--repo", type=Path, default=default_repo())
    parser.add_argument("--library", type=Path, default=os.environ.get("FP_LIBRARY"))
    parser.add_argument("--cad", type=Path)
    parser.add_argument("--mesh-unit-scale", type=float, default=1.0)
    parser.add_argument("--refine-model", type=Path, default=os.environ.get("FP_REFINE_MODEL_PATH"))
    parser.add_argument("--score-model", type=Path, default=os.environ.get("FP_SCORE_MODEL_PATH"))
    parser.add_argument("--engine-cache", type=Path, default=Path(os.environ.get("FP_ENGINE_CACHE_DIR", "engine_cache")))
    parser.add_argument("--config", type=Path, help="RuntimeConfig JSON overrides; explicit CLI flags take precedence")
    parser.add_argument("--precision", choices=("fp32", "tf32", "fp16", "bf16"), help="default: library configuration (TF32)")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--n-hypotheses", type=int, help="default: library configuration (252)")
    parser.add_argument("--register-hypotheses", type=int, help="per-call count; preserve a larger configured engine capacity")
    parser.add_argument("--n-refine", type=int, help="default: library configuration (5)")
    parser.add_argument("--n-track", type=int, help="default: library configuration (2)")
    parser.add_argument("--repeat", type=int, default=1, help="fresh estimator per repeat; shared engine cache")
    args = parser.parse_args()
    created_output = False
    try:
        if (args.repeat < 1 or (args.n_hypotheses is not None and args.n_hypotheses < 1)
                or (args.register_hypotheses is not None and args.register_hypotheses < 1)
                or any(value is not None and value < 0 for value in (args.n_refine, args.n_track))):
            raise ValueError("repeat/hypotheses must be positive and iteration counts nonnegative")
        if not np.isfinite(args.mesh_unit_scale) or args.mesh_unit_scale <= 0:
            raise ValueError("mesh-unit-scale must be positive and finite")
        overrides = json.loads(args.config.read_text()) if args.config else {}
        if not isinstance(overrides, dict):
            raise ValueError("--config must contain a JSON object of RuntimeConfig fields")
        manifest = read_manifest(args.manifest.resolve())
        if not args.validate_only and any(p is None for p in (args.cad, args.refine_model, args.score_model)):
            raise ValueError("replay requires --cad, --refine-model, --score-model (or model FP_* variables)")
        assets = {} if args.validate_only else {
            name: file_identity(path) for name, path in
            (("cad", args.cad), ("refine_model", args.refine_model), ("score_model", args.score_model))}
        args.output.mkdir(parents=True, exist_ok=False)
        created_output = True
        (args.output / "inputs").mkdir()
        saved_entries, input_reports = [], []
        max_height = max_width = 0
        for index, entry in enumerate(manifest["frames"]):
            frame = load_frame(entry["path"])
            height, width = frame["depth_m"].shape
            max_height, max_width = max(max_height, height), max(max_width, width)
            relative = f"inputs/{index:06d}.npz"
            shutil.copyfile(entry["path"], args.output / relative)
            saved_entries.append({**entry, "path": relative})
            input_reports.append({"id": entry["id"], "source": file_identity(entry["path"]),
                                  "metrics": inspect_frame(frame)})
        write_json(args.output / "manifest.json", {**manifest, "frames": saved_entries})
        write_json(args.output / "input_metrics.json", input_reports)
        provenance = {
            "schema_version": 1, "validate_only": args.validate_only,
            "command": sys.argv, "platform": platform.platform(), "python": sys.version,
            "numpy": np.__version__, "process_id": os.getpid(), "repo": str(args.repo.resolve()),
            "git_head": command_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"]),
            "git_status": command_output(["git", "-C", str(args.repo), "status", "--short"]),
            "assets": assets, "mesh_unit_scale": args.mesh_unit_scale,
            "renderer_requested": os.environ.get("FP_RENDERER", "builtin"),
            "trace_directory": os.environ.get("FP_NUMERIC_TRACE_DIR"),
            "trace_tensors": os.environ.get("FP_NUMERIC_TRACE_TENSORS") == "1",
            "repeat": args.repeat, "pose_convention": "camera_from_original_mesh, row-major, meters",
            "register_hypotheses": args.register_hypotheses,
            "config_overrides": overrides,
        }
        write_json(args.output / "run.json", provenance)
        if args.validate_only:
            print(f"Validated {len(saved_entries)} captures; evidence: {args.output}")
            return 0
        # Import only after validation, so analysis works without CUDA or the .so.
        sys.path.insert(0, str(args.repo.resolve() / "python" / "src"))
        from foundation_pose_nvidia import Estimator, EstimatorOptions, RgbdFrame, RuntimeConfig, load_library
        from foundation_pose_nvidia.core.config import Precision

        library = load_library(args.library)
        for field, value in (("n_hypotheses", args.n_hypotheses), ("n_refine_iters", args.n_refine),
                             ("n_track_iters", args.n_track)):
            if value is not None:
                overrides[field] = value
        if args.precision is not None:
            overrides["tensorrt_precision"] = Precision[args.precision.upper()]
        overrides.setdefault("max_image_width", max_width)
        overrides.setdefault("max_image_height", max_height)
        overrides.setdefault("capture_cuda_graph", False)
        config = RuntimeConfig(**overrides).to_ctypes(library)
        if config.n_hypotheses < 1 or min(config.n_refine_iters, config.n_track_iters) < 0:
            raise ValueError("effective configuration requires positive hypotheses and nonnegative iterations")
        if args.register_hypotheses is not None and args.register_hypotheses > config.n_hypotheses:
            raise ValueError("register-hypotheses must not exceed configured capacity")
        options = EstimatorOptions(args.cad.resolve(), args.refine_model.resolve(),
                                   args.score_model.resolve(), args.engine_cache.resolve(),
                                   device_id=args.device_id, mesh_unit_scale=args.mesh_unit_scale)
        provenance.update(library=file_identity(library.path), build_info=library.build_info,
                          config={name: getattr(config, name) for name, *_ in config._fields_},
                          device_id=args.device_id, engine_cache=str(args.engine_cache.resolve()),
                          gpu=command_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"]))
        write_json(args.output / "runtime.json", provenance)
        # Replay the saved copies, which now form a self-contained input package.
        entries = read_manifest(args.output / "manifest.json")["frames"]
        for repeat in range(args.repeat):
            run = args.output / f"repeat_{repeat:03d}"
            run.mkdir()
            with Estimator(options, config, library=library, prepare_batch=args.register_hypotheses or config.n_hypotheses) as est:
                write_json(run / "setup_timings.json", est.setup_timings)
                run_sequence(est, RgbdFrame, entries, run, args.register_hypotheses)
        write_json(args.output / "engines.json", [file_identity(p) for p in sorted(args.engine_cache.glob("*.plan"))])
        print(f"Saved {args.repeat} replay(s): {args.output}")
        return 0
    except Exception as exc:
        if created_output:
            write_json(args.output / "failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
