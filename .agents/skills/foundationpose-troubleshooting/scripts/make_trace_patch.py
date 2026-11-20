#!/usr/bin/env python3
"""Generate (do not apply) a temporary numeric-trace patch for this library."""

import argparse
import difflib
from pathlib import Path
from numeric import default_repo


def replace_once(text, anchor, replacement):
    count = text.count(anchor)
    if count != 1:
        raise ValueError(f"expected one source anchor, found {count}: {anchor[:100]!r}; inspect this revision")
    return text.replace(anchor, replacement, 1)


def instrument(source):
    source = replace_once(source, '#include "pipeline_utils.hpp"',
                          '#include "pipeline_utils.hpp"\n#include "fp_numeric_trace.hpp"')
    start = source.index("PoseEstimate FoundationPose::registerFrame(")
    middle = source.index("PoseEstimate FoundationPose::trackFrame(", start)
    end = source.index("void FoundationPose::validateFrame(", middle)
    register, track = source[start:middle], source[middle:end]
    register = replace_once(register, "  uploadFrameToDevice(",
        '  FpNumericTrace trace("register", active_hypotheses_, refine_iters, width, height,\n'
        '                       intrinsics, mesh_, config_, stream_->get());\n\n  uploadFrameToDevice(')
    register = replace_once(register, "  float* poses = workspace_->poses.as<float>();",
        '  trace.frame(*workspace_, width, height);\n'
        '  trace.device("init_translation_m", workspace_->init_translation.as<float>(), {3});\n'
        '  trace.device("initial_poses_centered", workspace_->poses.as<float>(), {active_hypotheses_, 4, 4});\n'
        '  float* poses = workspace_->poses.as<float>();')
    register = replace_once(register, "  inference_->enqueueScore(",
        '  trace.inputs("score", *workspace_, poses, active_hypotheses_);\n  inference_->enqueueScore(')
    register = replace_once(register, "  selectBestPoseOnDevice(",
        '  trace.device("scores", workspace_->scores.as<float>(), {active_hypotheses_});\n'
        '  selectBestPoseOnDevice(')
    register = replace_once(register, "  return estimate;",
        "  trace.finish(estimate.pose, estimate.score, host_result.index);\n  return estimate;")
    track = replace_once(track, "  uploadFrameToDevice(",
        '  FpNumericTrace trace("track", 1, refine_iters, width, height,\n'
        '                       intrinsics, mesh_, config_, stream_->get());\n  uploadFrameToDevice(')
    track = replace_once(track, "  float* poses = workspace_->poses.as<float>();",
        '  trace.frame(*workspace_, width, height);\n'
        '  trace.device("incoming_pose_centered", workspace_->poses.as<float>(), {1, 4, 4});\n'
        '  float* poses = workspace_->poses.as<float>();')
    track = replace_once(track, "  return readBestResult();",
        '  const auto estimate = readBestResult();\n'
        '  trace.finish(estimate.pose, estimate.score, 0);\n  return estimate;')
    source = source[:start] + register + track + source[end:]
    start = source.index("void FoundationPose::runRefinementLoop(")
    end = source.index("void FoundationPose::captureRefinementGraph(", start)
    refine = source[start:end]
    refine = replace_once(refine, "    inference_->enqueueRefine(",
        '    auto* trace = FpNumericTrace::current();\n'
        '    const std::string stage = "refine_" + std::to_string(iter);\n'
        '    if (trace) trace->inputs(stage, *workspace_, poses, batch_size);\n'
        '    inference_->enqueueRefine(')
    refine = replace_once(refine, "    applyPoseDeltasOnDevice(",
        '    if (trace) {\n'
        '      trace->device(stage + "_delta_translation_raw", workspace_->delta_translation.as<float>(), {batch_size, 3});\n'
        '      trace->device(stage + "_delta_rotation_raw", workspace_->delta_rotation.as<float>(), {batch_size, 3});\n'
        '    }\n    applyPoseDeltasOnDevice(')
    anchor = "                            mesh_.diameter, config_, stream_->get());"
    refine = replace_once(refine, anchor, anchor + '\n'
        '    if (trace) trace->device(stage + "_poses_after_centered", poses, {batch_size, 4, 4});')
    return source[:start] + refine + source[end:]


def make_patch(repo):
    source_path = repo / "src/foundation_pose.cpp"
    if (repo / "src/fp_numeric_trace.hpp").exists():
        raise ValueError("src/fp_numeric_trace.hpp already exists; do not replace existing instrumentation")
    original = source_path.read_text()
    modified = instrument(original)
    header = (Path(__file__).resolve().parents[1] / "assets/fp_numeric_trace.hpp").read_text()
    return "".join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                                       fromfile="a/src/foundation_pose.cpp", tofile="b/src/foundation_pose.cpp")) + "".join(
        difflib.unified_diff([], header.splitlines(True), fromfile="/dev/null", tofile="b/src/fp_numeric_trace.hpp"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=default_repo())
    parser.add_argument("--output", type=Path, required=True, help="new .patch file; inspect and apply separately")
    args = parser.parse_args()
    try:
        patch = make_patch(args.repo)
        with args.output.open("x") as stream:
            stream.write(patch)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Wrote {args.output}; checkout unchanged. Apply with git apply --check, then git apply; rebuild.")


if __name__ == "__main__":
    main()
