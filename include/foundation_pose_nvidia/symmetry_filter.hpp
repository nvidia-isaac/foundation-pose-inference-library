/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <span>
#include <string>
#include <vector>

#include "foundation_pose_nvidia/types.hpp"

namespace foundation_pose_nvidia {

// Temporal equivalent-pose selection, ported from nvlabs_symmetry_filter.py.
// Poses and center are in the original, scaled mesh frame (metres).
class SymmetryFilter {
 public:
  explicit SymmetryFilter(Vec3f model_center = {});

  // Isaac ROS syntax: x_180, y_90, z_full, etc. Numeric angles are explicit
  // alternatives, NOT steps. Identity is always included. "full" samples
  // 0..330 degrees in 30-degree steps; it is not exact continuous symmetry.
  // Multiple axes form Rz * Ry * Rx combinations. Empty disables the filter.
  // Invalid configuration throws without changing the existing configuration.
  void setAxes(std::span<const std::string> axes);
  bool enabled() const { return symmetries_.size() > 1; }

  // Choose T @ S nearest in rotation to the preceding accepted public pose.
  // S = Trans(center) @ R @ Trans(-center), preserving the physical center.
  // No preceding pose: retain the supplied first orientation as the anchor.
  Mat4f select(const Mat4f& pose, const Mat4f* previous_pose) const;
  Mat4f toCenteredPose(const Mat4f& pose) const;
  static void validatePose(const Mat4f& pose);

 private:
  Vec3f center_;
  std::vector<Mat4f> symmetries_;
};

}  // namespace foundation_pose_nvidia
