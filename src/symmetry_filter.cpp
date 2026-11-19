/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "foundation_pose_nvidia/symmetry_filter.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace foundation_pose_nvidia {
namespace {

Mat4f multiply(const Mat4f& a, const Mat4f& b) {
  Mat4f result;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      double sum = 0.0;
      for (int k = 0; k < 4; ++k) sum += static_cast<double>(a(r, k)) * b(k, c);
      result(r, c) = static_cast<float>(sum);
    }
  }
  return result;
}

Mat4f axisRotation(int axis, double degrees) {
  Mat4f result = Mat4f::identity();
  const double radians = degrees * std::numbers::pi / 180.0;
  const float c = static_cast<float>(std::cos(radians));
  const float s = static_cast<float>(std::sin(radians));
  const int a = (axis + 1) % 3;
  const int b = (axis + 2) % 3;
  result(a, a) = c;
  result(b, b) = c;
  result(a, b) = -s;
  result(b, a) = s;
  return result;
}

double rotationDistance(const Mat4f& a, const Mat4f& b) {
  // trace(R_b * R_a^T), as in rotation_delta_deg; radians preserve ordering.
  double trace = 0.0;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) trace += static_cast<double>(b(r, c)) * a(r, c);
  }
  return std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0));
}

void addAngle(std::vector<double>& angles, double angle) {
  angle = std::fmod(angle, 360.0);
  if (angle < 0.0) angle += 360.0;
  if (std::none_of(angles.begin(), angles.end(), [angle](double old) {
        return std::abs(std::remainder(old - angle, 360.0)) < 1e-7;
      })) {
    angles.push_back(angle);
  }
}

}  // namespace

SymmetryFilter::SymmetryFilter(Vec3f model_center)
    : center_(model_center), symmetries_{Mat4f::identity()} {
  if (!std::isfinite(center_.x) || !std::isfinite(center_.y) || !std::isfinite(center_.z)) {
    throw std::invalid_argument("Symmetry mesh center must be finite and in metres");
  }
}

void SymmetryFilter::setAxes(std::span<const std::string> axes) {
  if (axes.size() > 64) throw std::invalid_argument("At most 64 symmetry axis entries are supported");
  std::array<std::vector<double>, 3> angles{{{0.0}, {0.0}, {0.0}}};
  for (const auto& spec : axes) {
    if (spec.size() < 3 || spec[1] != '_' || spec[0] < 'x' || spec[0] > 'z') {
      throw std::invalid_argument("Invalid symmetry axis '" + spec + "': expected x_180, y_90, z_full, etc.");
    }
    auto& values = angles[spec[0] - 'x'];
    const std::string_view text(spec.data() + 2, spec.size() - 2);
    if (text == "full") {
      for (int angle = 30; angle < 360; angle += 30) addAngle(values, angle);
    } else {
      double angle = 0.0;
      const auto parsed = std::from_chars(text.data(), text.data() + text.size(), angle);
      if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !std::isfinite(angle)) {
        throw std::invalid_argument("Invalid symmetry angle '" + spec + "': expected a finite number or full");
      }
      addAngle(values, angle);
    }
  }
  const auto count = angles[0].size() * angles[1].size() * angles[2].size();
  if (count > 4096) throw std::invalid_argument("Symmetry configuration exceeds 4096 combinations");
  std::vector<Mat4f> transforms;
  transforms.reserve(count);
  const double center[] = {center_.x, center_.y, center_.z};
  for (double x : angles[0]) {
    for (double y : angles[1]) {
      for (double z : angles[2]) {
        Mat4f S = multiply(axisRotation(2, z), multiply(axisRotation(1, y), axisRotation(0, x)));
        for (int r = 0; r < 3; ++r) {
          double translation = center[r];
          for (int c = 0; c < 3; ++c) translation -= S(r, c) * center[c];
          S(r, 3) = static_cast<float>(translation);
        }
        transforms.push_back(S);
      }
    }
  }
  symmetries_ = std::move(transforms);
}

Mat4f SymmetryFilter::select(const Mat4f& pose, const Mat4f* previous_pose) const {
  if (!enabled()) return pose;
  validatePose(pose);
  if (previous_pose == nullptr) return pose;
  validatePose(*previous_pose);
  Mat4f selected = pose;
  double best = rotationDistance(*previous_pose, pose);
  // Identity first and strict improvement preserve the original pose on ties.
  for (std::size_t i = 1; i < symmetries_.size(); ++i) {
    const Mat4f candidate = multiply(pose, symmetries_[i]);
    const double distance = rotationDistance(*previous_pose, candidate);
    if (distance < best) {
      best = distance;
      selected = candidate;
    }
  }
  return selected;
}

Mat4f SymmetryFilter::toCenteredPose(const Mat4f& pose) const {
  Mat4f result = pose;
  for (int r = 0; r < 3; ++r) {
    result(r, 3) = static_cast<float>(pose(r, 3) + static_cast<double>(pose(r, 0)) * center_.x +
                                    static_cast<double>(pose(r, 1)) * center_.y +
                                    static_cast<double>(pose(r, 2)) * center_.z);
  }
  return result;
}

void SymmetryFilter::validatePose(const Mat4f& pose) {
  for (float value : pose.values) {
    if (!std::isfinite(value)) throw std::invalid_argument("Pose must be finite");
  }
  for (int c = 0; c < 4; ++c) {
    if (std::abs(pose(3, c) - (c == 3 ? 1.0f : 0.0f)) > 1e-6f) {
      throw std::invalid_argument("Pose must have homogeneous last row [0, 0, 0, 1]");
    }
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      double dot = 0.0;
      for (int r = 0; r < 3; ++r) dot += static_cast<double>(pose(r, a)) * pose(r, b);
      if (std::abs(dot - (a == b ? 1.0 : 0.0)) > 1e-3) {
        throw std::invalid_argument("Pose rotation must be orthonormal");
      }
    }
  }
  const double determinant =
      pose(0, 0) * (pose(1, 1) * pose(2, 2) - pose(1, 2) * pose(2, 1)) -
      pose(0, 1) * (pose(1, 0) * pose(2, 2) - pose(1, 2) * pose(2, 0)) +
      pose(0, 2) * (pose(1, 0) * pose(2, 1) - pose(1, 1) * pose(2, 0));
  if (std::abs(determinant - 1.0) > 1e-3) throw std::invalid_argument("Pose rotation must be proper (determinant +1)");
}

}  // namespace foundation_pose_nvidia
