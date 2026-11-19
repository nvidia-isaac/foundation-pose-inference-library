/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>

#include "foundation_pose_nvidia/symmetry_filter.hpp"

using namespace foundation_pose_nvidia;

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void nearPose(const Mat4f& actual, const Mat4f& expected, float tolerance = 2e-6f) {
  for (int i = 0; i < 16; ++i) require(std::abs(actual.values[i] - expected.values[i]) < tolerance, "Pose mismatch");
}
Vec3f point(const Mat4f& T, Vec3f p) {
  return {T(0, 0)*p.x + T(0, 1)*p.y + T(0, 2)*p.z + T(0, 3),
          T(1, 0)*p.x + T(1, 1)*p.y + T(1, 2)*p.z + T(1, 3),
          T(2, 0)*p.x + T(2, 1)*p.y + T(2, 2)*p.z + T(2, 3)};
}
Mat4f rotate(int axis, float degrees, Vec3f c = {}) {
  Mat4f T = Mat4f::identity();
  const float angle = degrees * std::numbers::pi_v<float> / 180.f;
  const int a = (axis + 1) % 3, b = (axis + 2) % 3;
  T(a, a) = T(b, b) = std::cos(angle);
  T(a, b) = -std::sin(angle); T(b, a) = std::sin(angle);
  const Vec3f rc = point(T, c);
  T(0, 3) = c.x - rc.x; T(1, 3) = c.y - rc.y; T(2, 3) = c.z - rc.z;
  return T;
}
Mat4f product(const Mat4f& a, const Mat4f& b) {
  Mat4f out;
  for (int i=0;i<4;++i) for(int j=0;j<4;++j) for(int k=0;k<4;++k) out(i,j)+=a(i,k)*b(k,j);
  return out;
}
template<class F> void rejects(F action) {
  bool threw=false;
  try { action(); } catch(const std::invalid_argument&) { threw=true; }
  require(threw,"Expected invalid_argument");
}
void configure(SymmetryFilter& filter, std::initializer_list<std::string> axes) {
  const std::vector<std::string> values(axes); filter.setAxes(values);
}
}  // namespace

int main() {
  try {
    const Vec3f center{.06388f, .04274f, .00795f};
    const Mat4f first = Mat4f::identity();
    SymmetryFilter filter(center);
    const Mat4f flipped = rotate(2, 180, center);
    require(filter.select(flipped, &first).values == flipped.values, "Disabled must be bit-identical");
    configure(filter, {"z_180"});
    require(filter.select(flipped, nullptr).values == flipped.values, "First pose must anchor orientation");
    nearPose(filter.select(flipped, &first), first);
    Mat4f translated = flipped; translated(0,3) += .003f; translated(2,3) += .002f;
    const Mat4f corrected = filter.select(translated, &first);
    require(std::abs(corrected(0,3)-.003f)<1e-6f && std::abs(corrected(2,3)-.002f)<1e-6f,
            "Real translation must survive symmetry correction");
    const auto raw_center = point(translated, center), filtered_center = point(corrected, center);
    require(std::abs(raw_center.x-filtered_center.x)<1e-6f && std::abs(raw_center.y-filtered_center.y)<1e-6f &&
            std::abs(raw_center.z-filtered_center.z)<1e-6f,"Physical center moved");
    const auto centered = filter.toCenteredPose(corrected);
    require(std::abs(centered(0,3)-filtered_center.x)<1e-6f && std::abs(centered(2,3)-filtered_center.z)<1e-6f,
            "Tracking seed must use centered mesh coordinates");

    for (int axis=0;axis<3;++axis) {
      const std::string name(1, static_cast<char>('x'+axis));
      configure(filter, {name+"_180"});
      nearPose(filter.select(rotate(axis,180,center), &first), first);
      configure(filter, {name+"_full"});
      for (int deg=0;deg<360;deg+=30) nearPose(filter.select(rotate(axis,static_cast<float>(deg),center), &first), first);
    }
    // Explicit angles do not silently expand into a cyclic group.
    configure(filter, {"z_90"});
    const auto still_rotated = filter.select(rotate(2,90,center), &first);
    require(std::abs(still_rotated(0,0))<1e-5f,"z_90 must not imply z_270");
    configure(filter, {"z_90", "z_180", "z_270"});
    nearPose(filter.select(rotate(2,90,center), &first), first);
    configure(filter, {"x_180", "y_180"});
    nearPose(filter.select(flipped, &first), first); // Rx(pi)*Ry(pi) gives Rz(pi).
    configure(filter, {"z_-180", "z_540", "z_180"});
    nearPose(filter.select(flipped, &first), first);

    for (const auto& bad : {"z", "a_180", "z_", "z_30junk", "z_nan", "z_inf", "z_1e309", "", "Z_180", "z_ full"}) {
      rejects([&] { configure(filter, {bad}); });
      nearPose(filter.select(flipped, &first), first); // Previous configuration survives.
    }
    Mat4f invalid = first; invalid(3,0)=1; rejects([&] { filter.select(invalid, &first); });
    invalid=first; invalid(0,0)=-1; rejects([&] { SymmetryFilter::validatePose(invalid); });
    invalid=first; invalid(0,0)=std::numeric_limits<float>::quiet_NaN(); rejects([&] { SymmetryFilter::validatePose(invalid); });
    invalid=first; invalid(1,1)=2; rejects([&] { SymmetryFilter::validatePose(invalid); });
    configure(filter, {}); require(!filter.enabled(),"Empty axes must disable");

    std::mt19937 rng(42); std::uniform_real_distribution<float> angles(-180,180), positions(-1,1);
    for (int i=0;i<200;++i) {
      const Vec3f c{positions(rng),positions(rng),positions(rng)};
      SymmetryFilter f(c); configure(f,{"z_180"});
      Mat4f T=product(rotate(0,angles(rng)),product(rotate(1,angles(rng)),rotate(2,angles(rng))));
      T(0,3)=positions(rng);T(1,3)=positions(rng);T(2,3)=positions(rng);
      nearPose(f.select(product(T,rotate(2,180,c)),&T),T);
    }
    std::cout << "PASS: symmetry axes, original-frame geometry, disable/anchor semantics, atomic validation, 200 random non-origin meshes\n";
    return 0;
  } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
