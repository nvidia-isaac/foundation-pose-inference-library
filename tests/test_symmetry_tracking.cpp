/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Real CUDA pipeline/state with deterministic inference and a recording renderer.
// No TensorRT engines, weights, or customer data are required.
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>

#include "foundation_pose_nvidia/foundation_pose.hpp"

using namespace foundation_pose_nvidia;

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void nearPose(const Mat4f& a, const Mat4f& b) {
  for (int i=0;i<16;++i) require(std::abs(a.values[i]-b.values[i])<3e-6f,"Pose/seed mismatch");
}
struct RecordingRenderer : IRenderer {
  Mat4f last_seed;
  int pixels = 160*160;
  void render(const DeviceMesh&, const float* poses, const float*, int batch,
              CameraIntrinsics, int, int, float3* rgb, float3* xyz, cudaStream_t stream) override {
    checkCuda(cudaMemcpyAsync(last_seed.values.data(), poses, 16*sizeof(float), cudaMemcpyDeviceToHost,stream),"read seed");
    checkCuda(cudaStreamSynchronize(stream),"read seed sync");
    checkCuda(cudaMemsetAsync(rgb,0,batch*pixels*sizeof(float3),stream),"zero render rgb");
    checkCuda(cudaMemsetAsync(xyz,0,batch*pixels*sizeof(float3),stream),"zero render xyz");
  }
};
struct DeterministicInference : IInferenceRunner {
  RecordingRenderer* renderer;
  bool flip = false;
  explicit DeterministicInference(RecordingRenderer* r) : renderer(r) {}
  void enqueueRefine(const float*,const float*,float* dt,float* dr,int batch,cudaStream_t stream) override {
    require(batch==1,"Test uses one hypothesis");
    float rotation[3]{};
    if(flip) {
      // Object-z rotation, expressed in camera axes. Test normalizer = 2*pi;
      // tanh(prediction)*normalizer then produces a pi-radian update.
      for(int i=0;i<3;++i) rotation[i]=std::atanh(.5f*renderer->last_seed(i,2));
    }
    checkCuda(cudaMemsetAsync(dt,0,3*sizeof(float),stream),"zero translation delta");
    checkCuda(cudaMemcpyAsync(dr,rotation,sizeof(rotation),cudaMemcpyHostToDevice,stream),"rotation delta");
    checkCuda(cudaStreamSynchronize(stream),"delta upload sync");
  }
  void enqueueScore(const float*,const float*,float* scores,int batch,cudaStream_t stream) override {
    std::vector<float> values(batch,.75f);
    checkCuda(cudaMemcpyAsync(scores,values.data(),batch*sizeof(float),cudaMemcpyHostToDevice,stream),"score");
    checkCuda(cudaStreamSynchronize(stream),"score upload sync");
  }
};
Mat4f centered(const Mat4f& pose,Vec3f c) { return SymmetryFilter(c).toCenteredPose(pose); }
Mat4f opposite(Mat4f T, Vec3f c) {
  const Mat4f physical=centered(T,c);
  for(int r=0;r<3;++r) { T(r,0)=-T(r,0);T(r,1)=-T(r,1);T(r,3)=physical(r,3)-T(r,0)*c.x-T(r,1)*c.y-T(r,2)*c.z; }
  return T;
}
}  // namespace

int main() {
  int devices=0;
  if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0) { std::cout<<"SKIP: CUDA device unavailable\n";return 77; }
  try {
    Config config; config.n_hypotheses=1;config.batch_size=1;config.max_image_width=160;config.max_image_height=160;
    config.rotation_normalizer_rad=2*std::numbers::pi_v<float>;
    Mesh geometry;
    geometry.vertices={{0,0,0},{.12f,0,0},{0,.08f,0},{0,0,.015f}};
    geometry.faces={{0,1,2},{0,3,1},{0,2,3},{1,3,2}};
    computeVertexNormals(geometry);
    auto mesh=preprocessMesh(geometry,config);const auto center=mesh.center;
    auto renderer=std::make_unique<RecordingRenderer>();auto* record=renderer.get();
    auto inference=std::make_unique<DeterministicInference>(record);auto* control=inference.get();
    FoundationPose estimator(std::move(mesh),{},std::move(inference),std::move(renderer),config);
    std::vector<std::uint8_t> rgb(160*160*3,128),mask(160*160,255);
    std::vector<float> depth(160*160,.7f);const CameraIntrinsics K{180,180,80,80};
    auto register_frame=[&] {return estimator.registerFrame(rgb.data(),depth.data(),mask.data(),160,160,K,0,1);};
    auto track_frame=[&](int iters=0) {return estimator.trackFrame(rgb.data(),depth.data(),160,160,K,iters);};
    const auto raw=register_frame();
    estimator.setSymmetryAxes({"z_180"});
    estimator.resetTracking();
    const auto first=register_frame();nearPose(first.pose,raw.pose);require(first.score==raw.score,"Filter altered score");
    const auto anchor=opposite(raw.pose,center);estimator.resetTracking(anchor);
    const auto registered=register_frame();nearPose(registered.pose,anchor);require(registered.score==.75f,"Registration score altered");
    nearPose(track_frame().pose,anchor); // Reads actual GPU seed, no refinement.
    control->flip=true;
    const auto tracked=track_frame(1);nearPose(tracked.pose,anchor);nearPose(record->last_seed,centered(anchor,center));
    require(tracked.score==1.f,"Tracking score altered");
    control->flip=false;nearPose(track_frame().pose,anchor); // Filtered tracking output fed back to GPU.
    estimator.setSymmetryAxes({});nearPose(track_frame().pose,anchor);
    nearPose(register_frame().pose,raw.pose); // Disabled registration does not canonicalize.
    estimator.setSymmetryAxes({"z_180"});estimator.resetTracking(anchor);
    Mat4f bad=anchor;bad(0,0)=10;
    bool rejected=false;try {estimator.resetTracking(bad);} catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"Invalid reset accepted");nearPose(track_frame().pose,anchor);
    FoundationPose moved(std::move(estimator));nearPose(moved.trackFrame(rgb.data(),depth.data(),160,160,K,0).pose,anchor);
    moved.resetTracking();rejected=false;
    try {moved.trackFrame(rgb.data(),depth.data(),160,160,K,0);} catch(const std::exception&) {rejected=true;}
    require(rejected,"Cleared tracker accepted track without registration");
    std::cout<<"PASS: register/track branch correction, unchanged scores, centered GPU feedback, re-registration, disable, reset, move\n";
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
