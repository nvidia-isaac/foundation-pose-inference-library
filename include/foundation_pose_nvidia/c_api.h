/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file c_api.h
 * @brief Stable C ABI for the NVIDIA FoundationPose runtime — 6-DoF object pose
 *        estimation (register) and tracking from RGB-D.
 *
 * This is the recommended integration surface for C, C++, and any FFI caller
 * (Python/ctypes, Rust, Go, ...). It exposes opaque handles, POD option/result
 * structs, and explicit per-frame calls. The underlying engine runs on the GPU
 * (nvdiffrast CUDA rasterizer + TensorRT FP32 inference).
 *
 * @see fp_create, fp_register_frame, fp_track_frame
 */

/**
 * @mainpage FoundationPose Runtime — C API Getting Started
 *
 * @section fp_intro What this is
 *
 * The FoundationPose runtime estimates and tracks the 6-DoF pose (a 4x4 SE(3)
 * transform) of a **known object** in an RGB-D stream. You give it a CAD mesh
 * (model-based) or a handful of reference RGB-D views (model-free) once, then
 * feed it frames:
 *
 * - **Register** (`fp_register_frame`) — the expensive one-shot global pose search
 *   used on first sight or after tracking loss. Requires an object **mask**.
 * - **Track** (`fp_track_frame`) — the cheap, high-rate per-frame refinement seeded
 *   from the previous pose. No mask needed.
 *
 * One estimator handle owns one object's GPU resources (TensorRT engines/contexts,
 * uploaded mesh, and a persistent workspace). Create one handle per object you want
 * resident.
 *
 * @section fp_lifecycle The call sequence
 *
 * @code
 *   fp_default_config()      // 1. start from defaults
 *   fp_create()              // 2. load CAD mesh + build the estimator
 *        (or fp_create_model_free() for reference-image mode)
 *   fp_prepare()             // 3. (optional) pre-build TensorRT contexts so the
 *                            //    first measured frame is steady-state
 *   fp_register_frame()      // 4. first frame (or re-acquire): needs a mask
 *   fp_track_frame()         // 5. every subsequent frame: no mask, high-rate
 *        ... repeat track ...
 *   fp_destroy()             // 6. release all resources
 * @endcode
 *
 * @section fp_minimal Minimal example
 *
 * @code{.c}
 * #include "foundation_pose_nvidia/c_api.h"
 * #include <stdio.h>
 *
 * int main(void) {
 *   char err[512] = {0};
 *
 *   // 1. Defaults, then point at the ONNX weights + a CAD mesh.
 *   fp_config_t cfg;
 *   fp_default_config(&cfg);
 *
 *   fp_create_options_t opt = {0};
 *   opt.cad_path         = "/models/obj.obj";   // OBJ or PLY
 *   opt.mesh_unit_scale  = 1.0f;                 // 0.001f if the mesh is in millimeters
 *   opt.refine_model_path= "/weights/refiner_net.onnx";
 *   opt.score_model_path = "/weights/score_net.onnx";
 *   opt.engine_cache_dir = "engine_cache";       // TensorRT plans cached here
 *
 *   // 2. Build the estimator (NULL => read `err`).
 *   fp_handle_t* h = fp_create(&opt, &cfg, err, sizeof err);
 *   if (!h) { fprintf(stderr, "fp_create failed: %s\n", err); return 1; }
 *
 *   // 3. Optional: pre-build TensorRT contexts for the register batch size.
 *   fp_prepare(h, cfg.n_hypotheses, err, sizeof err);
 *
 *   // 4. Register on the first frame (RGB-D + intrinsics + object mask).
 *   fp_image_view_t frame = {0};
 *   frame.width = W; frame.height = H;
 *   frame.rgb_u8      = rgb;          // W*H*3 bytes, HWC RGB
 *   frame.depth_m     = depth;        // W*H float32, meters
 *   frame.mask_u8     = mask;         // W*H uint8, non-zero = object
 *   frame.k_row_major = K;            // 3x3 row-major float32
 *
 *   fp_pose_result_t pose;
 *   if (fp_register_frame(h, &frame, -1, -1, &pose, err, sizeof err) != 0) {
 *     fprintf(stderr, "register failed: %s\n", err); return 1;
 *   }
 *   // pose.pose_row_major is the object-to-camera 4x4 (row-major); pose.score is its confidence.
 *
 *   // 5. Track subsequent frames (no mask required).
 *   for (int t = 0; t < num_frames; ++t) {
 *     frame.rgb_u8 = rgb_t; frame.depth_m = depth_t; frame.mask_u8 = NULL;
 *     fp_track_frame(h, &frame, -1, &pose, err, sizeof err);
 *   }
 *
 *   // 6. Clean up.
 *   fp_destroy(h);
 *   return 0;
 * }
 * @endcode
 *
 * @section fp_io Input / output conventions
 *
 * - **RGB**: `HWC uint8`, `width * height * 3` bytes.
 * - **Depth**: `HW float32`, in **meters** (convert `uint16` mm cameras with `* 0.001`).
 * - **Mask** (register only): `HW uint8`, non-zero marks object pixels.
 * - **Intrinsics `K`**: `3x3` row-major float32 (`fx, 0, cx, 0, fy, cy, 0, 0, 1`).
 * - **Output pose**: `fp_pose_result_t::pose_row_major`, a `4x4` row-major SE(3)
 *   **object-to-camera** transform (translation in meters). `score` is the ScoreNet
 *   confidence for register; track sets it to `1`.
 * - Frame buffers (`rgb_u8`, `depth_m`, `mask_u8`) may be **host or CUDA
 *   device/managed** pointers — the library detects the location via CUDA
 *   unified virtual addressing and copies accordingly (device input avoids the
 *   host-to-device transfer; layouts are identical either way). Device pointers
 *   must be valid on (or peer-accessible from) the estimator's device, and only
 *   need to stay valid for the duration of the call. `k_row_major` and the
 *   model-free reference views (::fp_reference_image_view_t) are host-only.
 *
 * @section fp_errors Error handling
 *
 * - `fp_create` / `fp_create_model_free` return the handle, or `NULL` on failure.
 * - `fp_prepare`, `fp_register_frame`, `fp_track_frame`, `fp_synchronize_device`
 *   return `0` on success and non-zero on failure.
 * - On any failure a human-readable message is written into the caller-provided
 *   `error_buffer` (pass a buffer of a few hundred bytes; may be `NULL` to ignore).
 *
 * @section fp_threading Threading & resources
 *
 * Each handle owns one CUDA stream and one persistent workspace; calls on a single
 * handle are serialized on that stream and are **not** internally thread-safe — use
 * one handle per thread/object. Memory grows per resident handle (engines + mesh +
 * workspace), so create/destroy per object as your memory budget requires.
 *
 * @section fp_more Build, test, and benchmark
 *
 * See the repository `README.md` for containerized build (`docker/docker_build.sh`),
 * the CLI smoke test, and the BOP YCB-V benchmark harness; see `doc/ARCHITECTURE.md`
 * for the internal pipeline and the swappable renderer/inference seams.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#ifdef FOUNDATION_POSE_NVIDIA_EXPORTS
#define FP_API __declspec(dllexport)
#else
#define FP_API __declspec(dllimport)
#endif
#else
#define FP_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Opaque estimator handle. Create with ::fp_create / ::fp_create_model_free,
 *  release with ::fp_destroy. */
typedef struct fp_handle fp_handle_t;

/**
 * @brief Construction options: model paths, device, and TensorRT tensor names.
 *
 * Zero-initialize (`= {0}`) and set at least @ref cad_path (model-based),
 * @ref refine_model_path, @ref score_model_path, and @ref engine_cache_dir. The
 * `*_name` fields default to the deployable ONNX tensor names when left `NULL`.
 */
typedef struct fp_create_options {
  const char* cad_path;          /**< Path to the CAD mesh (OBJ/PLY). Unused for model-free. */
  int device_id;                 /**< CUDA device index (0 by default). */
  float mesh_unit_scale;         /**< Mesh-to-meters scale: `1.0` for meters, `0.001` for mm (YCB-V PLY). */
  const char* refine_model_path; /**< Path to the RefineNet ONNX (`refiner_net.onnx`). */
  const char* score_model_path;  /**< Path to the ScoreNet ONNX (`score_net.onnx`). */
  const char* engine_cache_dir;  /**< Directory for cached TensorRT plans (created if needed). */
  const char* refine_model_name; /**< Engine-cache key for the refiner (optional; default "refine"). */
  const char* score_model_name;  /**< Engine-cache key for the scorer (optional; default "score"). */
  const char* rendered_input_name; /**< RefineNet/ScoreNet "rendered" input tensor name (default "inputA"). */
  const char* observed_input_name; /**< RefineNet/ScoreNet "observed" input tensor name (default "inputB"). */
  const char* refine_translation_output_name; /**< Refiner translation-delta output name (default "trans"). */
  const char* refine_rotation_output_name;    /**< Refiner rotation-delta output name (default "rot"). */
  const char* score_output_name;              /**< Scorer output tensor name (default "score"). */
} fp_create_options_t;

/**
 * @brief TensorRT compute precision for the RefineNet/ScoreNet engines.
 *
 * - ::FP_PRECISION_FP32 — strict IEEE FP32 (TF32 disabled); most accurate and
 *   deterministic across GPUs. Recommended when comparing results across x86 and
 *   Jetson, or when accuracy regressions are suspected.
 * - ::FP_PRECISION_TF32 — allow TF32 tensor-core tactics (TensorRT's own default).
 * - ::FP_PRECISION_FP16 / ::FP_PRECISION_BF16 — allow reduced precision (faster,
 *   lower accuracy).
 */
typedef enum fp_precision {
  FP_PRECISION_FP32 = 0,
  FP_PRECISION_TF32 = 1,
  FP_PRECISION_FP16 = 2,
  FP_PRECISION_BF16 = 3
} fp_precision_t;

/**
 * @brief Algorithm/runtime configuration. Fill with ::fp_default_config first, then
 *        override only what you need.
 */
typedef struct fp_config {
  int n_hypotheses;      /**< Register pose hypotheses (rotation grid size), default 252. */
  int n_refine_iters;    /**< RefineNet iterations in register mode, default 5. */
  int n_track_iters;     /**< RefineNet iterations in track mode, default 2. */
  float crop_ratio;      /**< Crop box size as a multiple of object diameter, default 1.2. */
  int input_width;       /**< Network crop width, default 160. */
  int input_height;      /**< Network crop height, default 160. */
  int max_image_width;   /**< Max supported frame width (sizes the workspace). */
  int max_image_height;  /**< Max supported frame height (sizes the workspace). */
  int capture_cuda_graph;/**< Non-zero: capture/replay the refinement loop as a CUDA graph (mutually exclusive with micro-batching; see ::batch_size). */
  int model_free_sample_stride;       /**< Model-free: pixel stride when back-projecting reference views. */
  int model_free_max_vertices;        /**< Model-free: cap on reconstructed mesh vertices. */
  float model_free_depth_edge_threshold; /**< Model-free: depth-discontinuity edge threshold (meters). */
  int tensorrt_precision;             /**< TensorRT engine precision; see ::fp_precision_t. Default TF32; use FP32 for strict accuracy. */
  int batch_size;        /**< RefineNet mini-batch chunk size for TensorRT execution, default 252 (must divide n_hypotheses; ScoreNet is not micro-batched). */
} fp_config_t;

/**
 * @brief One RGB-D frame passed to ::fp_register_frame / ::fp_track_frame.
 *
 * @ref rgb_u8, @ref depth_m, and @ref mask_u8 may each point to **host or CUDA
 * device/managed memory** (detected via unified virtual addressing; same layout
 * either way). Device pointers must be valid on the estimator's device and only
 * need to remain valid for the duration of the call. @ref k_row_major is
 * host-only. @ref mask_u8 is required for register and ignored (may be `NULL`)
 * for track.
 */
typedef struct fp_image_view {
  int width;                /**< Frame width in pixels. */
  int height;               /**< Frame height in pixels. */
  const uint8_t* rgb_u8;    /**< HWC RGB, `width * height * 3` bytes (host or device). */
  const float* depth_m;     /**< HW float32 depth in **meters** (host or device). */
  const uint8_t* mask_u8;   /**< HW uint8 object mask (non-zero = object; host or device). Required for register; optional for track. */
  const float* k_row_major; /**< 3x3 row-major camera intrinsics (float32, host memory). */
} fp_image_view_t;

/**
 * @brief One reference view for model-free construction (::fp_create_model_free).
 *
 * Unlike ::fp_image_view_t, all pointers here must be **host** memory (the mesh
 * reconstruction runs on the CPU at construction time).
 */
typedef struct fp_reference_image_view {
  int width;                 /**< View width in pixels. */
  int height;                /**< View height in pixels. */
  const uint8_t* rgb_u8;     /**< HWC RGB, `width * height * 3` bytes. */
  const float* depth_m;      /**< HW float32 depth in **meters**. */
  const uint8_t* mask_u8;    /**< HW uint8 object mask (non-zero = object). */
  const float* k_row_major;  /**< 3x3 row-major camera intrinsics (float32). */
  const float* camera_to_world_row_major; /**< 4x4 row-major camera-to-world (float32); `NULL` => identity. */
} fp_reference_image_view_t;

/**
 * @brief Estimator output: the pose and its confidence score.
 */
typedef struct fp_pose_result {
  float pose_row_major[16]; /**< 4x4 row-major SE(3) object-to-camera transform (translation in meters). */
  float score;              /**< ScoreNet confidence (register); track returns `1`. */
} fp_pose_result_t;

/**
 * @brief Populate @p config with the recommended default values.
 * @param[out] config Config struct to fill. Call this before overriding fields.
 */
FP_API void fp_default_config(fp_config_t* config);

/**
 * @brief Create a **model-based** estimator from a CAD mesh.
 * @param[in]  options            Model paths, device, and tensor names (see ::fp_create_options_t).
 * @param[in]  config             Runtime configuration (see ::fp_default_config).
 * @param[out] error_buffer       Optional buffer for a failure message; may be `NULL`.
 * @param[in]  error_buffer_size  Size of @p error_buffer in bytes.
 * @return A handle to destroy with ::fp_destroy, or `NULL` on failure (see @p error_buffer).
 */
FP_API fp_handle_t* fp_create(const fp_create_options_t* options,
                              const fp_config_t* config,
                              char* error_buffer,
                              size_t error_buffer_size);

/**
 * @brief Create a **model-free** estimator by reconstructing a mesh from reference views.
 * @param[in]  references         Array of reference RGB-D-mask views.
 * @param[in]  reference_count    Number of entries in @p references.
 * @param[in]  options            Model paths, device, and tensor names (fp_create_options_t::cad_path is unused).
 * @param[in]  config             Runtime configuration.
 * @param[out] error_buffer       Optional buffer for a failure message; may be `NULL`.
 * @param[in]  error_buffer_size  Size of @p error_buffer in bytes.
 * @return A handle to destroy with ::fp_destroy, or `NULL` on failure.
 */
FP_API fp_handle_t* fp_create_model_free(const fp_reference_image_view_t* references,
                                         size_t reference_count,
                                         const fp_create_options_t* options,
                                         const fp_config_t* config,
                                         char* error_buffer,
                                         size_t error_buffer_size);

/**
 * @brief Pre-build the TensorRT execution contexts for a given batch size.
 *
 * Optional but recommended for latency measurement: it moves the one-time engine
 * build/deserialization out of the first ::fp_register_frame / ::fp_track_frame call.
 * Prepare the register batch (e.g. `n_hypotheses`) and/or the track batch (`1`).
 *
 * @param[in]  handle             Estimator handle.
 * @param[in]  batch_size         Batch size to prepare contexts for.
 * @param[out] error_buffer       Optional failure message buffer; may be `NULL`.
 * @param[in]  error_buffer_size  Size of @p error_buffer in bytes.
 * @return `0` on success, non-zero on failure.
 */
FP_API int fp_prepare(fp_handle_t* handle,
                      int batch_size,
                      char* error_buffer,
                      size_t error_buffer_size);

/**
 * @brief Destroy an estimator handle and free all its GPU/host resources.
 * @param[in] handle Handle from ::fp_create / ::fp_create_model_free. `NULL` is a no-op.
 */
FP_API void fp_destroy(fp_handle_t* handle);

/**
 * @brief Register (initial global pose estimate) on one RGB-D frame.
 *
 * Runs the full hypothesis search + refinement + scoring. Requires
 * fp_image_view_t::mask_u8. The result also becomes the seed for subsequent
 * ::fp_track_frame calls.
 *
 * @param[in]  handle             Estimator handle.
 * @param[in]  image              RGB-D frame with mask and intrinsics.
 * @param[in]  n_refine           RefineNet iterations, or `-1` to use `config.n_refine_iters`.
 * @param[in]  n_hypotheses       Hypothesis count, or `-1` to use `config.n_hypotheses`.
 * @param[out] result             Best pose + score.
 * @param[out] error_buffer       Optional failure message buffer; may be `NULL`.
 * @param[in]  error_buffer_size  Size of @p error_buffer in bytes.
 * @return `0` on success, non-zero on failure.
 */
FP_API int fp_register_frame(fp_handle_t* handle,
                             const fp_image_view_t* image,
                             int n_refine,
                             int n_hypotheses,
                             fp_pose_result_t* result,
                             char* error_buffer,
                             size_t error_buffer_size);

/**
 * @brief Track (refine from the previous pose) on one RGB-D frame.
 *
 * Cheap, high-rate refinement seeded from the last register/track pose; no mask
 * required. A prior successful ::fp_register_frame is required to seed tracking.
 *
 * @param[in]  handle             Estimator handle.
 * @param[in]  image              RGB-D frame with intrinsics (`mask_u8` may be `NULL`).
 * @param[in]  n_refine           RefineNet iterations, or `-1` to use `config.n_track_iters`.
 * @param[out] result             Refined pose (`score` is `1`).
 * @param[out] error_buffer       Optional failure message buffer; may be `NULL`.
 * @param[in]  error_buffer_size  Size of @p error_buffer in bytes.
 * @return `0` on success, non-zero on failure.
 */
FP_API int fp_track_frame(fp_handle_t* handle,
                          const fp_image_view_t* image,
                          int n_refine,
                          fp_pose_result_t* result,
                          char* error_buffer,
                          size_t error_buffer_size);

/**
 * @defgroup fp_group Multi-object estimator group
 *
 * A group owns one estimator per object and evaluates them against the **same
 * RGB-D frame** concurrently (one worker thread per object, each estimator on
 * its own CUDA stream). This is the library-level counterpart of running
 * multiple FoundationPose pipelines in parallel behind one shared detector:
 * the caller runs detection/segmentation once per frame, hands each object's
 * mask to the group, and gets all poses back in a single call.
 *
 * TensorRT engines are shared process-wide across estimators (engines are
 * mesh-independent), so per-object marginal GPU cost is contexts + mesh +
 * workspace, not a full engine copy.
 *
 * Per-object outcomes are reported through the `statuses` array
 * (::FP_OBJECT_OK / ::FP_OBJECT_FAILED / ::FP_OBJECT_SKIPPED); the call itself
 * returns the number of failed objects.
 * @{
 */

/** @brief Opaque multi-object estimator group. Create with ::fp_group_create,
 *  release with ::fp_group_destroy. */
typedef struct fp_group fp_group_t;

/** @brief Per-object status: the object was evaluated successfully. */
#define FP_OBJECT_OK 0
/** @brief Per-object status: the object was evaluated and failed (see error buffer). */
#define FP_OBJECT_FAILED 1
/** @brief Per-object status: the object was skipped (no mask / inactive). */
#define FP_OBJECT_SKIPPED 2

/**
 * @brief Create a group of model-based estimators, one per object.
 *
 * Objects are created sequentially; on any failure the whole group is torn
 * down and `NULL` is returned.
 *
 * @param[in]  objects            Array of per-object create options (CAD path etc.).
 * @param[in]  object_count       Number of entries in @p objects.
 * @param[in]  config             Runtime configuration shared by all objects.
 * @param[out] error_buffer       Optional failure message buffer; may be `NULL`.
 * @param[in]  error_buffer_size  Size of @p error_buffer in bytes.
 * @return A group to destroy with ::fp_group_destroy, or `NULL` on failure.
 */
FP_API fp_group_t* fp_group_create(const fp_create_options_t* objects,
                                   size_t object_count,
                                   const fp_config_t* config,
                                   char* error_buffer,
                                   size_t error_buffer_size);

/** @brief Number of objects in the group. */
FP_API size_t fp_group_size(const fp_group_t* group);

/**
 * @brief Borrow the single-object handle for object @p index.
 *
 * Use this to drive one object individually (e.g. ::fp_register_frame to
 * re-acquire a lost object) with the existing single-object API. The handle is
 * owned by the group: do **not** pass it to ::fp_destroy, and do not call into
 * it concurrently with a group-level call.
 *
 * @return Borrowed handle, or `NULL` if @p index is out of range.
 */
FP_API fp_handle_t* fp_group_handle(fp_group_t* group, size_t index);

/**
 * @brief Pre-build TensorRT contexts for every object in the group.
 * @param[in]  group              Estimator group.
 * @param[in]  batch_size         Batch size to prepare (e.g. `n_hypotheses` and/or `1`).
 * @param[out] error_buffer       Optional failure message buffer; may be `NULL`.
 * @param[in]  error_buffer_size  Size of @p error_buffer in bytes.
 * @return `0` on success, otherwise the number of objects that failed.
 */
FP_API int fp_group_prepare(fp_group_t* group,
                            int batch_size,
                            char* error_buffer,
                            size_t error_buffer_size);

/**
 * @brief Register every masked object against one shared RGB-D frame, concurrently.
 *
 * fp_image_view_t::mask_u8 on @p image is ignored; per-object masks come from
 * @p masks (typically produced by one shared detector/segmenter). An object
 * with a `NULL` mask entry is skipped (::FP_OBJECT_SKIPPED) and its result is
 * left untouched.
 *
 * @param[in]  group              Estimator group.
 * @param[in]  image              Shared RGB-D frame with intrinsics.
 * @param[in]  masks              Array of `fp_group_size()` per-object mask pointers
 *                                (`width * height` uint8, non-zero = object).
 * @param[in]  n_refine           RefineNet iterations, or `-1` for `config.n_refine_iters`.
 * @param[in]  n_hypotheses       Hypothesis count, or `-1` for `config.n_hypotheses`.
 * @param[out] results            Array of `fp_group_size()` results.
 * @param[out] statuses           Optional array of `fp_group_size()` per-object statuses; may be `NULL`.
 * @param[out] error_buffer       Optional buffer; receives the first failure message.
 * @param[in]  error_buffer_size  Size of @p error_buffer in bytes.
 * @return `0` if no object failed, otherwise the number of failed objects.
 */
FP_API int fp_group_register_frame(fp_group_t* group,
                                   const fp_image_view_t* image,
                                   const uint8_t* const* masks,
                                   int n_refine,
                                   int n_hypotheses,
                                   fp_pose_result_t* results,
                                   int* statuses,
                                   char* error_buffer,
                                   size_t error_buffer_size);

/**
 * @brief Track every active object on one shared RGB-D frame, concurrently.
 *
 * Each tracked object must have been seeded by a prior successful register
 * (group-level or via ::fp_group_handle).
 *
 * @param[in]  group              Estimator group.
 * @param[in]  image              Shared RGB-D frame with intrinsics (mask ignored).
 * @param[in]  active             Optional array of `fp_group_size()` flags; zero skips
 *                                that object (::FP_OBJECT_SKIPPED). `NULL` tracks all.
 * @param[in]  n_refine           RefineNet iterations, or `-1` for `config.n_track_iters`.
 * @param[out] results            Array of `fp_group_size()` results.
 * @param[out] statuses           Optional array of `fp_group_size()` per-object statuses; may be `NULL`.
 * @param[out] error_buffer       Optional buffer; receives the first failure message.
 * @param[in]  error_buffer_size  Size of @p error_buffer in bytes.
 * @return `0` if no object failed, otherwise the number of failed objects.
 */
FP_API int fp_group_track_frame(fp_group_t* group,
                                const fp_image_view_t* image,
                                const int* active,
                                int n_refine,
                                fp_pose_result_t* results,
                                int* statuses,
                                char* error_buffer,
                                size_t error_buffer_size);

/**
 * @brief Destroy a group and every estimator it owns. `NULL` is a no-op.
 *
 * Invalidates all handles borrowed through ::fp_group_handle.
 */
FP_API void fp_group_destroy(fp_group_t* group);

/** @} */

/**
 * @brief Block until all queued GPU work on the calling context completes.
 * @param[out] error_buffer       Optional failure message buffer; may be `NULL`.
 * @param[in]  error_buffer_size  Size of @p error_buffer in bytes.
 * @return `0` on success, non-zero on failure.
 */
FP_API int fp_synchronize_device(char* error_buffer, size_t error_buffer_size);

/**
 * @brief Return a static human-readable build/version string.
 * @return A NUL-terminated string owned by the library (do not free).
 */
FP_API const char* fp_build_info(void);

#ifdef __cplusplus
}
#endif
