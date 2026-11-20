#!/usr/bin/env bash
# Run eval_libero.py in the LIBERO environment (.env-libero) with headless EGL rendering.
# GPU=<id> selects the GPU (default 0).
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/third_party/LIBERO:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PWD/third_party/libero_cfg"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="${GPU:-0}" MUJOCO_EGL_DEVICE_ID="${GPU:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
exec .env-libero/bin/python -u eval_libero.py "$@" 2> >(grep -vE "robosuite WARNING|EGLError|egl_context|glCheckError|OpenGL/error|Exception ignored|Traceback \(most recent call last\):$|^\s+(File|\^|self\.|raise|EGL)" >&2)
