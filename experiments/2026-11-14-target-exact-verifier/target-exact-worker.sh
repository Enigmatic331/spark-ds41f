#!/usr/bin/env bash
# Five-slot/4.8125-GiB-KV target-exact trial controller, adapted from the
# hash-recorded private exact-CED launcher. This is AI-assisted deployment glue;
# the engine and existing kernels remain imported Christopher/vLLM layers.
set -Eeuo pipefail
umask 077

action=${1:-status}
port=${2:-18888}
mm_encoder_tp_mode=${NATIVE_MM_ENCODER_TP_MODE:-weights}
vision_backend=${NATIVE_VISION_BACKEND:-tilelang}
spec_mode=${NATIVE_SPEC_MODE:-target-exact}
engram_projection_tp=${NATIVE_ENGRAM_PROJECTION_TP:-true}
cudagraph_mode=${NATIVE_CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}
moe_bf16=${NATIVE_MOE_BF16:-0}
root="$HOME/spark-ds41f-native-20261113"
runtime="$root/runtime"
pyroot="$runtime/site-packages"
state="$root/state"
logs="$root/logs"
pidfile="$state/candidate.pid"
portfile="$state/candidate.port"
identityfile="$state/candidate.identity"
lockfile="$state/controller-v2.lock"
model=DeepSeek-V4.1-Flash-1017-v4
revision=02d2ab346129098f9b3546143bb69944d88c7f977218627b916128cb9c1edfc9
sitecustomize_sha=71a731b4a7b86e0e581c6eee08062d81d43b4b14ab6abdcc54c60e10d721af78
ced_overlay_tag=accuracy-trace-target-exact-6ae956399820
ced_overlay="$root/diagnostics/$ced_overlay_tag"
ced_pyroot="$ced_overlay/site-packages"
ced_manifest_sha=7a5883cbb0ee59ee3113fc1f01f6c8e6c095d51f7c1c39b70e65f460673291b0
ced_source_sha=d7f04c2cf238099f223894e4a4a77a96b4b81b19a00056e870366bdfb8e1b3ff
ced_attention_sha=c5580b41ddef0d97269be07b1139376330ea8d95cd92b1506a4f6821032dff3e
ced_model_state_sha=46c5e570bd8fe5cd57daaab7677d481a150ff96f0d24cf77e633d3587f67318a
target_exact_config_sha=6983a9724d64f6654c64bb5b6e68af8978ff56de182e191e3bfd732bc9abe41e
target_exact_v1_sampler_sha=5acb2a52d68e3a68f8a6ad24d93ccf8c4ad015c0a7cb30aa2aa2b5474bd61018
target_exact_tilelang_sha=c65c43428cd317fe659b7c27af047e795a2e7f7e61b1e8f3b73a46d3dc676eb6
target_exact_sampler_sha=652e50aa5e0c0b56a22e28afaf5ca5286fd3b6d3428e31e80bd58b7a4e008f9f

die() { printf 'native-node: %s\n' "$*" >&2; exit 1; }

case "$(id -un)" in
  dgx02)
    rank=0
    control_ip=10.20.255.12
    checkpoint="$HOME/model-store/huggingface/hub/models--local--DeepSeek-V4.1-Flash-1017-v4/snapshots/$revision"
    peer_hcas='{"1":["rocep1s0f0","roceP2p1s0f0"],"3":["rocep1s0f1","roceP2p1s0f1"]}'
    roce_subnets='{"rocep1s0f0":"10.20.2.0/24","roceP2p1s0f0":"10.20.3.0/24","rocep1s0f1":"10.20.0.0/24","roceP2p1s0f1":"10.20.1.0/24"}'
    nccl_hcas='=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1'
    ;;
  dgx04)
    rank=1
    control_ip=10.20.255.14
    checkpoint="/srv/dsv41-weights/hub/models--local--DeepSeek-V4.1-Flash-1017-v4/snapshots/$revision"
    peer_hcas='{"0":["rocep1s0f1","roceP2p1s0f1"],"2":["rocep1s0f0","roceP2p1s0f0"]}'
    roce_subnets='{"rocep1s0f1":"10.20.2.0/24","roceP2p1s0f1":"10.20.3.0/24","rocep1s0f0":"10.20.4.0/24","roceP2p1s0f0":"10.20.5.0/24"}'
    nccl_hcas='=rocep1s0f1,roceP2p1s0f1,rocep1s0f0,roceP2p1s0f0'
    ;;
  dgx03)
    rank=2
    control_ip=10.20.255.13
    checkpoint="/srv/dsv41-weights/hub/models--local--DeepSeek-V4.1-Flash-1017-v4/snapshots/$revision"
    peer_hcas='{"1":["rocep1s0f1","roceP2p1s0f1"],"3":["rocep1s0f0","roceP2p1s0f0"]}'
    roce_subnets='{"rocep1s0f1":"10.20.4.0/24","roceP2p1s0f1":"10.20.5.0/24","rocep1s0f0":"10.20.6.0/24","roceP2p1s0f0":"10.20.7.0/24"}'
    nccl_hcas='=rocep1s0f1,roceP2p1s0f1,rocep1s0f0,roceP2p1s0f0'
    ;;
  dgx01)
    rank=3
    control_ip=10.20.255.11
    checkpoint="/srv/dsv41-weights/hub/models--local--DeepSeek-V4.1-Flash-1017-v4/snapshots/$revision"
    peer_hcas='{"2":["rocep1s0f1","roceP2p1s0f1"],"0":["rocep1s0f0","roceP2p1s0f0"]}'
    roce_subnets='{"rocep1s0f1":"10.20.6.0/24","roceP2p1s0f1":"10.20.7.0/24","rocep1s0f0":"10.20.0.0/24","roceP2p1s0f0":"10.20.1.0/24"}'
    nccl_hcas='=rocep1s0f1,roceP2p1s0f1,rocep1s0f0,roceP2p1s0f0'
    ;;
  *) die "unexpected user $(id -un)" ;;
esac

mkdir -p "$state" "$logs" "$root/cache" "$root/data/engram-packed-rank$rank"
exec 9>"$lockfile"
flock -x 9

read_pid() {
  [[ -f "$pidfile" && ! -L "$pidfile" ]] || return 1
  local value
  value=$(<"$pidfile")
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s\n' "$value"
}

prove_pid() {
  local pid=$1 cmd owner pgid exe start expected_pid expected_start expected_rank expected_port extra
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  owner=$(stat -c %u "/proc/$pid")
  [[ "$owner" == "$(id -u)" ]] || return 1
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
  [[ "$pgid" == "$pid" ]] || return 1
  exe=$(readlink -f "/proc/$pid/exe")
  [[ "$exe" == /usr/bin/python3.12 ]] || return 1
  grep -zFq "VLLM_HOST_IP=$control_ip" "/proc/$pid/environ" || return 1
  grep -zFq "PYTHONPATH=$pyroot/nvidia_cutlass_dsl/dsl_packages:$ced_pyroot:$pyroot" "/proc/$pid/environ" || return 1
  cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline")
  if [[ "$cmd" == *"vllm.entrypoints.cli.main serve $checkpoint"* && "$cmd" == *"--node-rank $rank"* ]]; then
    return 0
  fi
  [[ -f "$identityfile" && ! -L "$identityfile" ]] || return 1
  read -r expected_pid expected_start expected_rank expected_port extra <"$identityfile"
  [[ -z "${extra:-}" && "$expected_pid" == "$pid" && "$expected_rank" == "$rank" ]] || return 1
  [[ "$expected_port" == "$(cat "$portfile")" ]] || return 1
  start=$(awk '{print $22}' "/proc/$pid/stat")
  [[ "$start" == "$expected_start" ]] || return 1
  if [[ "$rank" == 0 ]]; then
    [[ "$cmd" == VLLM::APIServer* ]] || return 1
  else
    [[ "$cmd" == "VLLM::Worker_TP$rank"* ]] || return 1
  fi
}

write_identity() {
  local pid=$1 start temporary
  start=$(awk '{print $22}' "/proc/$pid/stat")
  temporary=$(mktemp "$state/candidate.identity.XXXXXX")
  printf '%s %s %s %s\n' "$pid" "$start" "$rank" "$(cat "$portfile")" >"$temporary"
  mv -f -- "$temporary" "$identityfile"
}

assert_stopped() {
  local proc owner candidate=0
  if read_pid >/dev/null 2>&1; then
    die "candidate PID record still exists; run stop before mutating its source"
  fi
  for proc in /proc/[1-9]*; do
    [[ -r "$proc/environ" ]] || continue
    owner=$(stat -c %u "$proc" 2>/dev/null) || continue
    [[ "$owner" == "$(id -u)" ]] || continue
    if grep -zFqx "PYTHONPATH=$pyroot/nvidia_cutlass_dsl/dsl_packages:$ced_pyroot:$pyroot" "$proc/environ" 2>/dev/null; then
      candidate=1
      printf 'native-node: candidate-owned process remains pid=%s\n' "${proc##*/}" >&2
    fi
  done
  (( candidate == 0 )) || die "candidate process remains without a PID record"
  printf 'rank=%s state=stopped-proven\n' "$rank"
}

adopt_running() {
  local pid
  pid=$(read_pid) || die "no candidate PID to adopt"
  [[ -r "/proc/$pid/cmdline" ]] || die "candidate PID is not running"
  write_identity "$pid"
  prove_pid "$pid" || { rm -f "$identityfile"; die "running PID failed candidate identity checks"; }
  printf 'rank=%s adopted verified running pid=%s\n' "$rank" "$pid"
}

status() {
  local pid
  if pid=$(read_pid) && prove_pid "$pid"; then
    printf 'rank=%s state=running pid=%s port=%s\n' "$rank" "$pid" "$(cat "$portfile" 2>/dev/null || printf unknown)"
    return 0
  fi
  printf 'rank=%s state=stopped\n' "$rank"
  return 1
}

stop_candidate() {
  local pid i owner_uid group_rows expected_pid expected_start expected_rank expected_port extra
  if ! pid=$(read_pid); then
    rm -f "$pidfile" "$portfile" "$identityfile"
    printf 'rank=%s already stopped\n' "$rank"
    return 0
  fi
  if [[ ! -e "/proc/$pid" ]]; then
    if ! kill -0 -- "-$pid" 2>/dev/null; then
      rm -f "$pidfile" "$portfile" "$identityfile"
      printf 'rank=%s cleaned stale exited pid=%s\n' "$rank" "$pid"
      return 0
    fi
    [[ -f "$identityfile" && ! -L "$identityfile" ]] ||
      die "leader exited but process group $pid remains without a verified identity record"
    read -r expected_pid expected_start expected_rank expected_port extra <"$identityfile"
    [[ -z "${extra:-}" && "$expected_pid" == "$pid" && "$expected_rank" == "$rank" ]] ||
      die "remaining process group $pid failed recorded identity checks"
    [[ -f "$portfile" && "$expected_port" == "$(cat "$portfile")" ]] ||
      die "remaining process group $pid failed recorded port checks"
    owner_uid=$(id -u)
    group_rows=$(ps -eo uid=,pgid=,pid=,stat= --no-headers | awk -v pgid="$pid" '$2 == pgid { print }')
    [[ -n "$group_rows" ]] || die "process group $pid was signallable but could not be enumerated"
    if awk -v uid="$owner_uid" '$1 != uid { bad=1 } END { exit bad ? 0 : 1 }' <<<"$group_rows"; then
      die "refusing to signal remaining process group $pid containing a foreign owner"
    fi
  else
    prove_pid "$pid" || die "refusing to signal unverified PID $pid"
  fi
  kill -TERM -- "-$pid"
  for i in $(seq 1 120); do
    if ! kill -0 -- "-$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 -- "-$pid" 2>/dev/null; then
    owner_uid=$(id -u)
    group_rows=$(ps -eo uid=,pgid=,pid=,stat= --no-headers | awk -v pgid="$pid" '$2 == pgid { print }')
    [[ -n "$group_rows" ]] || die "process group $pid was signallable but could not be enumerated"
    if awk -v uid="$owner_uid" '$1 != uid { bad=1 } END { exit bad ? 0 : 1 }' <<<"$group_rows"; then
      die "refusing to kill process group $pid containing a foreign owner"
    fi
    kill -KILL -- "-$pid"
    for i in $(seq 1 60); do
      if ! kill -0 -- "-$pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
  fi
  ! kill -0 -- "-$pid" 2>/dev/null || die "candidate process group $pid survived SIGKILL"
  rm -f "$pidfile" "$portfile" "$identityfile"
  printf 'rank=%s stopped pid=%s\n' "$rank" "$pid"
}

start_candidate() {
  [[ "$port" == 18888 || "$port" == 8888 ]] || die "port must be 18888 or 8888"
  [[ "$mm_encoder_tp_mode" == weights || "$mm_encoder_tp_mode" == data ]] ||
    die "NATIVE_MM_ENCODER_TP_MODE must be weights or data"
  [[ "$vision_backend" == tilelang || "$vision_backend" == torch ]] ||
    die "NATIVE_VISION_BACKEND must be tilelang or torch"
  [[ "$vision_backend" != torch || "$mm_encoder_tp_mode" == data ]] ||
    die "the torch vision backend requires NATIVE_MM_ENCODER_TP_MODE=data"
  [[ "$spec_mode" == dspark || "$spec_mode" == dspark-static || "$spec_mode" == target-exact || "$spec_mode" == off ]] ||
    die "NATIVE_SPEC_MODE must be dspark, dspark-static, target-exact, or off"
  [[ "$engram_projection_tp" == true || "$engram_projection_tp" == false ]] ||
    die "NATIVE_ENGRAM_PROJECTION_TP must be true or false"
  [[ "$cudagraph_mode" == FULL_AND_PIECEWISE || "$cudagraph_mode" == NONE ]] ||
    die "NATIVE_CUDAGRAPH_MODE must be FULL_AND_PIECEWISE or NONE"
  [[ "$moe_bf16" == 0 || "$moe_bf16" == 1 ]] || die "NATIVE_MOE_BF16 must be 0 or 1"
  [[ -d "$ced_pyroot/vllm" && ! -L "$ced_overlay" && ! -L "$ced_pyroot" && ! -L "$ced_pyroot/vllm" ]] || die "exact-CED overlay is missing or unsafe"
  [[ -f "$ced_overlay/ACCURACY_TRACE.sha256" && ! -L "$ced_overlay/ACCURACY_TRACE.sha256" ]] || die "exact-CED manifest is missing or unsafe"
  [[ "$(sha256sum "$ced_overlay/ACCURACY_TRACE.sha256" | cut -d' ' -f1)" == "$ced_manifest_sha" ]] || die "exact-CED manifest hash mismatch"
  (cd "$ced_overlay" && sha256sum -c ACCURACY_TRACE.sha256 --quiet) || die "exact-CED overlay hash mismatch"
  [[ "$(sha256sum "$ced_pyroot/vllm/models/deepseek_v4_1/ced.py" | cut -d' ' -f1)" == "$ced_source_sha" ]] || die "exact-CED source mismatch"
  [[ "$(sha256sum "$ced_pyroot/vllm/models/deepseek_v4_1/attention.py" | cut -d' ' -f1)" == "$ced_attention_sha" ]] || die "exact-CED attention mismatch"
  [[ "$(sha256sum "$ced_pyroot/vllm/models/deepseek_v4_1/nvidia/model_state.py" | cut -d' ' -f1)" == "$ced_model_state_sha" ]] || die "exact-CED model-state mismatch"
  [[ "$(sha256sum "$ced_pyroot/vllm/config/speculative.py" | cut -d' ' -f1)" == "$target_exact_config_sha" ]] || die "target-exact config mismatch"
  [[ "$(sha256sum "$ced_pyroot/vllm/v1/sample/rejection_sampler.py" | cut -d' ' -f1)" == "$target_exact_v1_sampler_sha" ]] || die "target-exact V1 sampler guard mismatch"
  [[ "$(sha256sum "$ced_pyroot/vllm/v1/worker/gpu/sample/tilelang_sampling.py" | cut -d' ' -f1)" == "$target_exact_tilelang_sha" ]] || die "target-exact TileLang source mismatch"
  [[ "$(sha256sum "$ced_pyroot/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py" | cut -d' ' -f1)" == "$target_exact_sampler_sha" ]] || die "target-exact sampler mismatch"
  unset VLLM_DS41_DISABLE_CED
  export VLLM_DS41_CED_EXACT_WARMUP=1
  local seqs=${1:-5}
  local chunk=8064
  local engram_config
  local compilation_config
  engram_config=$(printf '{"table_memory":"disk","disk_resident_scales":false,"projection_tp":%s}' \
    "$engram_projection_tp")
  compilation_config=$(printf '{"cudagraph_mode":"%s","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,3,4,6,8,12,16,20,24,28,32,40,48,56,64,72,80,88,96]}' \
    "$cudagraph_mode")
  local -a timing_args=()
  [[ "$seqs" == 5 ]] || die "final capacity profile requires exactly 5 slots"
  [[ "${NATIVE_LONG_PREFILL_TOKEN_THRESHOLD:-8064}" == 8064 ]] ||
    die "final capacity profile requires the retained 8064-token prefill chunk"
  timing_args=(--enable-per-request-metrics)
  if read_pid >/dev/null 2>&1; then
    if status >/dev/null 2>&1; then
      die "candidate already running"
    fi
    stop_candidate
  fi
  assert_stopped >/dev/null
  [[ -r "$checkpoint/config.json" ]] || die "checkpoint not readable: $checkpoint"
  [[ -r "$runtime/COMPILED-ARTIFACTS.sha256" ]] || die "runtime manifest missing"
  [[ "$(sha256sum "$pyroot/sitecustomize.py" | cut -d' ' -f1)" == "$sitecustomize_sha" ]] ||
    die "private Python isolation hook missing or modified"
  [[ "$(sha256sum "$runtime/python-include/python3.12/Python.h" | cut -d' ' -f1)" == 729ef157f6026e6e1b3104593f87dddc597c3b83b60c7c2965878c62a56c6f7d ]] ||
    die "private Python C headers missing or modified"
  [[ "$(sha256sum "$runtime/python-include/aarch64-linux-gnu/python3.12/pyconfig.h" | cut -d' ' -f1)" == c0dc659a80f55ab7b441abf018a9d05e8961bbb2d003e1c8757570b92e21f5ca ]] ||
    die "private Python multiarch C headers missing or modified"
  [[ ! -e "$pyroot/nvidia/nccl/lib/libnccl.so.2.upstream-2.29.7" ]] ||
    die "legacy NCCL backup is still visible to PyTorch"
  [[ "$(sha256sum "$runtime/preserved-upstream-nccl/libnccl.so.2.29.7" | cut -d' ' -f1)" == cc01f863461ea28f3ecc4390716a2d7a5aca9bc00e04eeb6fc7a5c2e666b3dfb ]] ||
    die "preserved upstream NCCL missing or modified"
  [[ "$(sha256sum "$pyroot/vllm/entrypoints/openai/chat_completion/protocol.py" | cut -d' ' -f1)" == 957b2ecf0742a53debaa5465fe92ef56bc5e93396db3bc521ade137d954ea51e ]] ||
    die "chat API compatibility overlay missing or modified"
  [[ "$(sha256sum "$pyroot/vllm/entrypoints/generate/base/protocol.py" | cut -d' ' -f1)" == e5d674a1679d9562b81d8def59c465b8b343b02178719f0b2cf10d5d6a011a9c ]] ||
    die "stream API compatibility overlay missing or modified"
  [[ "$(sha256sum "$pyroot/vllm/multimodal/media/connector.py" | cut -d' ' -f1)" == fdd722e172d644d077dc21fb35bcab2be2f70c814ddbf47c24d287af3724a80e ]] ||
    die "media request policy overlay missing or modified"
  [[ "$(sha256sum "$pyroot/native_health.py" | cut -d' ' -f1)" == feec3a6fec31e624b61d57d16eff3a278aae632d5aeae150069bae5be83778e4 ]] ||
    die "health adapter missing or modified"
  vl_model_sha=f4c899fde8117caaab1b18fc506389e0e504ce6cdb6c7ee19cad3100ba08c4d0
  if [[ -n "${VLLM_DSV41_VISION_SPAN_OVERRIDE:-}" ]]; then
    [[ "$VLLM_DSV41_VISION_SPAN_OVERRIDE" == etherscan-v1-swap &&
       "${NATIVE_E5_VL_MODEL_SHA:-}" == 0c6b8cc4943a1e46bcc41a74acd8958b53ef09a276785f626a7287ba79bb77ba &&
       -z "${NATIVE_E4_VL_MODEL_SHA:-}" ]] ||
      die "unrecognized vision-span diagnostic identity"
    vl_model_sha=$NATIVE_E5_VL_MODEL_SHA
  elif [[ -n "${NATIVE_E5_VL_MODEL_SHA:-}" || -n "${NATIVE_E4_VL_MODEL_SHA:-}" ]]; then
    die "diagnostic wrapper hash requires the exact vision-span override"
  fi
  [[ "$(sha256sum "$pyroot/vllm/models/deepseek_v4_1/nvidia/vl_model.py" | cut -d' ' -f1)" == "$vl_model_sha" ]] ||
    die "vision backend selector missing or modified"
  [[ "$(sha256sum "$pyroot/vllm/models/deepseek_v4/common/vision.py" | cut -d' ' -f1)" == 54f6fe4b4ae0d8a2876f101d725e5ab6e4ea522c813fde4d6f5ea718232279e7 ]] ||
    die "torch vision compatibility tower missing or modified"
  moe_source_sha=1332bf6c6ad649266df8896a60a9bf851e684e333716fa8ecbee66fa3907ca36
  if [[ "$moe_bf16" == 1 ]]; then
    moe_source_sha=7e44818dfb7fbf6e36d51a964af7741d9a585f5041101f10ca2d1266162c7448
  fi
  [[ "$(sha256sum "$pyroot/vllm/models/deepseek_v4_1/tilelang/moe.py" | cut -d' ' -f1)" == "$moe_source_sha" ]] ||
    die "routed-MoE source does not match requested 1107/PR-1030-plus-0799 mode"
  (cd "$runtime" && sha256sum -c COMPILED-ARTIFACTS.sha256 >/dev/null) || die "compiled runtime hash mismatch"

  export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH="$pyroot/nvidia_cutlass_dsl/dsl_packages:$ced_pyroot:$pyroot"
  export PATH="$pyroot/bin:/usr/local/cuda/bin:/usr/bin:/bin"
  export LD_LIBRARY_PATH="$runtime/native-libs:$pyroot/torch/lib:$pyroot/nvidia/nccl/lib:$pyroot/nvidia/cu13/lib:$pyroot/nvidia/cudnn/lib:$pyroot/nvidia/cusparselt/lib:$pyroot/nvidia/nvshmem/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/sbsa-linux/lib"
  export VLLM_NCCL_SO_PATH="$pyroot/nvidia/nccl/lib/libnccl.so.2"
  export CPATH="$runtime/python-include:$runtime/python-include/python3.12"
  export CUDA_CACHE_MAXSIZE=4294967296 CUDA_CACHE_PATH="$root/cache/nv-compute"
  export CUDA_DEVICE_MAX_CONNECTIONS=32
  export DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false VLLM_PLUGINS=
  export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=1048576 OMP_NUM_THREADS=8
  export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,garbage_collection_threshold:0.6'
  export SAFETENSORS_FAST_GPU=1
  export TILELANG_CACHE_DIR="$root/cache/tilelang" TRITON_CACHE_DIR="$root/cache/triton"
  export VLLM_CACHE_DIR="$root/cache/vllm" VLLM_CACHE_ROOT="$root/cache/vllm" XDG_CACHE_HOME="$root/cache"
  export VLLM_NATIVE_CACHE_DIR="$runtime/native-cache"
  export SPARKNET_ROCE_CACHE_DIR="$runtime/sparknet-roce"
  export B12X_ROCE_CACHE_DIR="$runtime/b12x-roce"
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V2_MODEL_RUNNER=1
  export VLLM_DSV41_VISION_BACKEND="$vision_backend"
  export SPARK_DS41F_MEDIA_DATA_ONLY=1 VLLM_MEDIA_URL_ALLOW_REDIRECTS=0
  export VLLM_DS41_HEAD_DTYPE=float32 VLLM_DS41_PACKED_BF16_LM_HEAD=1
  export VLLM_ENABLE_PCIE_ALLREDUCE=0 VLLM_DISABLE_PYNCCL=0
  export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900
  export VLLM_L2_PREFETCH_GRID=2 VLLM_L2_PREFETCH_KERNELS=tilelang
  export VLLM_LATE_ADMISSION_LAYER=30 VLLM_MAX_IMAGE_PIXELS=16777216 VLLM_USE_BREAKABLE_CUDAGRAPH=0
  export SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2 SPARK_DS41F_DSPARK_MAIN_PROJ_TP=1 SPARK_DS41F_DSPARK_VOCAB_PARALLEL=1
  export SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=0
  export SPARK_DS41F_ENGRAM_PACKED_DIR="$root/data/engram-packed-rank$rank"
  export SPARKNET_ROCE_ALLREDUCE_CAPACITY_BYTES=2097152 SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES=2097152
  export SPARKNET_ROCE_ALLGATHER_MAX_BYTES=2097152 SPARKNET_ROCE_PROXY_CPU=big SPARKNET_ROCE_TIMEOUT_MS=10000
  export SPARKNET_ROCE_TOPOLOGY=ring4 SPARKNET_ROCE_PEER_HCAS="$peer_hcas" SPARKNET_ROCE_SUBNETS="$roce_subnets"
  export NCCL_ALGO=Ring NCCL_BUFFSIZE=4194304 NCCL_COLLNET_ENABLE=0 NCCL_CROSS_NIC=1
  export NCCL_CUMEM_ENABLE=1 NCCL_CUMEM_HOST_ENABLE=0 NCCL_DEBUG=WARN NCCL_DEBUG_SUBSYS=INIT
  export NCCL_GIN_ENABLE=0 NCCL_IB_ADDR_FAMILY=AF_INET NCCL_IB_DISABLE=0 NCCL_IB_HCA="$nccl_hcas"
  export NCCL_IB_MERGE_NICS=1 NCCL_IB_ROCE_VERSION_NUM=2 NCCL_IB_SUBNET_AWARE_ROUTING=1 NCCL_IB_SUBNET_PREFIX_LEN=24
  export NCCL_IGNORE_CPU_AFFINITY=1 NCCL_LL128_BUFFSIZE=262144 NCCL_MAX_NCHANNELS=4 NCCL_MIN_NCHANNELS=4
  export NCCL_MIN_TRAFFIC_PER_CHANNEL=512 NCCL_MNNVL_ENABLE=0 NCCL_NET=IB NCCL_NET_PLUGIN=none
  export NCCL_NVLS_ENABLE=0 NCCL_P2P_DISABLE=1 NCCL_PAT_ENABLE=0 NCCL_PROTO='^LL128'
  export NCCL_RMA_DISABLE=1 NCCL_RUNTIME_CONNECT=1 NCCL_SHM_DISABLE=1
  export NCCL_SOCKET_IFNAME=ring0 NCCL_SWITCHLESS_BIDIRECTIONAL=2 NCCL_THREAD_THRESHOLDS='-2 -2 -2 1 1 1'
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 GLOO_SOCKET_IFNAME=ring0 TP_SOCKET_IFNAME=ring0 VLLM_HOST_IP="$control_ip"
  unset NCCL_IB_GID_INDEX SPARKNET_ROCE_GID_INDEX http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy

  mkdir -p "$CUDA_CACHE_PATH" "$TILELANG_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_DIR"
  ulimit -s 65536

  command=(
    /usr/bin/python3 -S -m vllm.entrypoints.cli.main serve "$checkpoint"
    --served-model-name "$model" --host 0.0.0.0 --port "$port"
    --middleware native_health.HealthMiddleware --worker-extension-cls native_health.HealthWorker
    --tokenizer-mode deepseek_v41 --dtype bfloat16 --tensor-parallel-size 4
    --load-format direct --block-size 256 --enable-prefix-caching
    --mm-encoder-tp-mode "$mm_encoder_tp_mode"
    --enable-prompt-tokens-details --enable-chunked-prefill
    --max-model-len 1048576 --max-num-seqs "$seqs" --max-parallel-prefills 1 "${timing_args[@]}"
    --gpu-memory-utilization 0.82 --max-num-batched-tokens 8192
    --kv-cache-memory-bytes 5167382528 --max-cudagraph-capture-size 96
    --generation-config vllm
    --default-chat-template-kwargs '{"reasoning_effort":"low"}'
    --override-generation-config '{"temperature":1.0,"top_p":1.0,"top_k":20,"max_new_tokens":131072}'
    --engram-config "$engram_config"
    --attention-backend TILELANG --linear-backend tilelang --moe-backend tilelang
    --jit-monitor-mode off
    --compilation-config "$compilation_config"
    --prefix-cache-retention-interval 8192 --long-prefill-token-threshold "$chunk"
    --async-scheduling --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice
    --distributed-executor-backend mp --nnodes 4 --no-scheduler-reserve-full-isl --no-enable-flashinfer-autotune
    --limit-mm-per-prompt '{"image":64}' --mm-processor-cache-gb 0
    --prefill-yield short --prefill-yield-limit-s 10
    --node-rank "$rank" --master-addr 10.20.255.12 --master-port 29841
  )
  if [[ "$vision_backend" == torch ]]; then
    command+=(--mm-encoder-attn-backend TORCH_SDPA)
  fi
  if [[ "$spec_mode" == dspark ]]; then
    command+=(--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_tensor_parallel_size":4,"attention_backend":"TILELANG","draft_sample_method":"greedy","rejection_sample_method":"block","enable_adaptive_verification":true,"adaptive_verification_row_cost":0.035}')
  elif [[ "$spec_mode" == dspark-static ]]; then
    command+=(--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_tensor_parallel_size":4,"attention_backend":"TILELANG","draft_sample_method":"greedy","rejection_sample_method":"block","enable_adaptive_verification":false}')
  elif [[ "$spec_mode" == target-exact ]]; then
    command+=(--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_tensor_parallel_size":4,"attention_backend":"TILELANG","draft_sample_method":"greedy","rejection_sample_method":"target_exact","enable_adaptive_verification":false}')
  fi
  [[ "$rank" == 0 ]] || command+=(--headless)

  log="$logs/candidate-rank${rank}-port${port}-$(date +%Y%m%dT%H%M%S%z).log"
  printf 'native-node: mm_encoder_tp_mode=%s vision_backend=%s spec_mode=%s\n' \
    "$mm_encoder_tp_mode" "$vision_backend" "$spec_mode" >"$log"
  printf 'native-node: engram_projection_tp=%s\n' "$engram_projection_tp" >>"$log"
  printf 'native-node: cudagraph_mode=%s\n' "$cudagraph_mode" >>"$log"
  printf 'native-node: long_prefill_token_threshold=%s\n' "$chunk" >>"$log"
  printf '%s\n' 'native-node: max_num_seqs=5 kv_cache_memory_bytes=5167382528 logical_pool_target=5242880' >>"$log"
  printf 'native-node: moe_bf16=%s source_sha256=%s\n' "$moe_bf16" "$moe_source_sha" >>"$log"
  printf 'native-node: exact_ced=1 overlay_tag=%s manifest_sha256=%s\n' "$ced_overlay_tag" "$ced_manifest_sha" >>"$log"
  case "$spec_mode" in
    dspark) printf '%s\n' 'native-node: dspark_tokens=5 block_verification=true adaptive_verification=true adaptive_verification_row_cost=0.035' >>"$log" ;;
    dspark-static) printf '%s\n' 'native-node: dspark_tokens=5 block_verification=true adaptive_verification=false' >>"$log" ;;
    target-exact) printf '%s\n' 'native-node: dspark_tokens=5 target_exact_verification=true adaptive_verification=false' >>"$log" ;;
    off) printf '%s\n' 'native-node: speculative_decoding=off' >>"$log" ;;
  esac
  spawn_pid=
  abort_spawn() {
    local rc=${1:-1} i rows
    trap - ERR INT TERM
    if [[ -n "$spawn_pid" ]] && kill -0 -- "-$spawn_pid" 2>/dev/null; then
      rows=$(ps -eo uid=,pgid=,pid=,stat= --no-headers | awk -v pgid="$spawn_pid" '$2 == pgid { print }')
      if [[ -n "$rows" ]] && ! awk -v uid="$(id -u)" '$1 != uid { bad=1 } END { exit bad ? 0 : 1 }' <<<"$rows"; then
        kill -TERM -- "-$spawn_pid" 2>/dev/null || true
        for i in $(seq 1 30); do
          kill -0 -- "-$spawn_pid" 2>/dev/null || break
          sleep 1
        done
        if kill -0 -- "-$spawn_pid" 2>/dev/null; then
          kill -KILL -- "-$spawn_pid" 2>/dev/null || true
          for i in $(seq 1 30); do
            kill -0 -- "-$spawn_pid" 2>/dev/null || break
            sleep 1
          done
        fi
      fi
    fi
    if [[ -z "$spawn_pid" ]] || ! kill -0 -- "-$spawn_pid" 2>/dev/null; then
      rm -f "$pidfile" "$portfile" "$identityfile"
    else
      printf 'native-node: launched process group %s could not be quiesced; identity retained\n' "$spawn_pid" >&2
    fi
    exit "$rc"
  }
  setsid nohup taskset -c 5-9,15-19 "${command[@]}" 9>&- >>"$log" 2>&1 < /dev/null &
  spawn_pid=$!
  pid=$spawn_pid
  trap 'abort_spawn $?' ERR
  trap 'abort_spawn 130' INT
  trap 'abort_spawn 143' TERM
  printf '%s\n' "$pid" >"$pidfile"
  printf '%s\n' "$port" >"$portfile"
  write_identity "$pid"
  sleep 2
  if ! prove_pid "$pid"; then
    tail -80 "$log" >&2 || true
    printf 'native-node: candidate failed immediate identity proof\n' >&2
    abort_spawn 1
  fi
  trap - ERR INT TERM
  printf 'rank=%s started pid=%s port=%s mm_encoder_tp_mode=%s vision_backend=%s spec_mode=%s engram_projection_tp=%s cudagraph_mode=%s moe_bf16=%s log=%s\n' \
    "$rank" "$pid" "$port" "$mm_encoder_tp_mode" "$vision_backend" "$spec_mode" \
    "$engram_projection_tp" "$cudagraph_mode" "$moe_bf16" "$log"
}

case "$action" in
  start) start_candidate "${3:-5}" ;;
  stop) stop_candidate ;;
  status) status ;;
  assert-stopped) assert_stopped ;;
  adopt) adopt_running ;;
  *) die "usage: $0 start [18888|8888] [5] | stop | status | assert-stopped | adopt" ;;
esac
