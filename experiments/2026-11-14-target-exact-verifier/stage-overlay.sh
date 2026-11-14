#!/usr/bin/env bash
# Materialize the AI-assisted target-exact files over the existing, immutable
# exact-CED overlay. The running overlay is only read; publication is an atomic
# rename into a new diagnostics path on each rank.
set -Eeuo pipefail
umask 077

here=$(cd -- "$(dirname -- "$0")" && pwd)
models=/home/enigmatic331/models
source_tree="$models/worktrees/vllm-target-exact"
trial="$models/pico-dgx-ds4-trial-20261111"
ssh_config="$trial/deployment/ssh_config"
remote_root=spark-ds41f-native-20261113
base_tag=accuracy-trace-20261114-v13cedexact
target_tag=accuracy-trace-target-exact-6ae956399820
incoming_tag=target-exact-6ae956399820-incoming
hosts=(pico-dgx02 pico-dgx04 pico-dgx03 pico-dgx01)
SSH=(ssh -F "$ssh_config" -o BatchMode=yes)
SCP=(scp -F "$ssh_config" -q)

declare -A source_hashes=(
  [config-speculative.py]=6983a9724d64f6654c64bb5b6e68af8978ff56de182e191e3bfd732bc9abe41e
  [sample-rejection_sampler.py]=5acb2a52d68e3a68f8a6ad24d93ccf8c4ad015c0a7cb30aa2aa2b5474bd61018
  [tilelang-sampling.py]=c65c43428cd317fe659b7c27af047e795a2e7f7e61b1e8f3b73a46d3dc676eb6
  [gpu-rejection_sampler.py]=652e50aa5e0c0b56a22e28afaf5ca5286fd3b6d3428e31e80bd58b7a4e008f9f
  [test-target-exact.py]=21306913aae3ff8871c4f112e9cf9059f09a7c1d36ad2348d48451acfd4c0089
)
declare -A source_paths=(
  [config-speculative.py]="$source_tree/vllm/config/speculative.py"
  [sample-rejection_sampler.py]="$source_tree/vllm/v1/sample/rejection_sampler.py"
  [tilelang-sampling.py]="$source_tree/vllm/v1/worker/gpu/sample/tilelang_sampling.py"
  [gpu-rejection_sampler.py]="$source_tree/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py"
  [test-target-exact.py]="$source_tree/tests/v1/worker/gpu/test_target_exact_verification.py"
)

for name in "${!source_paths[@]}"; do
  [[ "$(sha256sum "${source_paths[$name]}" | cut -d' ' -f1)" == "${source_hashes[$name]}" ]] || {
    printf 'local source hash mismatch: %s\n' "$name" >&2
    exit 1
  }
done

stage_host() {
  local host=$1 remote_incoming="$remote_root/diagnostics/$incoming_tag" name
  "${SSH[@]}" "$host" "set -eu; d=\"\$HOME/$remote_incoming\"; mkdir -p \"\$d\"; test ! -L \"\$d\""
  for name in "${!source_paths[@]}"; do
    "${SCP[@]}" "${source_paths[$name]}" "$host:$remote_incoming/$name.next"
    "${SSH[@]}" "$host" "set -eu; d=\"\$HOME/$remote_incoming\"; n=\"\$d/$name.next\"; t=\"\$d/$name\"; test \"\$(sha256sum \"\$n\" | cut -d' ' -f1)\" = '${source_hashes[$name]}'; chmod 0444 \"\$n\"; mv -fT -- \"\$n\" \"\$t\""
  done

  "${SSH[@]}" "$host" bash -s -- "$remote_root" "$base_tag" "$target_tag" "$incoming_tag" <<'REMOTE'
set -Eeuo pipefail
umask 077
remote_root=$1
base_tag=$2
target_tag=$3
incoming_tag=$4
root="$HOME/$remote_root/diagnostics"
base="$root/$base_tag"
target="$root/$target_tag"
incoming="$root/$incoming_tag"

verify_target() {
  [[ -d "$target/site-packages/vllm" && ! -L "$target" && ! -L "$target/site-packages" ]] || return 1
  [[ -f "$target/ACCURACY_TRACE.sha256" && ! -L "$target/ACCURACY_TRACE.sha256" ]] || return 1
  (cd "$target" && sha256sum -c ACCURACY_TRACE.sha256 --quiet)
}

if [[ -e "$target" ]]; then
  verify_target
  sha256sum "$target/ACCURACY_TRACE.sha256"
  exit 0
fi

[[ -d "$base/site-packages/vllm" && ! -L "$base" && ! -L "$base/site-packages" ]] || {
  printf 'unsafe base overlay: %s\n' "$base" >&2
  exit 1
}
[[ "$(sha256sum "$base/site-packages/vllm/config/speculative.py" | cut -d' ' -f1)" == 6f048d22a316cd29a6881b473b91e166ced30044d3a836ba1765c820678ff437 ]]
[[ "$(sha256sum "$base/site-packages/vllm/v1/sample/rejection_sampler.py" | cut -d' ' -f1)" == ec1a86e0cf6b75dcbea370b2919c0f031d4a8ee115706c661fc9c7e090aa5bde ]]
[[ "$(sha256sum "$base/site-packages/vllm/v1/worker/gpu/sample/tilelang_sampling.py" | cut -d' ' -f1)" == e5db220047551b7ea6f58d93e008708aabfbdddd13ea77de48e285debee3ffc1 ]]
[[ "$(sha256sum "$base/site-packages/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py" | cut -d' ' -f1)" == 2b0010defc3bb9daf43692fb619f3e83e6ebba00df968f4ec4bf2a5bdf85cef4 ]]
for file in config-speculative.py sample-rejection_sampler.py tilelang-sampling.py gpu-rejection_sampler.py test-target-exact.py; do
  [[ -f "$incoming/$file" && ! -L "$incoming/$file" ]]
done

temporary=$(mktemp -d "$root/.${target_tag}.XXXXXX")
mkdir -p "$temporary/site-packages" "$temporary/tests/v1/worker/gpu"
cp -a --reflink=auto "$base/site-packages/vllm" "$temporary/site-packages/vllm"
install -m 0444 "$incoming/config-speculative.py" "$temporary/site-packages/vllm/config/speculative.py"
install -m 0444 "$incoming/sample-rejection_sampler.py" "$temporary/site-packages/vllm/v1/sample/rejection_sampler.py"
install -m 0444 "$incoming/tilelang-sampling.py" "$temporary/site-packages/vllm/v1/worker/gpu/sample/tilelang_sampling.py"
install -m 0444 "$incoming/gpu-rejection_sampler.py" "$temporary/site-packages/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py"
install -m 0444 "$incoming/test-target-exact.py" "$temporary/tests/v1/worker/gpu/test_target_exact_verification.py"
(
  cd "$temporary"
  sha256sum \
    site-packages/vllm/models/deepseek_v4_1/ced.py \
    site-packages/vllm/models/deepseek_v4_1/attention.py \
    site-packages/vllm/models/deepseek_v4_1/nvidia/model_state.py \
    site-packages/vllm/config/speculative.py \
    site-packages/vllm/v1/sample/rejection_sampler.py \
    site-packages/vllm/v1/worker/gpu/sample/tilelang_sampling.py \
    site-packages/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py \
    tests/v1/worker/gpu/test_target_exact_verification.py >ACCURACY_TRACE.sha256
  chmod 0444 ACCURACY_TRACE.sha256
  sha256sum -c ACCURACY_TRACE.sha256 --quiet
)
mv -T -- "$temporary" "$target"
verify_target
sha256sum "$target/ACCURACY_TRACE.sha256"
REMOTE
}

pids=()
for host in "${hosts[@]}"; do
  stage_host "$host" &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
exit "$rc"
