#!/usr/bin/env bash
# Reproducible transition controller for the five-slot target-exact candidate.
# This is AI-assisted private deployment glue adapted from the hash-pinned
# qualification controller; it is not upstream spark-ds41f code or an
# upstream-ready contribution.
set -Eeuo pipefail
umask 077

here=/home/enigmatic331/models/spark-ds41f-native-audit-20261112
run=/home/enigmatic331/models/worktrees/spark-ds41f-target-exact/experiments/2026-11-14-target-exact-verifier
trial=/home/enigmatic331/models/pico-dgx-ds4-trial-20261111
experiment=/home/enigmatic331/models/spark-ds41f-review-20261112/experiments/2026-11-13-native-client-compat
ssh_config="$trial/deployment/ssh_config"
base=http://10.80.2.2:8888/v1
model=DeepSeek-V4.1-Flash-1017-v4
remote_root=spark-ds41f-native-20261113
head=pico-dgx02
workers=(pico-dgx04 pico-dgx03 pico-dgx01)
hosts=("$head" "${workers[@]}")
SSH=(ssh -F "$ssh_config" -o BatchMode=yes)
SCP=(scp -F "$ssh_config" -q)
profile='NATIVE_MM_ENCODER_TP_MODE=data NATIVE_VISION_BACKEND=torch NATIVE_SPEC_MODE=target-exact NATIVE_ENGRAM_PROJECTION_TP=true NATIVE_CUDAGRAPH_MODE=FULL_AND_PIECEWISE NATIVE_MOE_BF16=0 NATIVE_LONG_PREFILL_TOKEN_THRESHOLD=8064'
media_allowlist='["https://raw.githubusercontent.com/christopherowen/spark-ds41f/998b21a0b9ecdc5c61698be89e7318bef417116e/experiments/2026-09-25-vision/fixtures/shapes.png","https://raw.githubusercontent.com/christopherowen/spark-ds41f/998b21a0b9ecdc5c61698be89e7318bef417116e/experiments/2026-09-25-vision/fixtures/card.png"]'
head_rel="$remote_root/bin/target-exact-6ae956399820-head-v1.sh"
worker_rel="$remote_root/bin/target-exact-6ae956399820-worker-v1.sh"
head_sha=ce42589797a2999e1122056df2c51306f44855ae981960a4fb807d77c4be0db4
worker_sha=981841b611b4dea79a29d383f3bacc55b739b715abb556cd8d4655e129360b73
probe_sha=57ab64d4f2423e9d58799292109f6a9cbd210e2d88c99a18c067b326e8df475b

completed=0
touched=0
legacy_paused=0
deadlines_armed=0

log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

parallel_remote() {
  local command=$1 host pid rc=0
  local -a pids=()
  for host in "${hosts[@]}"; do
    "${SSH[@]}" "$host" "$command" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  return "$rc"
}

hold_check() {
  parallel_remote 'set -eu; for file in "$HOME/spark-hold.json" /tmp/spark-hold.json /run/user/"$(id -u)"/spark-hold.json; do test ! -e "$file" || { printf "GPU hold present: %s\n" "$file" >&2; exit 1; }; done'
}

pause_legacy() {
  systemctl --user stop dgx-cluster-watchdog.timer dgx-cluster-watchdog.service
  systemctl --user reset-failed dgx-cluster-watchdog.service || true
  [[ "$(systemctl --user is-active dgx-cluster-watchdog.timer || true)" == inactive ]]
  legacy_paused=1
}

resume_legacy() {
  (( legacy_paused )) || return 0
  systemctl --user start dgx-cluster-watchdog.timer
  systemctl --user is-active --quiet dgx-cluster-watchdog.timer
  legacy_paused=0
}

stage_launchers() {
  [[ "$(sha256sum "$run/target-exact-head.sh" | cut -d' ' -f1)" == "$head_sha" ]]
  [[ "$(sha256sum "$run/target-exact-worker.sh" | cut -d' ' -f1)" == "$worker_sha" ]]
  "${SCP[@]}" "$run/target-exact-head.sh" "$head:$head_rel.next"
  "${SSH[@]}" "$head" "set -eu; n=\"\$HOME/$head_rel.next\"; t=\"\$HOME/$head_rel\"; test \"\$(sha256sum \"\$n\" | cut -d' ' -f1)\" = '$head_sha'; chmod 0555 \"\$n\"; mv -fT -- \"\$n\" \"\$t\""
  local host
  for host in "${workers[@]}"; do
    "${SCP[@]}" "$run/target-exact-worker.sh" "$host:$worker_rel.next"
    "${SSH[@]}" "$host" "set -eu; n=\"\$HOME/$worker_rel.next\"; t=\"\$HOME/$worker_rel\"; test \"\$(sha256sum \"\$n\" | cut -d' ' -f1)\" = '$worker_sha'; chmod 0555 \"\$n\"; mv -fT -- \"\$n\" \"\$t\""
  done
}

arm_recovery() {
  local deadline host
  deadline=$(( $(date +%s) + 1800 ))
  if [[ -f "$here/state/watchdog.json" && ! -L "$here/state/watchdog.json" ]]; then
    python3 "$here/native-watchdog.py" set-mode transition --port 8888 --transition-seconds 1800
  else
    python3 "$here/native-watchdog.py" arm --deadline-epoch "$deadline" --port 8888 --transition-seconds 1800
  fi
  "$here/native-watchdog-loop.sh" start
  for host in "${hosts[@]}"; do
    "${SSH[@]}" "$host" "\"\$HOME/$remote_root/bin/native-cron-node.sh\" arm '$deadline'"
  done
  "$here/native-watchdog-cron.sh" verify
  deadlines_armed=1
}

disarm_rank_deadlines() {
  local host
  for host in "${hosts[@]}"; do
    "${SSH[@]}" "$host" '"$HOME/spark-ds41f-native-20261113/bin/native-cron-node.sh" disarm'
  done
  deadlines_armed=0
}

wait_reclaimed() {
  local started host available clean
  started=$(date +%s)
  while :; do
    clean=1
    for host in "${hosts[@]}"; do
      available=$("${SSH[@]}" "$host" 'set -eu; root="$HOME/spark-ds41f-native-20261113"; "$root/bin/native-node.sh" assert-stopped >/dev/null; awk '\''/^MemAvailable:/ {print $2 * 1024}'\'' /proc/meminfo') || { clean=0; break; }
      (( available >= 120000000000 )) || { clean=0; break; }
    done
    (( clean )) && break
    (( $(date +%s) - started < 300 )) || return 1
    sleep 5
  done
  sleep 15
}

start_profile() {
  local allowlist_q host pid rc=0
  local -a pids=()
  printf -v allowlist_q '%q' "$media_allowlist"
  "${SSH[@]}" "$head" "$profile SPARK_DS41F_MEDIA_HTTPS_ALLOWLIST_JSON=$allowlist_q \"\$HOME/$head_rel\" start 8888 5" &
  pids+=("$!")
  for host in "${workers[@]}"; do
    "${SSH[@]}" "$host" "$profile \"\$HOME/$worker_rel\" start 8888 5" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  return "$rc"
}

wait_ready() {
  local started
  started=$(date +%s)
  until curl -fsS --max-time 3 "$base/models" 2>/dev/null |
    jq -e --arg model "$model" 'any(.data[]; .id == $model)' >/dev/null; do
    (( $(date +%s) - started < 900 )) || return 1
    sleep 3
  done
}

verify_profile() {
  local host expected rel capacity
  for host in "${hosts[@]}"; do
    if [[ "$host" == "$head" ]]; then
      expected=$head_sha
      rel=$head_rel
    else
      expected=$worker_sha
      rel=$worker_rel
    fi
    "${SSH[@]}" "$host" "set -eu; root=\"\$HOME/$remote_root\"; \"\$root/bin/native-node.sh\" status; pid=\$(cat \"\$root/state/candidate.pid\"); log=\$(readlink -f \"/proc/\$pid/fd/1\"); test \"\$(sha256sum \"\$HOME/$rel\" | cut -d' ' -f1)\" = '$expected'; grep -Fqx 'native-node: exact_ced=1 overlay_tag=accuracy-trace-target-exact-6ae956399820 manifest_sha256=7a5883cbb0ee59ee3113fc1f01f6c8e6c095d51f7c1c39b70e65f460673291b0' \"\$log\"; grep -Fqx 'native-node: dspark_tokens=5 target_exact_verification=true adaptive_verification=false' \"\$log\"; grep -Fqx 'native-node: max_num_seqs=5 kv_cache_memory_bytes=5167382528 logical_pool_target=5242880' \"\$log\"; grep -Fqx 'native-node: long_prefill_token_threshold=8064' \"\$log\""
  done
  "${SSH[@]}" "$head" 'set -eu; root="$HOME/spark-ds41f-native-20261113"; pid=$(cat "$root/state/candidate.pid"); cmd=$(tr "\0" " " <"/proc/$pid/cmdline"); printf "%s" "$cmd" | grep -Fq -- "--max-num-seqs 5"; printf "%s" "$cmd" | grep -Fq -- "--kv-cache-memory-bytes 5167382528"; printf "%s" "$cmd" | grep -Fq -- "--long-prefill-token-threshold 8064"; printf "%s" "$cmd" | grep -Fq -- '"'"'"rejection_sample_method"'"'"':'"'"'"target_exact"'"'"''
  curl -fsS --max-time 15 "${base%/v1}/health" |
    jq -e --arg model "$model" '.status == "ok" and .model == $model and .max_batch_size == 5' >/dev/null
  capacity=$("${SSH[@]}" "$head" 'root="$HOME/spark-ds41f-native-20261113"; pid=$(cat "$root/state/candidate.pid"); log=$(readlink -f "/proc/$pid/fd/1"); sed -n "s/.*GPU KV cache size: \([0-9,]*\) tokens.*/\1/p" "$log" | tail -n1 | tr -d ,')
  [[ "$capacity" =~ ^[1-9][0-9]*$ ]] && (( capacity >= 5242880 ))
}

functional_canary() {
  local receipt
  receipt=$(mktemp -d -p "$run" restart-canary-XXXXXXXX)
  [[ "$(sha256sum "$experiment/release_probes.py" | cut -d' ' -f1)" == "$probe_sha" ]]
  timeout --signal=TERM --kill-after=30 3600 python3 "$experiment/release_probes.py" \
    --harness-dir "$trial/scripts" --base-url "$base" --model "$model" \
    --etherscan-dir /home/enigmatic331/models/dsv41-flash-results/2026-10-18-dgx-ds4-vision-concurrency/fixtures \
    --output "$receipt/release-probes.json" --repeats 1 --case-offset 60000 \
    --multimodal --render >"$receipt/stdout.log" 2>"$receipt/stderr.log"
  jq -e '.status == "PASS"' "$receipt/release-probes.json" >/dev/null
  log "restart canary passed receipt=$receipt"
}

recover() {
  log 'restart failed; restoring preserved TensorFold service'
  NATIVE_WATCHDOG_RECOVERY=1 "$here/native-fleet.sh" rollback || return 1
  NATIVE_WATCHDOG_RECOVERY=1 "$here/native-fleet.sh" restore-moe-1107 || return 1
}

cleanup() {
  local rc=$? recovery_rc=0
  trap - EXIT INT TERM
  if (( rc != 0 && touched )); then recover || recovery_rc=1; fi
  if (( deadlines_armed )); then disarm_rank_deadlines || recovery_rc=1; fi
  resume_legacy || recovery_rc=1
  if (( rc != 0 )); then exit "$rc"; fi
  exit "$recovery_rc"
}
trap cleanup EXIT INT TERM

bash -n "$run/target-exact-head.sh" "$run/target-exact-worker.sh"
[[ "$(sha256sum "$run/target-exact-head.sh" | cut -d' ' -f1)" == "$head_sha" ]]
[[ "$(sha256sum "$run/target-exact-worker.sh" | cut -d' ' -f1)" == "$worker_sha" ]]
if [[ "${1:-}" == --verify-only ]]; then
  verify_profile
  log 'five-slot production profile identity verified without restart'
  exit 0
fi
(( $# == 0 )) || die 'usage: restart-final-p5.sh [--verify-only]'
hold_check
bash "$run/stage-overlay.sh"
stage_launchers
arm_recovery
pause_legacy
touched=1
parallel_remote '"$HOME/spark-ds41f-native-20261113/bin/native-node.sh" stop'
wait_reclaimed
start_profile
wait_ready
verify_profile
functional_canary
python3 "$here/native-watchdog.py" set-mode qualified --port 8888
python3 "$here/native-watchdog.py" tick
disarm_rank_deadlines
resume_legacy
completed=1
log 'five-slot target-exact profile started and verified'
