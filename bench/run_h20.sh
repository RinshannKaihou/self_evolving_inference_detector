#!/usr/bin/env bash
# H20 上的接入验证与性能测试（docs/INTEGRATION.md §7、§10 的一键版）。在装有 vLLM 0.27.0 的 Python 环境里运行。
# r2（2026-10-22）：乱码监控 rc3 的在线接入；对照组见 bench/online_tpot.py（R2_PROFILES）。
#
#   bash bench/run_h20.sh env       环境检查：Python 依赖、vLLM 版本、按 wheel RECORD 逐文件核对 vLLM（只允许 v35 补丁行）、
#                                   旧框架补丁是否还在、v35 补丁状态、GPU 占用
#   bash bench/run_h20.sh install   pip install -e（--no-deps）本仓库，并跑单元测试
#   bash bench/run_h20.sh patch     vLLM 补丁：--check、--apply（已打 r1 补丁时替换为本版），再用真实的 output_processor.py 跑补丁测试
#   bash bench/run_h20.sh smoke     以 intercept + rc3 shadow 起服务：循环请求被 m4 拦截；强制乱码的请求 rc3 报警但不被拦截；
#                                   不带 logprobs 的请求 rc3 跳过；然后关服务
#   bash bench/run_h20.sh cpu       部署机 CPU 基准（m4 与 rc3 每 token 开销、钩子整体开销、状态大小、vLLM 原生检查）
#   bash bench/run_h20.sh tpot1k    在线 TPOT / CPU / 内存：1K 输出，并发 1 8 32 64，7 个 profile（约 30 分钟）
#   bash bench/run_h20.sh tpot8k    8K 输出，并发 1 8 16（r2 不跑：r1 的 8K 被 ignore_eos 诱发的循环干扰）
#   bash bench/run_h20.sh collect   把 runs/ 和环境信息打包成 v35_h20_results_<主机>_<时间>.tar.gz（在仓库根目录）
#   bash bench/run_h20.sh all       依次执行 env install patch smoke cpu tpot1k collect；任何一步失败都会先打包已有结果再退出
#
# 变量（用环境变量覆盖）：
#   PY     vLLM 环境的 python（默认 python）     MODEL  模型目录（默认 /workspace/models/Qwen3-30B-A3B）
#   TABLE  词表 qwen3 | glm | dsv4（默认 qwen3）  GPU    CUDA_VISIBLE_DEVICES（默认 5）   PORT  端口（默认 18790）
#   TP     张量并行卡数（默认 1）                 SERVE_ARGS  追加给 vllm serve 的参数（如 MTP 配置），整体加引号
#   FAKE=1 用 bench/fake_vllm_server.py 代替 vLLM 演练整个流程：不需要 GPU，几分钟跑完，结果目录带 _fake 后缀
set -euo pipefail
export COPYFILE_DISABLE=1                                   # macOS tar: no ._* entries
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
MODEL=${MODEL:-/workspace/models/Qwen3-30B-A3B}
TABLE=${TABLE:-qwen3}
GPU=${GPU:-5}
PORT=${PORT:-18790}
TP=${TP:-1}
SERVE_ARGS=${SERVE_ARGS:-}
FAKE=${FAKE:-0}
RUNS="$REPO/runs"
mkdir -p "$RUNS/logs"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUNS/logs/run_h20.log"; }
op_file() { "$PY" -c "import vllm,os;print(os.path.join(os.path.dirname(vllm.__file__),'v1','engine','output_processor.py'))"; }
vllm_bin() { echo "$(dirname "$("$PY" -c 'import sys;print(sys.executable)')")/vllm"; }

step_env() {
  local out="$RUNS/logs/env.txt"
  : > "$out"
  "$PY" -c 'import sys;print("python", sys.version.split()[0], sys.executable)' | tee -a "$out"
  for p in numpy aiohttp psutil transformers pytest; do
    "$PY" -c "import $p;print('$p', getattr($p, '__version__', '?'))" 2>/dev/null | tee -a "$out" || { echo "$p 缺失" | tee -a "$out"; }
  done
  if [ "$FAKE" = 1 ]; then log "FAKE=1：跳过 vLLM 与 GPU 检查"; return 0; fi
  local ver op
  ver=$("$PY" -c 'import vllm;print(vllm.__version__)')
  op=$(op_file)
  echo "vllm $ver  $op" | tee -a "$out"
  [ "$ver" = "0.27.0" ] || log "警告：vLLM 是 $ver，补丁针对 0.27.0（patch 步骤会先 --check）"
  if grep -q 'vllm_stability_monitor' "$op"; then
    log "错误：这个环境的 vLLM 还带着旧框架（vllm_stability_monitor）的补丁。请换干净的 vLLM 0.27.0 环境，或先撤掉旧补丁。"
    return 1
  fi
  if ! "$PY" - "$REPO" <<'PYEOF' 2>&1 | tee -a "$out"; then
import base64, hashlib, importlib.metadata as md, importlib.util, os, sys
spec = importlib.util.spec_from_file_location('p', os.path.join(sys.argv[1], 'integrations', 'vllm_0_27', 'patch_output_processor.py'))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
dist = md.distribution('vllm')
digest = lambda b: base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b'=').decode()
n, bad, op = 0, [], 'unchanged'
for f in dist.files:
    if not f.hash or f.hash.mode != 'sha256':
        continue
    n += 1
    try:
        data = open(f.locate(), 'rb').read()
    except FileNotFoundError:
        bad.append(('missing', str(f)))
        continue
    if digest(data) == f.hash.value:
        continue
    if str(f).replace(os.sep, '/').endswith('vllm/v1/engine/output_processor.py') and \
            digest(P.revert_text(data.decode('utf-8')).encode('utf-8')) == f.hash.value:
        op = 'v35 hook lines only'
        continue
    bad.append(('changed', str(f)))
direct = dist.read_text('direct_url.json')
print(f'vllm RECORD: {n} files checked, {len(bad)} differ; output_processor.py: {op}; installer: '
      f'{(dist.read_text("INSTALLER") or "?").strip()}; direct_url.json: {"present" if direct else "absent (from an index)"}')
for b in bad[:20]:
    print('  RECORD mismatch:', *b)
sys.exit(1 if bad else 0)
PYEOF
    log "错误：vLLM 安装与 wheel 的 RECORD 不一致（见上）。请用干净的 vLLM 0.27.0 环境（见交接说明 §3）。"
    return 1
  fi
  if grep -q 'v35-detector hook' "$op"; then
    if grep -q 'engine_core_output.new_logprobs' "$op"; then echo "v35 补丁：已打（本版）" | tee -a "$out"
    else echo "v35 补丁：已打（旧版 r1，patch 步骤会替换）" | tee -a "$out"; fi
  else echo "v35 补丁：未打" | tee -a "$out"; fi
  [ -x "$(vllm_bin)" ] || { log "错误：找不到 $(vllm_bin)"; return 1; }
  [ -d "$MODEL" ] || { log "错误：模型目录不存在：$MODEL"; return 1; }
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv | tee -a "$out"
  log "环境检查完成（GPU $GPU 应当空闲：memory.used 接近 0）"
}

has_pytest() { "$PY" -c 'import pytest' 2>/dev/null; }

step_install() {
  "$PY" -m pip install --no-deps -e "$REPO" >"$RUNS/logs/pip_install.log" 2>&1 || "$PY" -m pip install --no-deps "$REPO" >>"$RUNS/logs/pip_install.log" 2>&1
  "$PY" -c 'import v35_detector, numpy; print("v35_detector", v35_detector.__version__, v35_detector.__file__)' | tee -a "$RUNS/logs/pip_install.log"
  if has_pytest; then
    "$PY" -m pytest -q "$REPO/tests" 2>&1 | tee "$RUNS/logs/pytest.log" | tail -3
    log "单元测试通过"
  else
    log "警告：没有 pytest，跳过单元测试（可 pip install pytest 后再跑 install 这一步）"
  fi
}

step_patch() {
  if [ "$FAKE" = 1 ]; then log "FAKE=1：跳过 vLLM 补丁"; return 0; fi
  local op
  op=$(op_file)
  "$PY" "$REPO/integrations/vllm_0_27/patch_output_processor.py" --check | tee -a "$RUNS/logs/patch.log"
  "$PY" "$REPO/integrations/vllm_0_27/patch_output_processor.py" --apply | tee -a "$RUNS/logs/patch.log"   # r1 的旧钩子行会被替换
  grep -q 'v35_hooks.feed(req_state.request_id, new_token_ids, engine_core_output.new_logprobs)' "$op" \
    || { log "错误：补丁后的 feed 钩子没有传 new_logprobs：$op"; return 1; }
  "$PY" -c 'import vllm.v1.engine.output_processor as m; assert m.v35_hooks is not None; print("patched output_processor imports; v35 hooks loaded")' \
    | tee -a "$RUNS/logs/patch.log"
  if has_pytest; then
    V35_VLLM_OUTPUT_PROCESSOR="$op" "$PY" -m pytest -q "$REPO/tests/test_vllm_patch.py" 2>&1 | tee "$RUNS/logs/pytest_patch.log" | tail -2
  fi
  log "补丁已生效"
}

step_smoke() {
  local dir="$RUNS/smoke" serve
  mkdir -p "$dir"
  rm -f "$dir/events.jsonl"
  if [ "$FAKE" = 1 ]; then
    serve="$PY $REPO/bench/fake_vllm_server.py --port $PORT --step-ms 0.5 --loop-after 300"
  else
    serve="$(vllm_bin) serve $MODEL --served-model-name bench --host 127.0.0.1 --port $PORT --max-model-len 10240"
    serve="$serve --gpu-memory-utilization .85 --tensor-parallel-size $TP $SERVE_ARGS"
  fi
  V35_DETECTOR=intercept V35_RARE=shadow V35_DETECTOR_TABLE="$TABLE" V35_DETECTOR_EVENTS="$dir/events.jsonl" \
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$REPO/bench/smoke_intercept.py" --serve "$serve" --url "http://127.0.0.1:$PORT" \
    --events "$dir/events.jsonl" --server-log "$dir/server.log" --rare --table "$TABLE" | tee "$dir/smoke.txt"
  log "冒烟通过：m4 拦截；rc3 报警不拦截；无 logprobs 的请求 rc3 跳过"
}

step_cpu() {
  "$PY" "$REPO/bench/cpu_bench.py" --vllm --table "$TABLE" --out "$RUNS/cpu_bench_h20.json" | tee "$RUNS/logs/cpu_bench.log"
  log "CPU 基准完成：runs/cpu_bench_h20.json"
}

step_tpot() {                                                  # $1: tag; the rest: online_tpot.py arguments
  local tag=$1 out
  shift
  out="$RUNS/tpot_${TABLE}_${tag}"
  [ "$FAKE" = 1 ] && out="${out}_fake"
  if [ -f "$out/results.json" ]; then
    if "$PY" -c "import json,sys;sys.exit(0 if json.load(open('$out/results.json'))['status']=='complete' else 1)"; then
      log "$out 已完成，跳过（要重跑请先删掉该目录）"
      return 0
    fi
    mv "$out" "$out.failed.$(date +%Y%m%d_%H%M%S)"
    log "上次未完成的结果已改名保留，重跑 $tag"
  fi
  if [ "$FAKE" = 1 ]; then
    "$PY" "$REPO/bench/online_tpot.py" --output "$out" --model fake --synthetic-prompts --port "$PORT" \
      --server-cmd "$PY $REPO/bench/fake_vllm_server.py --step-ms 1" --input-length 64 --max-model-len 2048 \
      --table "$TABLE" --output-lengths 256 --concurrencies 1 4 --waves 1 --warmup-waves 1 2>&1 | tee "$RUNS/logs/tpot_${tag}.log"
  else
    "$PY" "$REPO/bench/online_tpot.py" --output "$out" --model "$MODEL" --table "$TABLE" --gpu "$GPU" --port "$PORT" \
      --tensor-parallel-size "$TP" --serve-args "$SERVE_ARGS" "$@" 2>&1 | tee "$RUNS/logs/tpot_${tag}.log"
  fi
  log "在线测试 $tag 完成：$out/results.json"
}

step_collect() {
  local env="$RUNS/logs/environment.txt" name f
  {
    date; hostname; uname -a; uptime
    nproc 2>/dev/null || true
    free -g 2>/dev/null || true
    nvidia-smi 2>/dev/null || true
    lscpu 2>/dev/null | head -20 || true
    "$PY" -m pip list 2>/dev/null | grep -i -E '^(vllm|torch|numpy|transformers|aiohttp|psutil|setproctitle|v35-detector) ' || true
  } > "$env" 2>&1
  : > "$RUNS/summary.txt"
  for f in "$RUNS"/tpot_*/results.json; do
    [ -f "$f" ] && { "$PY" "$REPO/bench/summarize_tpot.py" "$f" >> "$RUNS/summary.txt" 2>&1 || true; echo >> "$RUNS/summary.txt"; }
  done
  [ -f "$RUNS/cpu_bench_h20.json" ] && { echo "== cpu_bench_h20.json =="; cat "$RUNS/cpu_bench_h20.json"; echo; } >> "$RUNS/summary.txt"
  [ -f "$RUNS/smoke/smoke.txt" ] && { echo "== smoke =="; cat "$RUNS/smoke/smoke.txt"; } >> "$RUNS/summary.txt"
  name="v35_h20_results_$(hostname -s)_$(date +%Y%m%d_%H%M).tar.gz"
  tar czf "$REPO/$name" --exclude='._*' -C "$REPO" runs
  log "结果已打包：$REPO/$name（$(du -h "$REPO/$name" | cut -f1)），请把这个文件传回"
  log "文字摘要：$RUNS/summary.txt（不方便传文件时，把它的内容贴回即可）"
}

case "${1:-}" in
  env) step_env ;;
  install) step_install ;;
  patch) step_patch ;;
  smoke) step_smoke ;;
  cpu) step_cpu ;;
  tpot1k) step_tpot 1k --output-lengths 1024 --concurrencies 1 8 32 64 ;;
  tpot8k) step_tpot 8k --output-lengths 8192 --concurrencies 1 8 16 ;;
  collect) step_collect ;;
  all)
    for s in env install patch smoke cpu tpot1k; do
      log "=== $s ==="
      bash "$0" "$s" || { log "步骤 $s 失败，先打包已有结果"; bash "$0" collect; exit 1; }
    done
    bash "$0" collect ;;
  *) sed -n '2,27p' "$0"; exit 2 ;;
esac
