# 适配流程：把 m4 接入 vLLM 0.27

## 1. 接入位置

```text
API server 进程（可开多个，--api-server-count）
  AsyncLLM 的输出处理循环 → OutputProcessor.process_outputs
    反解码 → stop string 检查 → ★ v35_hooks.feed()        ← 本补丁只改这个文件
    命中 → finish_reason='stop'、stop_reason='v35_repetition: …' → 前端结束请求 → 通知 EngineCore 中止
EngineCore 进程（调度主循环）            不改
GPU worker 进程（前向、采样）            不改
```

为什么放在这一层：
- **和 stop string 走同一条路径**：vLLM 自己就是在这里检测 stop string、结束请求，再通知 EngineCore 中止。拦截复用了这条已有的路径。
- **不占调度线程，也不占 GPU 步**：vLLM 原生的重复检查在调度线程里逐 token 跑，开销直接加在每个 decode step 上。
- **晚一两步不影响结果**：检测器本来就每 64 个 token 才判一次，从前端发中止到 EngineCore 生效多出的一两步，相对这个粒度可以忽略。
- **只要 token id**：不需要 logprob，也不改 sampler。

与旧框架（`vllm_stability_monitor`，2026-09）的区别：
- 钩子位置相同；
- 去掉了 top-k 采集、sampler 和 logits 相关的补丁；
- 异步进程改用普通队列，每 64 个 token 发一次，不再用共享内存环；
- **新增了真正的拦截**：旧框架只写判定日志，不会结束请求。

## 2. 两种运行方式

| | `sync`（默认） | `async` |
|---|---|---|
| 检测在哪算 | 输出处理循环里，同一线程 | 独立检测进程（spawn），前端每 64 个 token 发一次 |
| 拦截时机 | 报警的那一步，当步结束 | 晚几步（本地测试晚 8 个 token） |
| 前端 CPU | 每 token 约 3.5 μs，加在 asyncio 事件循环上 | 每 64 token 一次入队 |
| 内存 | 无额外进程 | 多一个 Python 进程（本地 RSS 约 57 MB） |
| 何时选 | 默认。前端 CPU 有余量时，拦截更及时 | 前端 CPU 已经吃紧（高 QPS、多路反解码）时 |

两种方式的报警判定完全相同，只是拦截生效的时刻不同。在线测试会测两者的 TPOT 和 CPU（§7）。

## 3. 拦截语义

- `intercept` 模式：
  - 客户端收到报警之前生成的全部 token（包括报警那一块）；
  - `finish_reason` 为 `stop`，`stop_reason` 为 `v35_repetition: <组件>`；
  - 请求在 EngineCore 里被中止，不再占用 KV cache 和算力。
- `shadow` 模式：只写事件日志，不结束请求。用于上线前观察和性能测试。
- 已经由引擎结束的请求（EOS、长度上限、stop string），检测器不会改写它的 `finish_reason`。

## 4. 安装步骤（vLLM 0.27.0）

建议用干净的 vLLM 0.27.0 环境。旧框架的补丁如果还在，先用它自己的方式撤掉，避免两套钩子同时生效。

```bash
pip install -e /path/to/v35_m4_detector
python /path/to/v35_m4_detector/integrations/vllm_0_27/patch_output_processor.py --check
python /path/to/v35_m4_detector/integrations/vllm_0_27/patch_output_processor.py --apply
python -m pytest -q /path/to/v35_m4_detector/tests
```

启动：

```bash
V35_DETECTOR=shadow V35_DETECTOR_TABLE=qwen3 V35_DETECTOR_EVENTS=/var/log/v35-events.jsonl \
  vllm serve /path/to/Qwen3-30B-A3B ...
```

几点说明：
- 补丁会在原文件旁保存一份原件 `output_processor.py.v35-orig`。撤销用 `--revert`。
- 关闭检测有两种办法：设 `V35_DETECTOR=off` 后重启服务；或者不安装 `v35_detector` 包，此时打过补丁的 vLLM 行为与原版完全相同。
- 补丁只认 0.27.0。其他版本先跑 `--check --allow-version X.Y.Z`：6 个锚点都只出现一次、并且补丁后能编译，才可以用（0.18.0 也能通过）。

## 5. 新模型上线清单

1. **建词表**：`python -m v35_detector.vocab --tokenizer <tokenizer.json> --out <name>.npz`。只支持 byte-level BPE，Qwen、GLM、DeepSeek 都是这种。
2. **确定 think_end**：思考模式填 `</think>` 的 token id；非思考模式填 `none`。
3. **离线回放**：如果有这个模型的历史输出，用 `RepDetector` 跑一遍，统计报警率，并抽查报警的样本。
4. **影子模式上线**：设 `V35_DETECTOR=shadow`，在真实流量上观察几天：
   - 看报警率，以及按组件（`reason`）的分布；
   - 正常流量上的报警率应在千分之一量级；
   - 抽样复核报警的请求。
5. **打开拦截**：设 `V35_DETECTOR=intercept`。回滚时改回 `shadow` 或 `off` 即可。

阈值跨模型共用，不需要重调。DeepSeek-V4-Flash 没有调过参，在 2,039 条负例上只有 1 条误报。

## 6. 监控

- **事件日志**（JSONL，字段见接口说明 §8）：
  - `alert` 按 `reason` 统计；
  - `error` 应为 0；
  - `summary` 每 10 秒一行，含计数和检测耗时。
- **建议的告警**：
  - 出现 `error`；
  - `counters.worker_down > 0`（异步进程挂了）；
  - 报警率相对基线突增（可能是模型或硬件故障，也可能是误报）。

## 7. 上线前验证与性能测试（H20）

推荐用一键脚本 `bench/run_h20.sh`：

```bash
FAKE=1 bash bench/run_h20.sh all                                         # 先用模拟服务端演练一遍，不占 GPU，几分钟
PY=$(which python) MODEL=/workspace/models/Qwen3-30B-A3B GPU=5 bash bench/run_h20.sh all   # 正式运行，约 3 小时
```

- 脚本依次执行：env → install → patch → smoke → cpu → tpot1k → tpot8k → collect。
- 每一步都可以单独运行，例如 `bash bench/run_h20.sh tpot8k`。
- 任何一步失败，都会先把已有结果打包再退出。

下面是各步对应的手动命令。

```bash
cd /path/to/v35_m4_detector
python -m pytest -q
OP=$(python -c "import vllm,os;print(os.path.join(os.path.dirname(vllm.__file__),'v1','engine','output_processor.py'))")
V35_VLLM_OUTPUT_PROCESSOR=$OP python -m pytest -q tests/test_vllm_patch.py
```

**① 拦截冒烟**：起一个 `intercept` 模式的服务，发一个"重复一句话"的请求，确认请求以 `v35_repetition` 结束。

```bash
V35_DETECTOR=intercept V35_DETECTOR_TABLE=qwen3 V35_DETECTOR_EVENTS=/tmp/v35-smoke.jsonl CUDA_VISIBLE_DEVICES=5 \
  vllm serve /workspace/models/Qwen3-30B-A3B --served-model-name bench --port 18790 &
python bench/smoke_intercept.py --url http://127.0.0.1:18790 --events /tmp/v35-smoke.jsonl
```

**② 部署机 CPU 开销**：跑完把 `runs/cpu_bench_h20.json` 交回。

```bash
python bench/cpu_bench.py --vllm --out runs/cpu_bench_h20.json
```

**③ 在线 TPOT / CPU / 内存**：
- 6 个 profile：baseline、vLLM 默认、vLLM 调参、v35 sync、v35 async、baseline_end；
- v35 用影子模式，所有请求都生成满长度，各 profile 的工作量完全相同。

```bash
# 1K 输出，并发 1 / 8 / 32 / 64
python bench/online_tpot.py --output runs/tpot_qwen3_1k --model /workspace/models/Qwen3-30B-A3B --table qwen3 --gpu 5 \
  --output-lengths 1024 --concurrencies 1 8 32 64
# 8K 输出，并发 1 / 8 / 16（再高会超出 KV cache，触发抢占；结果里 preemptions 应为 0）
python bench/online_tpot.py --output runs/tpot_qwen3_8k --model /workspace/models/Qwen3-30B-A3B --table qwen3 --gpu 5 \
  --output-lengths 8192 --concurrencies 1 8 16
```

- **预计用时**：每个 profile 都要重启一次服务（约 3 分钟）。按 C=1 时 TPOT 约 4.7 ms 估算，1K 一轮约 40 分钟，8K 一轮约 1.5–2 小时（vLLM 原生检查在长输出上会更慢）。时间紧时可以先跑 1K，或改成 `--waves 1`。
- **交回什么**：每轮的 `results.json`，以及各 profile 目录下的 `server.log`、`v35_events.jsonl`。`collect` 会把它们打成一个包，并生成文字摘要 `runs/summary.txt`（由 `bench/summarize_tpot.py` 生成）。
- **测量要点**：
  - 两次 baseline 的差，就是这次测试本身的漂移；
  - vLLM 原生检查找到循环时，会提前结束请求，结果里记为 `early_stops`；
  - 内存看 API server 进程的 PSS 在 v35 profile 与 baseline 之间的差；async 模式另看检测进程的 PSS。

**④ GLM-5.3-Flash（4 卡、MTP）**：可选。命令同上，加上 `--tensor-parallel-size 4 --table glm --gpu 0,1,2,3`，以及该模型的启动参数（`--serve-args`）。MTP 下每步会产出多个 token，钩子会按块缓冲，判定结果不变；不过这个场景还没有在线验证过。

## 8. 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| 服务启动时报 `V35_DETECTOR...` 错误 | 环境变量取值不对，或自定义词表没有给 `V35_DETECTOR_THINK_END` |
| `--check` 报 anchor 出现 0 次 | vLLM 版本不同，或文件已被其他补丁改过 |
| 事件日志里有 `error` | 看 `where` 和 `error` 字段；该请求停止了检测，服务不受影响 |
| async 模式 `worker_down` | 检测进程退出了，看 server.log。重启服务即可恢复 |

## 9. 已知限制

- **streaming input 请求**：vLLM 0.27 的 `streaming_input`（输入分段送达）每结束一段都会经过 finish 钩子，所以第一段结束后检测就停了。普通请求不受影响。
- **未在线验证的场景**：投机解码（MTP）、流水线并行。
- **漏检**：周期 20–40 个 token、每份都有少量变化的短循环（例如带递增编号）会被新颖度守卫挡住，m4 报不出来。
- 召回和误报的口径见研究仓库 `detector_v35` 的 DESIGN.md §21。
