# 适配流程：把 m4 接入 vLLM 0.27（以及 rare_chars 的在线接入，§10）

## 1. 接入位置

```text
API server 进程（可开多个，--api-server-count）
  AsyncLLM 的输出处理循环 → OutputProcessor.process_outputs                     v1/engine/output_processor.py
    反解码 → stop string 检查 → ★ v35_hooks.feed(token ids, EngineCoreOutput.v35_rare)
    → logprobs 处理（请求要 logprobs 时，vLLM 原样） → 输出
    命中 → finish_reason='stop'、stop_reason='v35_repetition: …' → 前端结束请求 → 通知 EngineCore 中止
EngineCore 进程（调度主循环；TP=1 时采样也在这个进程里）
  Sampler.forward：采样后两个 Triton 核把每行原始 logits 规约成 (top-1, 生成 token) 两个 logprob   v1/sample/sampler.py
    → GPUModelRunner：异步调度时在输出拷贝流上与 sampled token 一起非阻塞拷回主机          v1/worker/gpu_model_runner.py
    → Scheduler.update_from_output：每个请求的两个值放进它的 EngineCoreOutput             v1/core/sched/scheduler.py
    → EngineCoreOutput 新增字段 v35_rare（None 时不上线）                                  v1/engine/__init__.py
  只有 V35_RARE=collect|shadow 时采样器才做这些（r3，§10）；m4 只用 output_processor.py 的钩子
```

为什么放在这一层：
- **和 stop string 走同一条路径**：vLLM 自己就是在这里检测 stop string、结束请求，再通知 EngineCore 中止。拦截复用了这条已有的路径。
- **不占调度线程，也不占 GPU 步**：vLLM 原生的重复检查在调度线程里逐 token 跑，开销直接加在每个 decode step 上。
- **晚一两步不影响结果**：检测器本来就每 64 个 token 才判一次，从前端发中止到 EngineCore 生效多出的一两步，相对这个粒度可以忽略。
- **m4 只要 token id**：不需要 logprob，不用采样器那一段。乱码监控 rc3 要的两个 logprob 由采样器算好、随 EngineCoreOutput
  送到同一个钩子（r3，§10），不经过客户端可见的 logprobs，请求不需要带 logprobs。

与旧框架（`vllm_stability_monitor`，2026-09）的区别：
- 前端钩子位置相同；
- 采样器采集只保留 v3.1 的"快行"数据（top-1 与生成 token），去掉了 top-20 与门控；值走自己的字段，
  不再借用 logprobs 通道，所以请求本身要 logprobs 时也照常返回 vLLM 的结果（v3.1 的实现要求请求不带 logprobs）；
- 异步进程改用普通队列，每 64 个 token 发一次，不再用共享内存环；
- **新增了真正的拦截**：旧框架只写判定日志，不会结束请求。

## 2. 两种运行方式

| | `sync`（默认） | `async` |
|---|---|---|
| 检测在哪算 | 输出处理循环里，同一线程 | 独立检测进程（spawn），前端每 64 个 token 发一次 |
| 拦截时机 | 报警的那一步，当步结束 | 晚几步（本地测试晚 8 个 token） |
| 前端 CPU | 每 token 约 3.5 μs（Mac）、9.5 μs（H20 服务器实测），加在 asyncio 事件循环上 | 每 64 token 一次入队；检测进程每 token 10–14 μs（H20 实测） |
| 内存（CPU，不占显存） | 都在 API server 进程内：词表约 2.3 MB，加每个在途请求约 165 KB | 检测进程内：进程本身 PSS 26–30 MB（H20 实测，含词表），加每个在途请求约 165 KB；前端启动时也缓存一份词表，每个请求只缓存不满 64 个的 token id |
| 何时选 | 默认。前端 CPU 有余量时，拦截更及时 | 前端 CPU 已经吃紧（高 QPS、多路反解码）时 |

两种方式的报警判定完全相同，只是拦截生效的时刻不同。在线测试会测两者的 TPOT 和 CPU（§7）。
开启乱码监控 rc3（§10）时，它和 m4 在同一处算：rc3 的表每进程常驻约 64 MB（H20 实测 PSS：sync 时 API server 多 71–82 MB，async 时检测进程 102–105 MB），sync 时在 API server 进程里（启动时加载），
async 时在检测进程里（第一个请求时加载）；每个在途请求另有约 1.3 KB 状态。

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
python /path/to/v35_m4_detector/integrations/vllm_0_27/patch_vllm.py --check
python /path/to/v35_m4_detector/integrations/vllm_0_27/patch_vllm.py --apply
python -m pytest -q /path/to/v35_m4_detector/tests
```

r3 起补丁改 5 个文件、共 35 行，每行带 `# v35-detector hook` 标记：

| 文件（vllm/ 下） | 行数 | 做什么 |
|---|---|---|
| `v1/engine/output_processor.py` | 17 | m4 与 rc3 的 register / feed / finish / abort 钩子；feed 传入这一步的 token id 与 `EngineCoreOutput.v35_rare` |
| `v1/sample/sampler.py` | 7 | `V35_RARE` 开启时：处理器之前留住原始 logits（bf16 直接引用，float32 才复制），采样后用两个 Triton 核算出每行的 (top-1, 生成 token) logprob |
| `v1/worker/gpu_model_runner.py` | 8 | 把 `[行数, 2]` 的值挂到 ModelRunnerOutput：异步调度（默认）在输出拷贝流上非阻塞拷回，`get_output()` 里转成列表；同步调度在本步同步之后直接转列表 |
| `v1/core/sched/scheduler.py` | 2 | 每个有新 token 的请求，把它那一行放进 `EngineCoreOutput(v35_rare=…)` |
| `v1/engine/__init__.py` | 1 | `EngineCoreOutput` 末尾新增字段 `v35_rare: list[float] \| None = None`（msgspec，None 时编码里省略） |

客户端可见的 logprobs 完全不动：值不进 `LogprobsTensors` / `new_logprobs`，请求带 logprobs 时得到的就是 vLLM 原本的结果。

启动：

```bash
V35_DETECTOR=shadow V35_DETECTOR_TABLE=qwen3 V35_DETECTOR_EVENTS=/var/log/v35-events.jsonl \
  vllm serve /path/to/Qwen3-30B-A3B ...
```

几点说明：
- 补丁会在每个原文件旁保存一份原件 `<文件>.v35-orig`。撤销用 `--revert`（5 个文件一起）。`--root DIR` 改打指定的 vllm 包目录。
- 已经打过 r1 / r2 版补丁（只改 output_processor.py）的环境：`--check` 对它显示 `patched by an earlier version`，`--apply` 直接把旧的钩子行
  换成本版并给另外 4 个文件打上，不需要先 `--revert`。
- 先检查、后写入：15 个锚点里任何一个不是恰好出现一次、或任何一个文件补丁后不能编译，`--apply` 一个文件都不写。
- 关闭检测有两种办法：设 `V35_DETECTOR=off`、`V35_RARE=off` 后重启服务；或者不安装 `v35_detector` 包，此时打过补丁的 vLLM 行为与原版完全相同。
  `V35_RARE=off` 时采样器不做任何额外的 GPU 工作，其余几处只多读一个属性。
- 补丁只认 0.27.0。其他版本先跑 `--check --allow-version X.Y.Z`：所有锚点都只出现一次、并且补丁后能编译，才可以用。
  锚点核对的是 GitHub 标签 v0.27.0 的源码，与 PyPI 上 0.27.0 wheel 的 RECORD 哈希逐个文件一致。

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
  - `alert`（m4）按 `reason` 统计；`rare_alert`（rc3）按 `tier`、`reason` 统计；
  - `error` 应为 0；
  - `summary` 每 10 秒一行，含计数和检测耗时（m4、rc3 分开）；
  - 开启 rc3 时，`finish` 的 `rare` 应全是 `checked`；出现 `no_values` 说明引擎没送值（补丁没打全、模型走了 V2 model runner、
    或 EngineCore 进程里 `V35_RARE` 不是 collect/shadow），`lost` 说明中途断了（投机解码、采集出错），见 §9、§10。
- **建议的告警**：
  - 出现 `error`；
  - `counters.worker_down > 0`（异步进程挂了）；
  - 报警率相对基线突增（可能是模型或硬件故障，也可能是误报）。

## 7. 上线前验证与性能测试（H20）

推荐用一键脚本 `bench/run_h20.sh`：

```bash
FAKE=1 bash bench/run_h20.sh all                                         # 先用模拟服务端演练一遍，不占 GPU，几分钟
PY=$(which python) MODEL=/workspace/models/Qwen3-30B-A3B GPU=5 bash bench/run_h20.sh all   # 正式运行，约 45 分钟
```

- 脚本依次执行：env → install → patch → gpucheck → smoke → cpu → tpot1k → collect（r2 起不再跑 tpot8k，见下文 r1 结果的最后一条）。
- 每一步都可以单独运行，例如 `bash bench/run_h20.sh tpot1k`。
- 任何一步失败，都会先把已有结果打包再退出。
- r3 的对照组与判读方法见 §10.4（r2 的见 §10.3）。

下面是各步对应的手动命令。

```bash
cd /path/to/v35_m4_detector
python -m pytest -q
R=$(python -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
V35_VLLM_SAMPLER=$R/v1/sample/sampler.py V35_VLLM_GPU_MODEL_RUNNER=$R/v1/worker/gpu_model_runner.py \
V35_VLLM_SCHEDULER=$R/v1/core/sched/scheduler.py V35_VLLM_ENGINE=$R/v1/engine/__init__.py \
V35_VLLM_OUTPUT_PROCESSOR=$R/v1/engine/output_processor.py python -m pytest -q tests/test_vllm_patch.py
CUDA_VISIBLE_DEVICES=5 python bench/collect_gpu_check.py --out runs/collect_gpu_check.json   # 采样器 Triton 核对 torch.log_softmax
```

**① 冒烟**：起一个 m4 `intercept`、rc3 `shadow` 的服务（另设 `V35_RARE_TRACE`，让钩子把采样器送来的值记进事件日志），确认：
- "重复一句话"的请求（不带 logprobs）以 `v35_repetition` 结束，rc3 也检查了它（`checked`），客户端没收到 logprob；
- 用 `logit_bias` 强制输出非法 UTF-8、同时带 `logprobs=1` 的请求：rc3 报 `bad_utf8`，请求照常跑满；
  采样器的两个值与 vLLM 返回的 raw logprobs 逐 token 一致（默认容差 1e-3），且被强制的 token 的 raw logprob 远低于 top-1（说明取的是偏置之前的原始分布）；
- `logprobs=1`、T=1 采样的请求：两个值同样与 vLLM 返回的一致（生成 token 常常不是 top-1）；
- 不带 logprobs 的请求：rc3 `checked`，响应里没有 logprob。

```bash
V35_DETECTOR=intercept V35_RARE=shadow V35_RARE_TRACE=4096 V35_DETECTOR_TABLE=qwen3 V35_DETECTOR_EVENTS=/tmp/v35-smoke.jsonl \
  CUDA_VISIBLE_DEVICES=5 vllm serve /workspace/models/Qwen3-30B-A3B --served-model-name bench --port 18790 &
python bench/smoke_intercept.py --url http://127.0.0.1:18790 --events /tmp/v35-smoke.jsonl --rare --table qwen3
```

**② 部署机 CPU 开销**：跑完把 `runs/cpu_bench_h20.json` 交回。

```bash
python bench/cpu_bench.py --vllm --out runs/cpu_bench_h20.json
```

**③ 在线 TPOT / CPU / 内存**：
- r3 默认 8 个 profile（§10.4）：baseline、collect_only、rc3_sync、rc3_async、m4_rc3_sync、m4_rc3_async、baseline_logprobs、baseline_end；
- r1 的 profile（vllm_default、vllm_tuned、v35_sync、v35_async）仍可用 `--profiles` 指定；r2 的 rc3_shadow_* 已去掉（r3 起 rc3 不再靠请求带 logprobs）；
- 检测器都用影子模式，所有请求都生成满长度，各 profile 的工作量完全相同。

```bash
# 1K 输出，并发 1 / 8 / 32 / 64
python bench/online_tpot.py --output runs/tpot_qwen3_1k --model /workspace/models/Qwen3-30B-A3B --table qwen3 --gpu 5 \
  --output-lengths 1024 --concurrencies 1 8 32 64
# r1 的对照组：
python bench/online_tpot.py --output runs/tpot_qwen3_1k_r1 --model /workspace/models/Qwen3-30B-A3B --table qwen3 --gpu 5 \
  --output-lengths 1024 --concurrencies 1 8 32 64 --profiles baseline vllm_default vllm_tuned v35_sync v35_async baseline_end
```

- **预计用时**：每个 profile 都要重启一次服务。r1 实测 1K 每个 profile 约 3.5 分钟（含启动），r2 的 7 个 profile 实测 27.5 分钟；
  r3 的 8 个 profile 约 30 分钟。时间紧时可以改成 `--waves 1`。
- **交回什么**：每轮的 `results.json`，以及各 profile 目录下的 `server.log`、`v35_events.jsonl`。`collect` 会把它们打成一个包，并生成文字摘要 `runs/summary.txt`（由 `bench/summarize_tpot.py` 生成）。
- **测量要点**：
  - 两次 baseline 的差，就是这次测试本身的漂移；
  - vLLM 原生检查找到循环时，会提前结束请求，结果里记为 `early_stops`；
  - 内存看 API server 进程的 PSS 在 v35 profile 与 baseline 之间的差；async 模式另看检测进程的 PSS。

**2026-10-21 的 H20 实测结果**（单卡、TP=1、Qwen3-30B-A3B、vLLM 0.27.0，代码标签 `h20-r1`；原始结果归档在研究仓库 `report/data/h20_20261021/`）：
- 测试与冒烟：138 项测试全部通过（含真实 vLLM 文件上的补丁测试）；拦截冒烟在第 1,984 个 token 处以 `v35_repetition` 结束。
- TPOT：开启检测（shadow）后，变化在 ±22 μs/token（≤ 0.14%）以内，与两次 baseline 之差同量级，即小于测量分辨率；全程无抢占。
- CPU：检测器每 token 离线 6.67 μs，线上 sync 9.5 μs、async 10.1–14.0 μs；vLLM 原生检查 92.0 μs（[50,500]×3）、248.6 μs（[32,2000]×3）。
- 内存：每请求状态最大 166,071 B；API server 进程的 PSS 与 baseline 相差不到 10 MB；async 检测进程 PSS 26–30 MB。
- 8K 一轮用了 `ignore_eos`，多数输出退化成循环（v35 在 75 个请求中报警 69 个，vLLM 原生检查提前结束 42 个），所以 TPOT 结论以 1K 为准。

**④ GLM-5.3-Flash（4 卡、MTP）**：可选。命令同上，加上 `--tensor-parallel-size 4 --table glm --gpu 0,1,2,3`，以及该模型的启动参数（`--serve-args`）。MTP 下每步会产出多个 token，钩子会按块缓冲，判定结果不变；不过这个场景还没有在线验证过。

## 8. 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| 服务启动时报 `V35_DETECTOR...` 错误 | 环境变量取值不对，或自定义词表没有给 `V35_DETECTOR_THINK_END` |
| `--check` 报 anchor 出现 0 次 | vLLM 版本不同，或文件已被其他补丁改过 |
| 事件日志里有 `error` | 看 `where` 和 `error` 字段；该请求停止了检测，服务不受影响 |
| async 模式 `worker_down` | 检测进程退出了，看 server.log。重启服务即可恢复 |
| 启动时报 `V35_RARE needs a shipped table` | rc3 只接受内置词表名（qwen3 / glm / dsv4）：它还需要该模型的乱码词表与参照表 |
| `finish` 的 `rare` 都是 `no_values` | 引擎没送值：5 个文件没打全（`patch_vllm.py --check`）；模型走了 V2 model runner（0.27.0 的稠密模型默认如此，见 §9）；或 EngineCore 进程里 `V35_RARE` 不是 collect / shadow |
| `rare` 出现 `lost` | 值中途断了：投机解码（MTP）下有草稿的步不采集；或采集出错，server.log 里有 `v35 rc3 collection failed`（采样不受影响）；见接口说明 §9.7 |
| server.log 有 `v35 rc3 collection failed` | 采样器那两个核出错（例如 Triton 编译失败），这一步没有值，服务照常。先跑 `bench/collect_gpu_check.py` 定位 |

## 9. 已知限制

- **streaming input 请求**：vLLM 0.27 的 `streaming_input`（输入分段送达）每结束一段都会经过 finish 钩子，所以第一段结束后检测就停了。普通请求不受影响。
- **未在线验证的场景**：投机解码（MTP）、流水线并行、张量并行（TP > 1 时采样在 worker 进程里，值随 ModelRunnerOutput 经 pickle 传回，
  按代码应当可用，没有测过）。
- **rc3 的采集范围（r3）**：只接 vLLM 0.27.0 的 V1 GPU model runner（`vllm/v1/worker/gpu_model_runner.py`）。0.27.0 对 MoE 模型
  （Qwen3-30B-A3B、GLM、DeepSeek）默认用它；稠密模型（如 Qwen3-8B / 32B）默认走 V2 model runner，那里的采样器没有打补丁，
  rc3 记 `no_values`，可设 `VLLM_USE_V2_MODEL_RUNNER=0` 改回 V1。投机解码有草稿的步走 rejection sampler，r3 不采集，rc3 记 `lost`。
- **rc3 的口径**：采样器取的是原始 logits（vLLM 默认 `raw_logprobs` 的口径：allowed-token 掩码、bad words、logit bias、惩罚、温度、
  top-k/top-p 之前；结构化输出的 bitmask 在采样器之前，两者都包含），与 `--logprobs-mode` 的设置、请求要不要 logprobs 都无关。
  r2 依赖请求的 logprobs 时的两条限制（服务端必须是 `raw_logprobs`；同批有 `logprob_token_ids` 请求时第 1 列不是 top-1）在 r3 不再存在。
- **漏检**：周期 20–40 个 token、每份都有少量变化的短循环（例如带递增编号）会被新颖度守卫挡住，m4 报不出来。
- 召回和误报的口径见研究仓库 `detector_v35` 的 DESIGN.md §21。

## 10. 乱码监控 rc3 的在线接入

rc3（`RareDetector`、`CONFIG_RC3`，接口见接口说明 §9）接在 m4 的同一组钩子上：0.2.0（交接包 r2）靠请求带 logprobs 取数；
0.3.0（交接包 r3）改为在 vLLM 的采样器里采集，请求不需要带 logprobs（§10.2）。

### 10.1 定位：只监控，不拦截

- 严格档（误报率上限 0.1%）在 Qwen3 holdout 上误报率 0.213%，超过上限；监控档（上限 0.5%）在上限内（0.434%）。
- 所以 `V35_RARE` 没有拦截模式（`off` / `collect` / `shadow`）：报警只写事件日志（`rare_alert`），用来统计乱码发生率和告警，从不结束请求。
- m4 的拦截不受影响：可以 `V35_DETECTOR=intercept` 与 `V35_RARE=shadow` 同时开。

### 10.2 上线方式（r3：采样器采集）

rc3 每个生成 token 要两个数：原始分布（`raw_logprobs` 口径）里 top-1 的 logprob、生成 token 自己的 logprob。r3 按 9 月 v3.1 的做法在
采样器里用 Triton 规约算出它们，只保留 v3.1 的"快行"部分（top-1 与生成 token），去掉 top-20 与门控：

1. **采集（GPU，EngineCore 进程）**：`Sampler.forward` 在处理器之前留住原始 logits（bf16 / fp16 的 logits 采样器会先转成 float32 副本再处理，
   原张量不变，直接引用；float32 的 logits 会被原地修改，才复制一份）；采样后两个 Triton 核（`v35_detector/serving/collect.py`）：
   - 核 1，网格（行，4096 一段）：每段的最大值 m 与 Σexp(x − m)；
   - 核 2，网格（行）：合并成 M 与 S = Σexp(x − M)，读生成 token 的原始 logit x_c，输出 float32 `[行, 2]`：
     `top1 = M − lse = −log S`，`chosen = x_c − lse = (x_c − M) − log S`。
   - 不写整词表的 float32 logprob，不算排名，不排序；读写都按边界掩码。
2. **传输**：`[行, 2]` 挂在 SamplerOutput 上 → GPUModelRunner 放进 `ModelRunnerOutput.v35_rare`（异步调度时在输出拷贝流上与 sampled token
   一起非阻塞拷回，`get_output()` 里转成列表）→ Scheduler 把每个请求那一行放进它的 `EngineCoreOutput.v35_rare`（`[top1, chosen]`）
   → msgpack 到 API server → feed 钩子。客户端可见的 logprobs 路径一行不动；不要 logprobs 的请求，响应里也没有任何多出的内容。
3. **开关**：`V35_RARE` 由 EngineCore 进程从 API server 继承（TP=1 时 UniProcExecutor，采样就在 EngineCore 进程里）。
   - `off`（默认）：采样器只多一次 `is not None` 判断，不做任何 GPU 工作；m4 的路径与 r2 / h20-r1 相同；
   - `collect`：采集与传输照做，前端只记覆盖（`finish` 的 `rare`），不跑 rc3。用来单独测采集的成本（profile collect_only）；
   - `shadow`：采集，并在前端跑 rc3。词表沿用 `V35_DETECTOR_TABLE`（只接受内置名），运行方式沿用 `V35_DETECTOR_RUNNER`：
     sync 时 rc3 在输出处理循环里算，async 时 token 与两个值按 64 个一块一起入队，检测进程里同时跑 m4 与 rc3。
4. **覆盖**：`finish` 事件的 `rare` 应全是 `checked`（每个 token 都有值）。`no_values`：引擎没送值；`lost`：中途断了（§8、§9）。
5. **出错时放行**：采集出任何 Python 层面的错误，这一步就没有值（server.log 记 `v35 rc3 collection failed`，最多记 3 次），采样照常；
   rc3 对受影响的请求记 `lost`。核内的读写都按边界掩码，越界的 token id 得到 NaN，不会读越界。
6. **不再保留 r2 的 logprobs 取数作为后备**：值只有一个来源，口径固定是原始分布、与请求和 `--logprobs-mode` 无关；
   后备只能覆盖带 logprobs 的请求，还会让同一请求的值混用两种来源。采样器采不到的场景（V2 model runner、投机解码）明确记为 `no_values` / `lost`。
7. **观察**：先看监控档（`tier=monitor`）的报警率与按 `reason` 的分布，并抽样复核报警的请求。

### 10.3 开销测试与结果（H20，交接包 r2，2026-10-22）：r2 的取数方式

r2 的取数方式（已由 r3 取代）：每个请求带 `logprobs=1`，feed 钩子读这一步的 `EngineCoreOutput.new_logprobs`（第 0 列生成 token、第 1 列 top-1）。

r1 测过 m4 本身（§7）。r2 要回答两个问题，所以对照组同时覆盖"线上请求是否已经带 logprobs"两种情况：

| profile | 请求带 logprobs | m4 | rc3 | 运行方式 | 作用 |
|---|---|---|---|---|---|
| baseline | 否 | 关 | 关 | — | 基线 |
| baseline_logprobs | `logprobs=1` | 关 | 关 | — | 减 baseline = **vLLM 自带 logprobs 路径的开销**（r2 的取数方式；线上请求原本不带 logprobs 时要付的） |
| rc3_shadow_sync | `logprobs=1` | 关 | shadow | sync | 减 baseline_logprobs = **rc3 的增量**（线上请求已经带 logprobs 时只付这一项） |
| rc3_shadow_async | `logprobs=1` | 关 | shadow | async | 同上，异步 |
| m4_rc3_sync | `logprobs=1` | shadow | shadow | sync | 减 baseline_logprobs = m4 + rc3 一起的增量 |
| m4_rc3_async | `logprobs=1` | shadow | shadow | async | 同上，异步 |
| baseline_end | 否 | 关 | 关 | — | 减 baseline = 本次测试的漂移（分辨率） |

- 输出只用 1K（r1 的 8K 用了 `ignore_eos`，多数输出退化成循环），并发 1 / 8 / 32 / 64，每档 1 个预热、2 个正式 wave。
- 记录：服务端与客户端 TPOT、吞吐、服务端 CPU（整棵进程树与 API server 进程）、API server 与检测进程的 PSS、抢占数，
  m4 与 rc3 各自每 token 的检测耗时（事件日志 `summary`）；每个 logprob 请求都核对返回了逐 token 的 logprob，rc3 覆盖了每个请求。
- `bench/summarize_tpot.py` 的 r2 表格直接给出"vLLM logprobs 路径的开销"（`logprobs_cost`）与"rc3 的增量"（`+<profile>`）两类列，旁边是漂移。
- 冒烟（§7 ①）另外在真实模型上确认：m4 照常拦截；rc3 对强制的非法 UTF-8 报 `bad_utf8`，请求跑满不被结束；不带 logprobs 的请求 rc3 跳过。
- CPU 基准（§7 ②）另给出 rc3 每 token 开销（合成 logprob，通过学习打分门限的 token 比真实输出多，数字偏高、偏保守）、
  钩子整体每 token 开销（m4、rc3、两者、仅缓冲）和 rc3 的静态内存。

**r2 结果（2026-10-22，原始结果在研究仓 `report/data/h20_20261022/`，数字见其 `report/numbers.md` §14.7）**：

| 并发 | baseline TPOT | 漂移 | vLLM 自带 logprobs 路径（请求带 logprobs=1） | rc3 增量 sync / async | m4 + rc3 增量 sync / async |
|---|---|---|---|---|---|
| 1 | 4.381 ms | +4.9 μs | +136.1 μs（+3.11%） | −2.2 / −0.7 μs | −0.6 / −1.2 μs |
| 8 | 9.412 ms | −2.9 μs | +147.3 μs（+1.57%） | +1.3 / +1.4 μs | +1.5 / +1.1 μs |
| 32 | 15.751 ms | −0.3 μs | +206.4 μs（+1.31%） | −7.2 / −0.5 μs | +3.3 / +3.2 μs |
| 64 | 22.121 ms | −0.6 μs | +363.8 μs（+1.64%） | −3.4 / −9.0 μs | +4.8 / −15.6 μs |

- 冒烟、覆盖全部通过：m4 在第 1,984 个 token 拦截；rc3 对强制的非法 UTF-8 报 `bad_utf8` 而请求跑满；不带 logprobs 的请求 rc3 跳过；
  每组 315 个请求都是 `checked`，无 `error`、无抢占。
- **rc3 本身**：TPOT 增量测不出；检测器每 token 3.7–4.3 μs（真实生成流），m4 + rc3 合计 sync 11.8、async 13.9 μs；
  sync 时 API server 的 PSS 多 71–82 MB，async 时检测进程 102–105 MB。
- **取数是主要成本，而且是 r2 取数方式的成本**：`baseline_logprobs − baseline` 是 vLLM 自带 logprobs 路径
  （整词表 float32 log-softmax、top-k、遍历全表算排名、k+1 列传到前端、前端的 Logprob 对象与响应 JSON）的开销，不是 rc3 必须付的。
  9 月的 V31 在线报告（REPORT_V31_ZH.html §5，同一块 H20、同一模型）里，采样器里的 Triton top-1 规约为 C=1 +26.4、C=8 +15.1 μs/token，
  vLLM 自带路径取 top-20 为 +141.6 / +167.9 μs/token（都不返回给客户端、含当时的检测器）。返回与否、k 取 1 还是 20 差别不大；
  所以 r3 改为在采样器里采集（§10.2），开销见 §10.4。

### 10.4 r3 的开销测试、判读与结果（H20，交接包 r3，2026-10-22）

r3 要回答：在采样器里采集要多少钱，rc3 在它之上再加多少，与 r2 的取数方式（vLLM 自带 logprobs 路径）相比如何。
除 baseline_logprobs 外，所有 profile 的请求都不带 logprobs：

| profile | 请求带 logprobs | `V35_RARE` | m4 | 运行方式 | 作用 |
|---|---|---|---|---|---|
| baseline | 否 | off | 关 | — | 基线 |
| collect_only | 否 | collect | 关 | — | 减 baseline = **采集成本**（采样器的两个 Triton 核、拷回、调度器与前端的传输） |
| rc3_sync | 否 | shadow | 关 | sync | 减 collect_only = **rc3 的增量** |
| rc3_async | 否 | shadow | 关 | async | 同上，异步 |
| m4_rc3_sync | 否 | shadow | shadow | sync | 减 collect_only = m4 + rc3 一起的增量 |
| m4_rc3_async | 否 | shadow | shadow | async | 同上，异步 |
| baseline_logprobs | `logprobs=1` | off | 关 | — | 减 baseline = **r2 的取数方式**（vLLM 自带 logprobs 路径），作对照 |
| baseline_end | 否 | off | 关 | — | 减 baseline = 本次测试的漂移（分辨率） |

- 协议同 r2：1K 输出，并发 1 / 8 / 32 / 64，每档 1 个预热、2 个正式 wave，所有请求生成满长度；8 个 profile 约 30 分钟。
- 覆盖：采集类 profile 里每个请求都必须是 `checked`；除 baseline_logprobs 外，响应里出现任何 logprob 都算失败（不多返回东西）。
- `runs/summary.txt` 开头的 r3 表格：
  - `collect` 列 = collect_only − baseline，**这是 r3 要和 r2 的 `logprobs_path` 列（baseline_logprobs − baseline，r2 实测 +136 / +147 / +206 / +364 μs）比的数**，
    下面一行给出两者占 baseline TPOT 的百分比；
  - `+rc3_sync` 等列 = 各 profile − collect_only，即 rc3（与 m4）本身，r2 实测测不出；
  - `drift` 列是两次 baseline 之差，比它小的差值都不可区分；
  - 同样的差值另有客户端 TPOT、吞吐、CPU（整棵进程树、API server 进程）几张表。
- `runs/collect_gpu_check.json`（gpucheck 步骤）：Triton 核对 `torch.log_softmax` 的最大绝对误差（应在 1e-5 量级，容差 1e-3），
  以及行数 1 / 8 / 32 / 64 时每次调用的 CPU（发射）与 GPU 耗时，旁边是 vLLM 自己 logprobs=1 那几个算子的耗时。
- 冒烟（`runs/smoke/smoke.txt`）里 `sampler values vs vLLM logprobs` 两行给出采样器的值与 vLLM 返回的 raw logprobs 之差（应在 1e-5 量级）。
- 9 月 v3.1 的参照（同一块 H20、同一模型）：采样器 Triton top-1 规约 + 当时的检测器 C=1 +26.4、C=8 +15.1 μs/token。

**r3 结果（原始结果在研究仓 `report/data/h20_20261022_r3/`，数字见其 `report/numbers.md` §14.8）**：

| 并发 | baseline TPOT | 漂移 | 采集（collect_only − baseline） | r2 的取数方式（同一轮） | rc3 增量 sync / async | m4 + rc3 增量 sync / async | 全部打开（m4 + rc3 sync − baseline） |
|---|---|---|---|---|---|---|---|
| 1 | 4.382 ms | +1.8 μs | +3.5 μs（+0.08%） | +134.2 μs（+3.06%） | +0.9 / −0.3 μs | −1.0 / −0.6 μs | +2.4 μs |
| 8 | 9.406 ms | +0.8 μs | +7.0 μs（+0.07%） | +157.1 μs（+1.67%） | +2.9 / +0.6 μs | +2.5 / +0.6 μs | +9.5 μs |
| 32 | 15.743 ms | +18.5 μs | +29.1 μs（+0.18%） | +220.8 μs（+1.40%） | +0.4 / −3.2 μs | +8.2 / −9.8 μs | +37.3 μs |
| 64 | 22.114 ms | +5.9 μs | +18.6 μs（+0.08%） | +357.8 μs（+1.62%） | +4.3 / +0.3 μs | +8.3 / −5.0 μs | +26.9 μs |

- 全部检查通过：RECORD 只差 5 个文件的 35 行钩子；gpucheck 72 组，最大误差 6.7e-7（top-1）、1.9e-6（生成 token）；冒烟时采样器的值与
  vLLM 返回的 logprobs 相差不到 1e-6（含 T=1 采样、生成 token 不是 top-1 的情况）；8 个 profile 共 2,520 个请求，0 抢占，采集类 profile 每个请求都是
  `checked`，除 baseline_logprobs 外客户端收到的 logprob 为 0。
- **采集**比 r2 的取数方式便宜 10–40 倍：吞吐下降 0.07–0.15%（r2 的方式 1.3–2.9%），CPU 多 0.004–0.015 核（与漂移相当）。
  并发 32 的漂移为 +18.5 μs，那一档只能说采集不超过约 30 μs。gpucheck 里两个核单独调用约 30 μs/次且不随行数变化，服务里与 GPU 计算重叠，实际每步只多上表的量。
- **rc3 与 m4** 在采集之上的增量与漂移同量级；m4 + rc3 sync 在高并发下约 +8 μs（客户端约 +14 μs）。检测器每 token rc3 3.9–5.2 μs、m4 8.7–9.2 μs，
  合计 sync 约 12.6、async 约 13.2 μs。
- **内存**：只采集时 API server 不变；sync 时 API server 多约 58 MB，async 时检测进程 101–105 MB。
- 两个 server.log 在关服务（SIGTERM）之后有 `EngineDeadError`，与 r1、r2 相同，属于 vLLM 关服时序。

### 10.5 r4：rc4 与 rc3 的同轮对比（H20，交接包 r4，2026-10-22 UTC）

rc4（接口说明 §9.8）与 rc3 的输入和采集完全相同，r3 的 5 个文件、35 行补丁不变；只多一个开关 `V35_RARE_CONFIG=rc3|rc4`（默认 rc3）。
r4 要回答：rc4 在线的开销是否不高于 rc3（预期更低：学习打分只在约 0.2% 的 token 上算，不查参照表）。

| profile | `V35_RARE` | `V35_RARE_CONFIG` | m4 | 运行方式 | 作用 |
|---|---|---|---|---|---|
| baseline | off | — | 关 | — | 基线 |
| collect_only | collect | — | 关 | — | 减 baseline = 采集成本（与 r3 对照） |
| rc3_sync | shadow | rc3 | 关 | sync | 减 collect_only = rc3 的增量（同轮参照） |
| rc4_sync | shadow | rc4 | 关 | sync | 减 collect_only = **rc4 的增量** |
| rc4_async | shadow | rc4 | 关 | async | 同上，异步 |
| m4_rc4_sync | shadow | rc4 | shadow | sync | 减 collect_only = m4 + rc4 一起的增量 |
| baseline_end | off | — | 关 | — | 减 baseline = 漂移 |

- 协议同 r3：1K 输出，并发 1 / 8 / 32 / 64，每档 1 个预热、2 个正式 wave；7 个 profile 约 27 分钟。r2 的 logprobs 路径已测过，不再测。
- 判读：`runs/summary.txt` 开头的表里，`+rc3_sync` 与 `+rc4_sync` 是同一轮、同一采集之上的增量，可以直接比；比漂移小的差都不可区分。
  另看每个 profile 的检测器自身耗时（`rc3 x us/token`、`rc4 x us/token`，真实生成流）与内存（sync 时 API server 的 PSS、async 时检测进程的 PSS）。
- CPU 基准（`runs/cpu_bench_h20.json`）：`rare_static_mb` / `rc4_static_mb` 是两者的 tracemalloc 常驻内存（各在新进程里测）；
  `hooks_us_per_token` 的 `rc3` / `rc4` / `m4_rc4` 是同一批英文式合成流上钩子整体的每 token 开销（两者都不报警，可比）。
  `rc4_us_per_token` 在混杂文字的合成流上很快报警停下，只覆盖少量 token，不用于比较。
- 冒烟默认用 rc4：m4 照常拦截；rc4 对强制的非法 UTF-8 报 `bad_utf8` 而请求跑满；每个请求的 `rare` 都是 `checked`；采样器的值与 vLLM 返回的
  logprobs 一致。

**r4 结果（原始结果在研究仓 `report/data/h20_20261022_r4/`，数字见其 `report/numbers.md` §14.9）**：

| 并发 | baseline TPOT | 漂移 | 采集 | +rc3 sync | +rc4 sync | +rc4 async | +m4+rc4 sync |
|---|---|---|---|---|---|---|---|
| 1 | 4.389 ms | −8.4 μs | −2.5 μs | +0.3 μs | −0.5 μs | −0.2 μs | −0.4 μs |
| 8 | 9.388 ms | +0.4 μs | +7.1 μs | −1.1 μs | −1.8 μs | −2.1 μs | −1.9 μs |
| 32 | 15.691 ms | +53.2 μs | +54.4 μs | +6.7 μs | +10.2 μs | +11.5 μs | −2.2 μs |
| 64 | 22.122 ms | +44.3 μs | +63.1 μs | −1.4 μs | −12.0 μs | +3.5 μs | −8.1 μs |

- 全部检查通过：冒烟（rc4）、GPU 核对、7 个 profile 无抢占、每个请求 `checked`；单元测试 196 项（含真实 vLLM 文件 201 项）。
- 这一轮并发 32 / 64 的漂移为 +53 / +44 μs，那两档分辨不出；并发 1 / 8 上 rc3 与 rc4 的增量都在 ±2 μs 以内，采集与 r3 一致。
- 检测器自身每 token（真实生成流）：rc4 2.40（sync）、3.15（async）、与 m4 同开时 2.00 μs；同一轮 rc3 4.68 μs。m4 + rc4 合计约 11.0 μs。
- 内存：sync 时 API server 的 PSS rc4 比 rc3 少约 17 MB；async 检测进程 rc4 43 MB（r3 里 rc3 101–105 MB）；静态 rc4 21.7 MB、rc3 63.9 MB。
- 报警：rc4 的三个检测组都对同一个压测请求（固定 seed，第 915 个 token）报了严格档（`gb>2.07 & gap<0.3`），rc3 对它只报监控档；
  输出原文没有保存，不能判断真假。

### 10.6 rc1t：不经过采样器的 token-only 接入（0.5.0）

rc1t（接口说明 §9.9）只读原始输出 token id，适合与 m4 一样在 API server 检测，不需要 rc3/rc4 的 raw-logits
采集、GPU Triton 核或 logprob 传输：

```bash
V35_DETECTOR=intercept V35_DETECTOR_RUNNER=sync \
V35_RARE=shadow V35_RARE_CONFIG=rc1t V35_DETECTOR_TABLE=qwen3 \
V35_DETECTOR_EVENTS=/var/log/v35-events.jsonl vllm serve ...
```

- `sync` 在输出处理循环里与 m4 并行消费同一 token 流；`async` 也可把同一流送入独立检测进程。选择哪种方式取决于
  CPU 隔离和部署约束，rc1t 本身不要求 worker。m4 的 64-token 分块不变，rc1t 的向量化、一个 token 前瞻独立维护。
- `V35_RARE=collect` 和 `V35_RARE_TRACE` 不适用于 rc1t；选择 rc1t 时采样器不创建 Collector，前端收到
  `v35_rare=None` 是正常情况，请求覆盖仍应为 `checked`。rare_chars 只监控，m4 是否拦截由 `V35_DETECTOR` 决定。
- 上线顺序仍是先 `shadow` 观察，再按需打开 m4 的 `intercept`。尤其先检查正常多语言混合：源 labels_v3_3
  heldout 的 lang_mix bonus 有较高报警比例，不能把它等同于普通负例误报，也不能忽略实际拦截策略的影响。
- 本仓已完成合成流、sync/async 服务与全部 19,321 条 dev 的逐条等价验证；Mac 离线组合开销见接口说明 §9.9。
  **H20 在线 TPOT、CPU 和内存未测，DSV4 效果未测**，不能把 rc4 的 H20 结果直接套到 rc1t。

### 10.7 新模型

- 建 rare 词表（`python -m v35_detector.rare_vocab`）；
- rc3 的学习打分依赖用该模型带标签的正常输出统计的参照表，没有参照表时它不触发，只剩机制规则；rc4 不查参照表，
  rc1t 在没有参照训练数据时使用中性零值但仍运行 GB。在线钩子目前只接受内置词表名，新模型要先把它的表加进包里。

**上线前要看的**：
- 学习打分的提升是针对注入故障学到的，换一种故障或模型，召回预期回到前一版 rc2 的水平；
- 上线后先看监控档的报警率和按规则的分布，并抽样复核。
