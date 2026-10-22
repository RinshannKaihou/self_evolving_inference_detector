# 适配流程：把 m4 接入 vLLM 0.27（以及乱码监控 rc3 的在线接入，§10）

## 1. 接入位置

```text
API server 进程（可开多个，--api-server-count）
  AsyncLLM 的输出处理循环 → OutputProcessor.process_outputs
    反解码 → stop string 检查 → ★ v35_hooks.feed(token ids, new_logprobs)   ← 本补丁只改这个文件
    → logprobs 处理（请求要 logprobs 时） → 输出
    命中 → finish_reason='stop'、stop_reason='v35_repetition: …' → 前端结束请求 → 通知 EngineCore 中止
EngineCore 进程（调度主循环）            不改
GPU worker 进程（前向、采样）            不改
```

为什么放在这一层：
- **和 stop string 走同一条路径**：vLLM 自己就是在这里检测 stop string、结束请求，再通知 EngineCore 中止。拦截复用了这条已有的路径。
- **不占调度线程，也不占 GPU 步**：vLLM 原生的重复检查在调度线程里逐 token 跑，开销直接加在每个 decode step 上。
- **晚一两步不影响结果**：检测器本来就每 64 个 token 才判一次，从前端发中止到 EngineCore 生效多出的一两步，相对这个粒度可以忽略。
- **m4 只要 token id**：不需要 logprob，也不改 sampler。乱码监控 rc3 在同一个位置读这一步的 `new_logprobs`（请求带 `logprobs=1` 时才有），见 §10。

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
| 前端 CPU | 每 token 约 3.5 μs（Mac）、9.5 μs（H20 服务器实测），加在 asyncio 事件循环上 | 每 64 token 一次入队；检测进程每 token 10–14 μs（H20 实测） |
| 内存（CPU，不占显存） | 都在 API server 进程内：词表约 2.3 MB，加每个在途请求约 165 KB | 检测进程内：进程本身 PSS 26–30 MB（H20 实测，含词表），加每个在途请求约 165 KB；前端启动时也缓存一份词表，每个请求只缓存不满 64 个的 token id |
| 何时选 | 默认。前端 CPU 有余量时，拦截更及时 | 前端 CPU 已经吃紧（高 QPS、多路反解码）时 |

两种方式的报警判定完全相同，只是拦截生效的时刻不同。在线测试会测两者的 TPOT 和 CPU（§7）。
开启乱码监控 rc3（§10）时，它和 m4 在同一处算：rc3 的表每进程常驻约 64 MB，sync 时在 API server 进程里（启动时加载），
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
- 已经打过 r1 版补丁（feed 不传 logprobs）的环境：`--check` 会显示 `patched by an earlier version`，`--apply` 直接把旧的钩子行换成本版，
  不需要先 `--revert`。
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
  - `alert`（m4）按 `reason` 统计；`rare_alert`（rc3）按 `tier`、`reason` 统计；
  - `error` 应为 0；
  - `summary` 每 10 秒一行，含计数和检测耗时（m4、rc3 分开）；
  - 开启 rc3 时，`finish` 的 `rare` 应以 `checked` 为主；`no_logprobs` 多说明请求没带 logprobs。
- **建议的告警**：
  - 出现 `error`；
  - `counters.worker_down > 0`（异步进程挂了）；
  - 报警率相对基线突增（可能是模型或硬件故障，也可能是误报）。

## 7. 上线前验证与性能测试（H20）

推荐用一键脚本 `bench/run_h20.sh`：

```bash
FAKE=1 bash bench/run_h20.sh all                                         # 先用模拟服务端演练一遍，不占 GPU，几分钟
PY=$(which python) MODEL=/workspace/models/Qwen3-30B-A3B GPU=5 bash bench/run_h20.sh all   # 正式运行，约 40 分钟
```

- 脚本依次执行：env → install → patch → smoke → cpu → tpot1k → collect（r2 起不再跑 tpot8k，见下文 r1 结果的最后一条）。
- 每一步都可以单独运行，例如 `bash bench/run_h20.sh tpot1k`。
- 任何一步失败，都会先把已有结果打包再退出。
- r2 的对照组与判读方法见 §10.3。

下面是各步对应的手动命令。

```bash
cd /path/to/v35_m4_detector
python -m pytest -q
OP=$(python -c "import vllm,os;print(os.path.join(os.path.dirname(vllm.__file__),'v1','engine','output_processor.py'))")
V35_VLLM_OUTPUT_PROCESSOR=$OP python -m pytest -q tests/test_vllm_patch.py
```

**① 冒烟**：起一个 m4 `intercept`、rc3 `shadow` 的服务，确认三件事：
"重复一句话"的请求以 `v35_repetition` 结束；用 `logit_bias` 强制输出非法 UTF-8 的请求，rc3 报 `bad_utf8` 但请求照常跑满；
不带 logprobs 的请求 rc3 跳过。

```bash
V35_DETECTOR=intercept V35_RARE=shadow V35_DETECTOR_TABLE=qwen3 V35_DETECTOR_EVENTS=/tmp/v35-smoke.jsonl CUDA_VISIBLE_DEVICES=5 \
  vllm serve /workspace/models/Qwen3-30B-A3B --served-model-name bench --port 18790 &
python bench/smoke_intercept.py --url http://127.0.0.1:18790 --events /tmp/v35-smoke.jsonl --rare --table qwen3
```

**② 部署机 CPU 开销**：跑完把 `runs/cpu_bench_h20.json` 交回。

```bash
python bench/cpu_bench.py --vllm --out runs/cpu_bench_h20.json
```

**③ 在线 TPOT / CPU / 内存**：
- r2 默认 7 个 profile（§10.3）：baseline、baseline_logprobs、rc3_shadow_sync、rc3_shadow_async、m4_rc3_sync、m4_rc3_async、baseline_end；
- r1 的 profile（vllm_default、vllm_tuned、v35_sync、v35_async）仍可用 `--profiles` 指定；
- 检测器都用影子模式，所有请求都生成满长度，各 profile 的工作量完全相同。

```bash
# 1K 输出，并发 1 / 8 / 32 / 64
python bench/online_tpot.py --output runs/tpot_qwen3_1k --model /workspace/models/Qwen3-30B-A3B --table qwen3 --gpu 5 \
  --output-lengths 1024 --concurrencies 1 8 32 64
# r1 的对照组：
python bench/online_tpot.py --output runs/tpot_qwen3_1k_r1 --model /workspace/models/Qwen3-30B-A3B --table qwen3 --gpu 5 \
  --output-lengths 1024 --concurrencies 1 8 32 64 --profiles baseline vllm_default vllm_tuned v35_sync v35_async baseline_end
```

- **预计用时**：每个 profile 都要重启一次服务。r1 实测 1K 每个 profile 约 3.5 分钟（含启动），r2 的 7 个 profile 约 25–30 分钟。时间紧时可以改成 `--waves 1`。
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
| `finish` 的 `rare` 都是 `no_logprobs` | 请求没带 `logprobs`；rc3 只检查带 logprobs（≥ 1）的请求 |
| `rare` 出现 `lost` | 请求中途没了 logprobs，或 `logprobs=0`（没有 top-1）；见接口说明 §9.7 |

## 9. 已知限制

- **streaming input 请求**：vLLM 0.27 的 `streaming_input`（输入分段送达）每结束一段都会经过 finish 钩子，所以第一段结束后检测就停了。普通请求不受影响。
- **未在线验证的场景**：投机解码（MTP）、流水线并行。
- **rc3 的 logprob 口径**：要求服务端用默认的 `--logprobs-mode raw_logprobs`；同一批里有 `logprob_token_ids`（generative scoring）请求时，
  该批的第 1 列不是 top-1（接口说明 §9.7）。
- **漏检**：周期 20–40 个 token、每份都有少量变化的短循环（例如带递增编号）会被新颖度守卫挡住，m4 报不出来。
- 召回和误报的口径见研究仓库 `detector_v35` 的 DESIGN.md §21。

## 10. 乱码监控 rc3 的在线接入

rc3（`RareDetector`、`CONFIG_RC3`，接口见接口说明 §9）接在 m4 的同一组钩子上（0.2.0，交接包 r2）。

### 10.1 定位：只监控，不拦截

- 严格档（误报率上限 0.1%）在 Qwen3 holdout 上误报率 0.213%，超过上限；监控档（上限 0.5%）在上限内（0.434%）。
- 所以 `V35_RARE` 只有 `off` / `shadow`：报警只写事件日志（`rare_alert`），用来统计乱码发生率和告警，从不结束请求。
- m4 的拦截不受影响：可以 `V35_DETECTOR=intercept` 与 `V35_RARE=shadow` 同时开。

### 10.2 上线方式

1. **补丁**：同 §4。feed 钩子把这一步的 `engine_core_output.new_logprobs` 一并传给 `v35_hooks.feed`（vLLM 0.27.0 源码核对：
   `EngineCoreOutput.new_logprobs: LogprobsLists | None`，钩子位于 logprobs 处理之前）。已打 r1 补丁的环境，`--apply` 直接替换旧钩子行。
2. **让请求带 logprobs**：rc3 只检查带 `logprobs` ≥ 1 的请求。
   - 每个请求带 `logprobs=1`（OpenAI completions 接口的 `logprobs: 1`，chat 接口的 `logprobs: true, top_logprobs: 1`）；
   - 不带的请求 rc3 跳过（`finish` 的 `rare` 为 `no_logprobs`），m4 照常；
   - 服务端统一打开 logprobs 而不改请求，本版没有实现。
3. **开关**：`V35_RARE=shadow`，词表沿用 `V35_DETECTOR_TABLE`（只接受内置名），运行方式沿用 `V35_DETECTOR_RUNNER`：
   sync 时 rc3 在输出处理循环里算，async 时 token 与两个 logprob 按 64 个一块一起入队，检测进程里同时跑 m4 与 rc3。
4. **观察**：先看监控档（`tier=monitor`）的报警率与按 `reason` 的分布，并抽样复核报警的请求。
5. **口径**：服务端保持默认 `--logprobs-mode raw_logprobs`。

### 10.3 开销测试（H20，交接包 r2，待测）

r1 测过 m4 本身（§7）。r2 要回答两个问题，所以对照组同时覆盖"线上请求是否已经带 logprobs"两种情况：

| profile | 请求带 logprobs | m4 | rc3 | 运行方式 | 作用 |
|---|---|---|---|---|---|
| baseline | 否 | 关 | 关 | — | 基线 |
| baseline_logprobs | `logprobs=1` | 关 | 关 | — | 减 baseline = **打开 logprobs 本身的开销**（线上请求原本不带 logprobs 时要付的） |
| rc3_shadow_sync | `logprobs=1` | 关 | shadow | sync | 减 baseline_logprobs = **rc3 的增量**（线上请求已经带 logprobs 时只付这一项） |
| rc3_shadow_async | `logprobs=1` | 关 | shadow | async | 同上，异步 |
| m4_rc3_sync | `logprobs=1` | shadow | shadow | sync | 减 baseline_logprobs = m4 + rc3 一起的增量 |
| m4_rc3_async | `logprobs=1` | shadow | shadow | async | 同上，异步 |
| baseline_end | 否 | 关 | 关 | — | 减 baseline = 本次测试的漂移（分辨率） |

- 输出只用 1K（r1 的 8K 用了 `ignore_eos`，多数输出退化成循环），并发 1 / 8 / 32 / 64，每档 1 个预热、2 个正式 wave。
- 记录：服务端与客户端 TPOT、吞吐、服务端 CPU（整棵进程树与 API server 进程）、API server 与检测进程的 PSS、抢占数，
  m4 与 rc3 各自每 token 的检测耗时（事件日志 `summary`）；每个 logprob 请求都核对返回了逐 token 的 logprob，rc3 覆盖了每个请求。
- `bench/summarize_tpot.py` 的 r2 表格直接给出"打开 logprobs 的开销"（`logprobs_cost`）与"rc3 的增量"（`+<profile>`）两类列，旁边是漂移。
- 冒烟（§7 ①）另外在真实模型上确认：m4 照常拦截；rc3 对强制的非法 UTF-8 报 `bad_utf8`，请求跑满不被结束；不带 logprobs 的请求 rc3 跳过。
- CPU 基准（§7 ②）另给出 rc3 每 token 开销（合成 logprob，通过学习打分门限的 token 比真实输出多，数字偏高、偏保守）、
  钩子整体每 token 开销（m4、rc3、两者、仅缓冲）和 rc3 的静态内存。

**已知的数**（Mac，Apple M3 Pro，单线程）：rc3 离线 0.76 μs/token（真实 dev 记录）；钩子自身缓冲约 0.5 μs/token；
每进程常驻约 64 MB（不是此前写的 33 MB，接口说明 §9.5）。服务器上的数和 TPOT 影响等 r2 结果。

### 10.4 新模型

- 建 rare 词表（`python -m v35_detector.rare_vocab`）；
- 学习打分依赖用该模型带标签的正常输出统计的参照表，没有参照表时它不触发，只剩机制规则（DeepSeek 就是这种情况）。
  在线钩子目前只接受内置词表名，新模型要先把它的表加进包里。

**上线前要看的**：
- 学习打分的提升是针对注入故障学到的，换一种故障或模型，召回预期回到前一版 rc2 的水平；
- 上线后先看监控档的报警率和按规则的分布，并抽样复核。
