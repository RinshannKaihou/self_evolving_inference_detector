# 接口说明：v35 检测器（循环输出 m4、乱码监控 rc3）

版本 0.3.0。判定逻辑与研究仓库 `detector_v35` 被评测的版本逐字节相同（`src/v35_detector/MANIFEST.json`）。§1–§8 是循环输出检测器 m4；§9 是乱码检测器 rc3（离线接口，以及同一组 vLLM 钩子上的在线监控）。

## 1. 适用范围

- 检测对象：taxonomy v3.5 的 `repetition` core，即模型陷在循环里：
  - 逐字循环；
  - 跑到上限的重复墙；
  - 推理原地打转；
  - 空白或单字符填充。
- 已评测的模型：Qwen3-8B / 30B-A3B / 32B（思考模式）、GLM-5.3-Flash（思考模式，MTP）、DeepSeek-V4-Flash（非思考模式）。
  - 换模型时阈值不需要重调，只需建词表（见适配流程文档 §5）。
- 乱码（rare_chars）由另一个检测器 rc3 负责，见 §9。它需要两个 logprob 输入（top-1 与生成 token 的；在线时由打过补丁的 vLLM 采样器算出，
  请求不需要带 logprobs），只做监控、不拦截。

## 2. 安装与依赖

```bash
pip install -e <仓库路径>          # Python >= 3.10，只依赖 numpy
pip install -e '<仓库路径>[vocab]'  # 为新模型建词表时另需 tokenizers
```

## 3. API

```python
from v35_detector import RepDetector, Result, CONFIG_M4

det = RepDetector(corpus, config=CONFIG_M4, think_end='auto')
st  = det.start()                  # 每个请求一份状态
res = det.feed(st, token_ids)      # 报警时返回 Result，否则 None
res = det.finish(st)               # 输出结束：处理最后不满 64 个 token 的一块，返回 Result（未报警时 is_ill=False）
mon = det.monitor(st)              # 监控档的首次报警；m4 没有监控档组件，结果与拦截档相同
```

| 参数 | 说明 |
|---|---|
| `corpus` | 内置词表名 `qwen3` / `glm` / `dsv4`，或 `vocab.py` 生成的 `.npz` 路径 |
| `config` | 配置 dict 或 JSON 路径；默认 `CONFIG_M4`；`None` 为探针模式（逐判定点记下全部特征，供分析用） |
| `think_end` | 回答段起点的 token id（`</think>`）。`'auto'` 只适用于内置词表（qwen3 151668、glm 154842、dsv4 无）；`None` 表示非思考模式 |

`Result` 的字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `is_ill` | bool | 是否报警 |
| `fired_at` | int | 报警所在的判定点，即此时已消费的 token 数；未报警为 −1 |
| `reason` | str | 触发报警的组件（见 §6），未报警为空串 |

**线程与生命周期**：
- `RepDetector` 只持有只读的词表和配置，可以在多个请求之间共用。
- 每个请求的 `State` 只能由一个线程按顺序 feed。
- 请求结束后丢弃 `State` 即可，不需要显式释放。

## 4. 输入约定

- **只喂生成的 token id**，不喂 prompt；整数列表或 numpy 数组均可。
- **块大小任意**，可以是每步 1 个，也可以是投机解码的多个，或一整段。检测器内部按 64 个 token 一块处理，结果与分块方式无关（有测试保证）。
- **回答段规则**：第一个 `think_end` 之后，n-gram 只往前数到回答开头。回答复述思考段内容是合法的，回答内部的循环照样计数。
- 停止 token（EOS）喂进去也没关系；评测数据里正常结束的记录都以它结尾。

## 5. 输出语义

- 每 64 个 token 设一个判定点，输出结束时（`finish`）再设一个。
- 报警条件：任一组件的条件在连续 S 个判定点都成立。按配置顺序检查，最先满足的组件写进 `reason`。
- 报警之后状态即结束（`st.done`）：之后再调用 `feed`，返回的仍是同一个 `Result`。
- `fired_at` 总是判定点（64 的倍数，或 `finish` 时的总长度）。对应的 token 已经生成，所以"报警时已经多生成了多少"以这个数为准。

## 6. 配置（`configs/m4.json`）

组件类型：

| kind | 字段 | 条件 |
|---|---|---|
| `M` | `feat`、`theta`、`S`，可选 `guard`、`nov`、`and` | `feat ≥ theta`；可选：分隔符占比 ≤ guard，新颖度 ≤ nov.max，其他特征的上下限（and） |
| `V` | `mn`、`mx`、`c` | 与 vLLM `check_sequence_repetition([mn, mx] × c)` 逐 token 等价，在其后的第一个判定点上报 |
| `RUN` | `min_chars` | 同一个非空白字符连续 ≥ min_chars 个字符 |
| `WS`、`G` | — | m4 未使用（空白占比；梯度提升树） |

m4 的 14 个组件，以及 `reason` 的字符串：

| # | reason | 族 | 持续（判定点 / 约 token 数） |
|---|---|---|---|
| 1 | `vllm[32,500]x5` | vLLM 同规则（被 #2 覆盖，从不先报） | 1 |
| 2 | `vllm[32,2000]x3` | vLLM 同规则（dev 调参） | 1 |
| 3 | `run>=500` | 单字符填充 | 1 |
| 4 | `A3_n4_W1024>=0.6 S8 N_n4_W512<=0.05` | 逐字 n-gram 份数 | 8 / 512 |
| 5 | `A5_n4_W512>=0.7 S4 N_n4_W512<=0.1` | 逐字 n-gram 份数 | 4 / 256 |
| 6 | `A5_n4_W512>=0.5 S4 sep<=0.5 N_n4_W512<=0.05` | 逐字 n-gram 份数 | 4 / 256 |
| 7 | `P_n8_W1024>=0.6 S24 N_n4_W512<=0.02` | 周期结构 | 24 / 1,536 |
| 8 | `wrep_W2048>=0.95 S16 & mark_W1024>=0.05` | 词汇无进展 | 16 / 1,024 |
| 9 | `wrep_W2048>=0.99 S24` | 词汇无进展 | 24 / 1,536 |
| 10 | `wrep_W1024>=0.99 S8 & mark_W1024>=0.04` | 词汇无进展 | 8 / 512 |
| 11 | `wrep_W1024>=0.98 S8 & mark_W1024>=0.05` | 词汇无进展 | 8 / 512 |
| 12 | `sdup_N16>=0.7 S24 & wrep_W2048>=0.9 & mark_W1024>=0.03` | 句子复述 | 24 / 1,536 |
| 13 | `sdup_N16>=0.6 S40 & mark_W1024>=0.03` | 句子复述 | 40 / 2,560 |
| 14 | `wrep_W2048>=0.95 S48 & mark_W1024>=0.03` | 词汇无进展 | 48 / 3,072 |

特征定义在 `core.py` 的模块说明里：
- A3 / A5：至少第 4 / 第 6 份；
- P：周期性；
- N：新颖度；
- wrep：词复现；
- mark：绕圈标记密度；
- sdup：句子复述。

配置与词表的 sha256 记在 MANIFEST 里，改动后测试会失败。要换阈值，就是另一个检测器，需要重新评测。

## 7. 资源与性能

| 项 | 数值 | 说明 |
|---|---|---|
| 每 token CPU | Mac 3.4–3.6 μs；H20 服务器 6.67 μs（离线），线上 9.5 μs（sync）、10.1–14.0 μs（async） | 单线程。Mac 为 Apple M3 Pro：真实 dev 数据 3.43（研究仓库），合成数据 3.60（`bench/cpu_bench.py`）。服务器为 Xeon 8558P，2026-10-21 实测 |
| 每请求状态 | ≤ 166 KB | 有上界：窗口填满后不再增长（`tests/test_state.py`） |
| 词表 | 每个 tokenizer 约 2.3 MB 内存 | 每进程加载一次后缓存 |
| 对照：vLLM 原生检查 | Mac 84.6 μs（[50,500]×3）、259.8 μs（[32,2000]×3）；H20 服务器 92.0、248.6 μs | 同一台机器上，用上游原代码、按逐 token 调用测得 |
| 线上 TPOT | H20 单卡 Qwen3-30B-A3B：变化在 ±22 μs/token（≤ 0.14%）以内 | 与两次 baseline 之差同量级，即小于测量分辨率（`docs/INTEGRATION.md` §7） |

## 8. 服务钩子（`v35_detector.serving.hooks`）

由打过补丁的 vLLM OutputProcessor 调用，见适配流程文档：

| 函数 | 调用时机 | 返回 |
|---|---|---|
| `register(request_id)` | 请求进入 OutputProcessor | — |
| `feed(request_id, token_ids, rare=None)` | 每个引擎步的新 token（stop string 检查之后），以及这一步的 `EngineCoreOutput.v35_rare`：采样器算出的 rc3 输入，`[top1_0, chosen_0, top1_1, chosen_1, …]`，每 token 两个自然对数 logprob；引擎没采集时为 None（见 §9.7） | m4 拦截模式下报警时返回 `'v35_repetition: <reason>'`，否则 None；rc3 从不返回停止原因 |
| `finish(request_id, finish_reason)` | 请求以任何原因结束 | — |
| `abort(request_id)` | 请求被取消，或引擎出错 | — |

环境变量（服务启动、导入模块时读取；配置错误会让服务启动失败）：

| 变量 | 取值 | 默认 |
|---|---|---|
| `V35_DETECTOR` | m4：`off` / `shadow`（只检测、记日志）/ `intercept`（检测、记日志并结束请求） | `off` |
| `V35_RARE` | rc3：`off` / `collect`（采样器采集并送到前端，只记覆盖，不跑 rc3）/ `shadow`（采集，跑 rc3，只记日志；没有拦截模式）。EngineCore 进程读同一个变量决定采样器是否采集，见 §9.7 | `off` |
| `V35_RARE_TRACE` | N > 0 时，每个请求前 N 个 token 收到的值写进 `rare_trace` 事件（冒烟测试用来与 vLLM 返回的 logprobs 对比） | `0` |
| `V35_DETECTOR_TABLE` | `qwen3` / `glm` / `dsv4` / `.npz` 路径（m4 词表）；rc3 只接受内置名 | m4 开启或 `V35_RARE=shadow` 时必填（只 `collect` 时不需要） |
| `V35_DETECTOR_THINK_END` | `auto` / `none` / token id（两个检测器共用） | `auto` |
| `V35_DETECTOR_RUNNER` | `sync`（在输出处理循环里算）/ `async`（独立检测进程）；两个检测器在同一处算 | `sync` |
| `V35_DETECTOR_CONFIG` | m4 配置 JSON 路径（rc3 固定用内置 rc3） | 内置 m4 |
| `V35_DETECTOR_EVENTS` | 事件日志路径（JSONL） | `./v35-detector-events.jsonl` |

事件日志每行一个 JSON，公共字段为 `ts`、`pid`、`event`：

| event | 其他字段 | 何时写 |
|---|---|---|
| `alert` | `request_id`、`fired_at`、`reason`、`tokens_seen`、`action`（`shadow` / `intercept`）；或 `after_end: true` | m4 报警。`after_end` 表示请求已经结束（最后不满一块的判定，或异步结果晚到） |
| `rare_alert` | `request_id`、`tier`（`monitor` / `strict`）、`fired_at`、`reason`（rc3 的规则）、`tokens_seen`、`action`（`shadow`）；或 `after_end: true` | rc3 报警，见 §9.7。每个请求每档至多一行；从不结束请求 |
| `finish` | `request_id`、`tokens`、`finish_reason`；开启 rc3（collect 或 shadow）时另有 `rare`（rc3 对该请求的覆盖，见 §9.7） | 每个请求结束 |
| `rare_trace` | `request_id`、`token_ids`、`top1`、`chosen`（前 N 个 token 收到的值） | 只在 `V35_RARE_TRACE` > 0 时，请求结束时写 |
| `error` | `request_id`、`where`、`error` | 检测器出错。该检测器对该请求停止检测，另一个检测器与服务照常进行。`where` 以 `rare` 开头的是 rc3 |
| `summary` | `counters`、`detector_seconds_sync`（m4）、`worker`、`open_requests`；开启 rc3 时另有 `rare`、`rare_seconds_sync`（rc3），`worker` 里另有 `rare_tokens`、`rare_seconds`、`rare_open_requests` | 每 10 秒（随请求结束时检查）以及进程退出时 |

`counters` 里 rc3 相关的计数：`rare_requests`（有值、rc3 覆盖的请求）、`rare_no_values`（引擎没送值、rc3 没有检查的请求）、`rare_lost`（值中途断了的请求）、`rare_values`（带着值到达的 token 数）、`rare_tokens`（前端交给 rc3 的 token 数，只 shadow；sync 时严格档报警后不再计，async 时是入队数，检测进程实际处理的数在 `worker.rare_tokens`）、`rare_monitor` / `rare_strict`（两档报警数）。

**出错时放行（fail-open）**：检测器出任何错，都只会让该请求停止检测；异步进程挂掉或队列满，只影响相关请求。服务本身不会因为检测器出错而失败。

## 9. 乱码监控：rare_chars rc3

### 9.1 适用范围

- 检测对象：taxonomy v3.5.1 的 `rare_chars` core，即一眼可见的乱码：
  - 英文里混入的西里尔、泰文等其他文字碎片；
  - 中文里的拉丁字母碎片、英文里的汉字；
  - 非法 UTF-8、控制字符。
- 两档输出，一次计算同时给出：
  - **严格档**：每语料误报率上限 0.1%（`feed` / `finish` 的返回值）；
  - **监控档**：上限 0.5%，包含严格档的全部规则（`monitor(st)`）。
- **只做监控**：严格档在 Qwen3 holdout 上误报率 0.213%，超过 0.1% 的上限（§9.6），所以报警只用来记录与告警，不用来结束请求。
- 学习打分只对有参照表的词表生效（qwen3、glm）。DeepSeek 等没有参照表的词表上，学习打分不触发，rc3 退化为它的机制规则。

### 9.2 API

```python
from v35_detector import RareDetector, CONFIG_RC3

det = RareDetector('qwen3', CONFIG_RC3, think_end='auto')
st  = det.start()                                          # 每个请求一份状态
res = det.feed(st, token_ids, top1_logprob, chosen_logprob)  # 严格档报警时返回 Result，否则 None
res = det.finish(st)                                       # 输出结束：判定最后一个 token，返回严格档 Result（未报警 is_ill=False）
mon = det.monitor(st)                                      # 监控档的首次报警（未报警 is_ill=False）
```

| 参数 | 说明 |
|---|---|
| `corpus` | 内置词表名 `qwen3` / `glm` / `dsv4`，或 `rare_vocab.py` 生成的 `.npz` 路径（此时必须显式给 `think_end`） |
| `config` | 配置 dict 或 JSON 路径；`CONFIG_RC3` 为内置 rc3 |
| `think_end` | 同 m4：回答段起点的 token id；`'auto'` 只适用于内置词表；`None` 表示非思考模式 |
| `token_ids` | 生成的 token id，任意块大小 |
| `top1_logprob` | 每个位置 top-1 候选的 logprob，**自然对数**（vLLM 返回的单位） |
| `chosen_logprob` | 每个位置生成 token 自己的 logprob，自然对数。vLLM `logprobs=1` 时两者都会返回；在线时由打过补丁的采样器算出（§9.7） |

`Result` 与 m4 相同：`is_ill`、`fired_at`、`reason`（触发的规则，如 `bad_utf8`）。

### 9.3 输入与输出约定

- **只喂生成的 token**，不喂 prompt；三个数组等长。
- **块大小任意**：流式结果与整条记录的参照实现（`rare.decide`）一致，有测试保证。
- **晚一个 token**：rc3 的学习打分要看下一个 token，所以每次报警都在触发它的 token 的下一个 token 到达时给出（`fired_at` = 触发位置 + 2）；输出的最后一个 token 在 `finish` 时判定。
- **报警之后**：严格档报警后状态即结束（`st.done`），再调用 `feed` 返回同一个 `Result`；监控档只记录首次报警。
- 回答段规则与 m4 相同：第一个 `think_end` 之后算回答段，部分规则只在回答段生效。

### 9.4 配置（`configs/rc3.json`）

- `rules`：规则列表，每条是原子条件的合取（`{"and": [{"f": 特征, "lt" / "gt" / "in": 值}, ...]}`）；没有 `"tier"` 的规则属于严格档，`"tier": "monitor"` 的只进监控档。
- `gb`：学习打分模型的路径（`tables/rc3_gb.npz`）。
- rc3 严格档 4 条：非法 UTF-8；回答段占比 < 1% 的文字紧贴拉丁字母；生成概率 < 1e-4 且占比 < 0.2% 的文字紧贴拉丁字母；学习打分 > 3 且模型自己选的字母 token。监控档再加 2 条。
- 配置、词表（`tables/*_rare.npz`）、参照表（`tables/{qwen3,glm}_rcref.npz`）、模型（`tables/rc3_gb.npz`）的 sha256 记在 MANIFEST 里。改阈值就是另一个检测器，需要重新评测。
- 新模型：`python -m v35_detector.rare_vocab --tokenizer <tokenizer.json> --out <name>_rare.npz`。参照表要用这个模型带标签的正常输出统计（研究仓库 `detector/rare_ref.py`），本仓库只加载；没有参照表时学习打分不触发。

### 9.5 资源

| 项 | 数值 | 说明 |
|---|---|---|
| 每 token CPU | 0.76 μs（Mac 离线）；H20 线上 3.7–4.3 μs | Mac：单线程，Apple M3 Pro，负载 6.4，真实 dev 记录（研究仓库 `tools/bench.py`）。H20：交接包 r2 在真实生成流上实测（Xeon 8558P） |
| 钩子开销 | 每 token 约 0.5 μs（Mac）、0.69 μs（H20） | 在线时钩子自己缓冲 logprob、按 64 个一块交给 rc3 的开销，不含 rc3 本身（`bench/cpu_bench.py` 的 `rc3_buffering_only`） |
| 每请求状态 | 约 1.3 KB | 有上界，不随输出长度增长（`tests/test_rare.py`）；在线时另缓冲不满 64 个的 token id 与两个 logprob |
| 静态内存 | 每个进程常驻约 64 MB（qwen3 63.8 MB、glm 65.1 MB），加载时峰值约 118 MB | tracemalloc 实测（2026-10-22），参照表 51 MB 加乱码词表与学习打分。研究仓库与本文档此前写的"约 33 MB"偏低。磁盘上参照表 1.3–1.4 MB、模型 0.56 MB。sync 时在 API server 进程里，async 时在检测进程里。H20 实测 PSS：sync 时 API server 多 71–82 MB，async 时检测进程 102–105 MB |

### 9.6 评测结果（严格档，研究仓库 `detector_v35` 的评测）

| 集合 | core 召回 | 误报 / 负例（FPR） |
|---|---|---|
| dev Qwen3（样本内 / 按 prompt 分组 5 折的折外） | 0.383 / 0.299 | 6 / 15,277（0.039%）/ 折外 7 条 |
| dev GLM（样本内 / 折外） | 0.273 / 0.303 | 0 / 1,948 / 折外 1 条 |
| Qwen3 holdout（一次性评测） | 0.433 | 23 / 10,822（0.213%，超过 0.1% 上限） |
| DeepSeek external（1 条 core） | 0 / 1 | 0 / 2,031 |

- 前一版 rc2 在 Qwen3 holdout 上是 0.294、21 条误报。
- 监控档（上限 0.5%）：Qwen3 holdout 0.551、47 条误报（0.434%）。
- 提升主要是针对注入故障学到的：学习打分没见过某个模型的故障时，在该模型上不比 rc2 好。详见研究仓库 `report/REPORT.md` §9。

### 9.7 在线接入（vLLM 钩子，只监控）

rc3 与 m4 用同一组钩子（§8），由 `V35_RARE` 单独开关：

```bash
V35_RARE=shadow V35_DETECTOR_TABLE=qwen3 V35_DETECTOR_EVENTS=/var/log/v35-events.jsonl vllm serve ...   # 只开 rc3
V35_DETECTOR=intercept V35_RARE=shadow V35_DETECTOR_TABLE=qwen3 ... vllm serve ...                     # m4 拦截 + rc3 监控
V35_RARE=collect V35_DETECTOR_EVENTS=... vllm serve ...                                               # 只采集、不跑 rc3（测成本用）
```

- **只监控**：`V35_RARE` 没有拦截模式，rc3 的报警只写事件日志（`rare_alert`），`feed` 永远不会因为 rc3 返回停止原因。
- **输入从哪来（0.3.0，交接包 r3）**：打过补丁的 vLLM 采样器（`integrations/vllm_0_27/patch_vllm.py`，适配流程文档 §4、§10.2）
  在 `V35_RARE` 为 `collect` / `shadow` 时，对批里每一行在采样后算出原始分布的两个 logprob（`v35_detector/serving/collect.py` 的两个 Triton 核：
  `top1 = M − lse`、`chosen = x_c − lse`，float32），经 `ModelRunnerOutput.v35_rare` → `EngineCoreOutput.v35_rare` 送到 feed 钩子：
  - `EngineCoreOutput.v35_rare` 是 `[top1_0, chosen_0, top1_1, chosen_1, …]`（Python float，每 token 两个），没有值时为 None；
  - **与请求要不要 logprobs 无关**：每个请求都有值；请求要 logprobs 时，客户端拿到的仍是 vLLM 原本的结果，值不经过 logprobs 通道；
  - EngineCore 进程从 API server 继承 `V35_RARE`；`off` 时采样器不做任何额外的 GPU 工作。
  - 0.2.0（r2）读的是请求自己的 logprobs（`EngineCoreOutput.new_logprobs`）。r3 不再保留它作为后备（理由见适配流程文档 §10.2 第 6 条）。
- **哪些请求会被检查**：请求的第一批 token 决定。
  - 第一批带着值：rc3 检查整条输出（`collect` 只记覆盖）；
  - 第一批没有值：rc3 不检查该请求（`no_values`），m4 照常。在 0.27.0 里这发生在：补丁没打全、EngineCore 进程的 `V35_RARE` 是 off、
    模型走了 V2 model runner（0.27.0 稠密模型的默认，采样器不同，没有补丁）；
  - 中途某一步没有值，或值的个数不是 token 数的两倍：rc3 对该请求停止（`lost`）。投机解码有草稿的步走 rejection sampler，r3 不采集，
    所以 MTP 下的请求会在第一个草稿步记 `lost`；采样器那一步采集出错（server.log 有 `v35 rc3 collection failed`）也会这样。
- **logprob 的口径**：原始 logits 的 log-softmax，即 vLLM 默认 `--logprobs-mode raw_logprobs` 的口径（allowed-token 掩码、bad words、
  logit bias、惩罚、温度、top-k/top-p 之前；结构化输出的 bitmask 在采样器之前，包含在内）。rc3 就是按这个口径评测的；
  研究侧 Qwen3 数据是 greedy、GLM / DeepSeek 是 T=1 采样，与原始分布一致。r3 的值与 `--logprobs-mode` 的设置无关。
- **分块**：token 与两个值一起按 64 个一块交给 rc3（async 时一起入队），报警与离线 `rare.decide` 一致，与引擎每步出几个 token 无关
  （`tests/test_serving.py`）。
- **`finish` 事件的 `rare` 字段**：

| 值 | 含义 |
|---|---|
| `checked` | 每个 token 都带着值；`shadow` 时 rc3 检查了全部 token（或检查到严格档报警为止） |
| `no_values` | 引擎没有送值（见上），rc3 没有检查 |
| `lost` | 值中途断了（投机解码、采集出错，或值的个数不对）；rc3 从那里起停止 |
| `error` | rc3 出错（另有 `error` 事件）；从那里起停止，m4 不受影响 |
| `none` | 没有 token，或检测不可用（async 检测进程不在） |

- **`rare_alert`**：`tier` 为 `monitor`（监控档首次报警，含全部规则）或 `strict`（严格档首次报警，之后 rc3 对该请求结束）。
  `fired_at` 是报警时已消费的 token 数（rc3 晚一个 token，见 §9.3）；在线时报警在该 token 所在的 64 块交给 rc3 时写出，
  请求结束后才判出的写 `after_end: true`。
- **出错时放行**：rc3 出错只让 rc3 对该请求停止，m4 与服务照常；采集出错只让那一步没有值，采样照常。
- **`V35_RARE=off` 时**：采样器只多一次 `is not None` 判断，model runner 与调度器各多读一个属性（不存在），`EngineCoreOutput.v35_rare`
  为 None、编码里省略；钩子多读一个属性、多做一次布尔判断，m4 的代码路径与 H20 r1 实测的相同（`bench/cpu_bench.py` 的 `m4` 与
  `m4_values_ignored` 两项相同）。
- **开销**：
  - r2（适配流程文档 §10.3）：rc3 本身的 TPOT 增量测不出，检测器每 token 3.7–4.3 μs；主要成本在取数：请求带 `logprobs=1`
    走 vLLM 自带的 logprobs 路径，TPOT 增加 1.3–3.1%。
  - r3（适配流程文档 §10.4）：采样器采集的成本是 collect_only − baseline，待 H20 实测；9 月 v3.1 的采样器 Triton top-1 采集
    实测 C=1 +26、C=8 +15 μs/token（含当时的检测器）。钩子收值、不跑 rc3 的前端开销在 `bench/cpu_bench.py` 的 `rc3_collect` 一项。
- **已知限制**：只接 V1 GPU model runner；投机解码不采集（`lost`）；张量并行、流水线并行没有在线验证。
