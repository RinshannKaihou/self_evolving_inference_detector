# 接口说明：v35 重复检测器（m4）

版本 0.1.0。判定逻辑与研究仓库 `detector_v35` 被评测的版本逐字节相同（`src/v35_detector/MANIFEST.json`）。

## 1. 适用范围

- 检测对象：taxonomy v3.5 的 `repetition` core，即模型陷在循环里：
  - 逐字循环；
  - 跑到上限的重复墙；
  - 推理原地打转；
  - 空白或单字符填充。
- 已评测的模型：Qwen3-8B / 30B-A3B / 32B（思考模式）、GLM-5.3-Flash（思考模式，MTP）、DeepSeek-V4-Flash（非思考模式）。
  - 换模型时阈值不需要重调，只需建词表（见适配流程文档 §5）。
- 不检测：rare_chars（乱码 / 异常字符），该检测器仍在迭代。

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
| 每 token CPU | 3.4–3.6 μs | 单线程，Apple M3 Pro。真实 dev 数据 3.43（研究仓库）；合成数据 3.60（`bench/cpu_bench.py`）。部署机上需重测 |
| 每请求状态 | ≤ 166 KB | 有上界：窗口填满后不再增长（`tests/test_state.py`） |
| 词表 | 每个 tokenizer 约 2.3 MB 内存 | 每进程加载一次后缓存 |
| 对照：vLLM 原生检查 | 84.6 μs（[50,500]×3）、259.8 μs（[32,2000]×3） | 研究仓库在同一台机器上，用上游原代码、按逐 token 调用测得 |

## 8. 服务钩子（`v35_detector.serving.hooks`）

由打过补丁的 vLLM OutputProcessor 调用，见适配流程文档：

| 函数 | 调用时机 | 返回 |
|---|---|---|
| `register(request_id)` | 请求进入 OutputProcessor | — |
| `feed(request_id, token_ids)` | 每个引擎步的新 token（stop string 检查之后） | 拦截模式下报警时返回 `'v35_repetition: <reason>'`，否则 None |
| `finish(request_id, finish_reason)` | 请求以任何原因结束 | — |
| `abort(request_id)` | 请求被取消，或引擎出错 | — |

环境变量（服务启动、导入模块时读取；配置错误会让服务启动失败）：

| 变量 | 取值 | 默认 |
|---|---|---|
| `V35_DETECTOR` | `off` / `shadow`（只检测、记日志）/ `intercept`（检测、记日志并结束请求） | `off` |
| `V35_DETECTOR_TABLE` | `qwen3` / `glm` / `dsv4` / `.npz` 路径 | 开启时必填 |
| `V35_DETECTOR_THINK_END` | `auto` / `none` / token id | `auto` |
| `V35_DETECTOR_RUNNER` | `sync`（在输出处理循环里算）/ `async`（独立检测进程） | `sync` |
| `V35_DETECTOR_CONFIG` | 配置 JSON 路径 | 内置 m4 |
| `V35_DETECTOR_EVENTS` | 事件日志路径（JSONL） | `./v35-detector-events.jsonl` |

事件日志每行一个 JSON，公共字段为 `ts`、`pid`、`event`：

| event | 其他字段 | 何时写 |
|---|---|---|
| `alert` | `request_id`、`fired_at`、`reason`、`tokens_seen`、`action`（`shadow` / `intercept`）；或 `after_end: true` | 检测器报警。`after_end` 表示请求已经结束（最后不满一块的判定，或异步结果晚到） |
| `finish` | `request_id`、`tokens`、`finish_reason` | 每个请求结束 |
| `error` | `request_id`、`where`、`error` | 检测器出错。该请求停止检测，服务照常进行 |
| `summary` | `counters`、`detector_seconds_sync`、`worker`、`open_requests` | 每 10 秒（随请求结束时检查）以及进程退出时 |

**出错时放行（fail-open）**：检测器出任何错，都只会让该请求停止检测；异步进程挂掉或队列满，只影响相关请求。服务本身不会因为检测器出错而失败。

## 9. rare_chars

留空。乱码 / 异常字符检测器（研究仓库 rc1）的效果还在迭代，定稿后补充本节。
