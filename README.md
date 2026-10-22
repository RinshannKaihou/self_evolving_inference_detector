# v35_m4_detector：LLM 推理异常的流式检测器（循环输出 m4、乱码监控 rc3 / rc4）

在 vLLM 生成过程中逐块检查每个请求的输出，发现两类严重异常：

- **循环输出**（taxonomy v3.5 的 `repetition` core）：检测器 **m4**，命中后可以像 stop string 一样立刻结束请求；
- **乱码**（`rare_chars` core：一眼可见的生僻字乱码，如英文里的西里尔、泰文碎片、非法 UTF-8；纯 ASCII 的碎片不算 core）：检测器 **rc3**，只监控，报警写进日志，从不结束请求。

判定逻辑、配置与表格是研究仓 `detector_v35` 被评测版本的逐字节拷贝（`src/v35_detector/MANIFEST.json`）。
`tools/verify_against_research.py` 在全部 19,321 条选型集记录上逐条复现了研究仓的报警。

## 两个检测器

| | m4（循环输出） | rc3（乱码监控） |
|---|---|---|
| 输入 | 生成的 token id | token id，加每个 token 的 top-1 logprob 与生成 token 的 logprob（模型原始分布） |
| 判定 | 每 64 个 token 一个检查点，14 个条件，任一条件连续成立即报警 | 逐 token 的规则：严格档 4 条（3 条机制规则、1 条学习打分），监控档再加 2 条；报警比触发的 token 晚 1 个 token |
| 线上开关 | `V35_DETECTOR=off \| shadow \| intercept`（shadow 只记录，intercept 结束请求） | `V35_RARE=off \| collect \| shadow`（collect 只采集输入，shadow 只记录），没有拦截模式 |
| 检测效果（研究仓评测，Qwen3 留出集） | 检出率 0.444，误报率 0.062%（上限 0.1%）；vLLM 自带检查调参后 0.317 | 严格档检出率 0.794，误报率 0.213%，**超过 0.1% 的上限**，所以只监控；前一版 rc2 为 0.670（2026-10-22 起的乱码口径） |
| 每 token CPU | Mac 3.4–3.6 μs；H20 服务器 6.7–14 μs | Mac 0.76 μs；H20 线上 3.7–4.3 μs |
| 内存 | 每请求最多约 166 KB，不随输出长度增长；词表每进程 2.3 MB | 每请求约 1.3 KB；参照表等每进程约 64 MB |

## 当前状态

- **m4**：已在单卡 H20 上验证（标签 `h20-r1`）。开启检测后每 token 时延变化在 ±22 μs（≤ 0.14%）以内，小于测量分辨率；真实拦截的冒烟测试通过。
- **rc3**：判定代码已在选型集上逐条复现。线上输入的取法换过一次：
  - r2：请求带 `logprobs=1`，从 vLLM 自带的 logprobs 路径取这两个值。H20 实测这条路径本身使每 token 时延增加 1.3–3.1%（+136–364 μs）；
    rc3 自身的时延增量测不出，它的 CPU 是每 token 3.7–4.3 μs。
  - r3（当前代码）：改在 vLLM 的采样器里用两个 Triton 核直接算出这两个值，请求不用再带 logprobs，客户端看到的 logprobs 不变。
    H20 实测（交接包 r3）：采集开销 +3.5–29 μs/token（时延的 0.07–0.18%），同一轮里 r2 的取法要 +134–358 μs；全部打开（m4 + rc3）
    时延增加 0.12–0.24%；采样器的值与 vLLM 返回的 logprobs 相差不到 1e-6。
- **rc4**（0.4.0，研究仓提交 5118334，只在选型集上选型、没有跑留出集）：输入与采集方式同 rc3，用 `V35_RARE_CONFIG=rc4` 切换。
  学习打分只在少数文字的字母与非法 UTF-8 上算（约 0.2% 的 token，rc3 约 1.3%），不用参照表。选型集折外严格档检出 Qwen3 0.823（rc3 的
  方法在同一口径下 0.584），误报 7 条（0.046%）。H20 实测（交接包 r4，与 rc3 同轮）：时延增量测不出；检测器每 token 2.0–2.4 μs（sync），
  同一轮的 rc3 为 4.7 μs；async 检测进程 43 MB（rc3 约 100 MB）；静态内存 21.7 MB（rc3 63.9 MB）。
  与选型集逐条一致：严格档 947 次、监控档 1,179 次报警，0 处差异（`tools/verify_against_research.py --detector rc4`）。

## 版本与线上测试

每个标签对应一次交给 H20 线上测试的交接包（代码快照、交接文档、一键脚本）。各版本里 m4 与 rc3 的判定代码相同，r1–r3 变的只是接入 vLLM 的方式；r4 新增 rc4。

| 标签 | 版本 | 包含 | vLLM 补丁 | H20 测试 |
|---|---|---|---|---|
| `h20-r1` | 0.1.0 | m4 | 1 个文件、17 行 | 已测，原始结果在研究仓 `report/data/h20_20261021/` |
| `h20-r2` | 0.2.0 | m4，rc3（输入取自请求的 logprobs） | 1 个文件、17 行 | 已测，原始结果在研究仓 `report/data/h20_20261022/` |
| `h20-r3` | 0.3.0 | m4，rc3（输入在采样器里采集） | 5 个文件、35 行 | 已测，原始结果在研究仓 `report/data/h20_20261022_r3/` |
| `h20-r4` | 0.4.0 | m4，rc3 与 rc4（`V35_RARE_CONFIG` 切换） | 同 r3 | 已测，原始结果在研究仓 `report/data/h20_20261022_r4/` |

## 安装

```bash
pip install -e .            # 依赖只有 numpy；为新模型建词表另需 tokenizers：pip install -e '.[vocab]'
```

在 macOS 的 exFAT 盘（如 T7）上原地构建 wheel 会被系统生成的 `._*` 文件打断。复制到服务器或本机磁盘时排除它们：
`rsync -a --exclude '._*' <仓库>/ <目标>/`。

## 离线用法

循环输出（m4）：

```python
from v35_detector import RepDetector, CONFIG_M4

det = RepDetector('qwen3', CONFIG_M4)        # 词表：qwen3 | glm | dsv4，或 vocab.py 生成的 .npz
st = det.start()                             # 每个请求一份状态
for ids in stream_of_new_token_ids:          # 任意块大小
    res = det.feed(st, ids)
    if res is not None:                      # 报警：res.fired_at（已消费 token 数）、res.reason（触发的条件）
        break
res = res or det.finish(st)                  # 输出结束时处理最后不满 64 个 token 的一块
```

乱码监控（rc3）：

```python
from v35_detector import RareDetector, CONFIG_RC3

rare = RareDetector('qwen3', CONFIG_RC3)
st = rare.start()
res = rare.feed(st, ids, top1_logprobs, chosen_logprobs)   # 自然对数 logprob，任意块大小；严格档报警时返回 Result
res = res or rare.finish(st)
mon = rare.monitor(st)                                      # 监控档（误报上限 0.5%）的首次报警
```

## 接入 vLLM 0.27.0

```bash
pip install -e .
python integrations/vllm_0_27/patch_vllm.py --check     # 15 个锚点各出现一次、补丁后能编译；不写文件
python integrations/vllm_0_27/patch_vllm.py --apply     # 改 5 个文件、共 35 行，每行带 "# v35-detector hook"，原件另存为 <文件>.v35-orig
V35_DETECTOR=shadow V35_RARE=shadow V35_DETECTOR_TABLE=qwen3 \
  V35_DETECTOR_EVENTS=/var/log/v35-events.jsonl vllm serve /path/to/Qwen3-30B-A3B ...
```

- **开关**：
  - `V35_DETECTOR`（m4）与 `V35_RARE`（rc3）见上表；
  - `V35_DETECTOR_RUNNER=sync`（默认，在前端输出处理循环里算）或 `async`（独立检测进程）；
  - `V35_DETECTOR_TABLE` 是词表名，开启任一检测器时必填；`V35_DETECTOR_EVENTS` 是事件日志路径。
- **撤销**：`--revert`，5 个文件一起撤。不开检测、或不安装本包时，打过补丁的 vLLM 与原版行为相同。
- **上线顺序**：先用 `shadow` 观察报警率和触发条件的分布，再打开 m4 的 `intercept`（适配流程文档 §5）。rc3 一直只用 `shadow`。

## 文档

- `docs/INTERFACE.md`：接口说明。
  - §1–§8：m4 的 API、输入输出约定、配置、资源、服务钩子与事件日志；
  - §9：rc3 的 API、配置、资源、评测结果，§9.7 是在线接入。
- `docs/INTEGRATION.md`：适配流程。
  - 接入位置、两种运行方式、安装（§4，补丁改的 5 个文件）、新模型上线清单、监控；
  - H20 验证与性能测试（§7）、故障排查、已知限制；
  - §10：rc3 的在线接入，以及 r2、r3 两轮测试的对照组与结果。

## 目录

```text
src/v35_detector/       core.py（判据定义）  stream.py（流式检测器）  offline.py（整条记录参照实现）
                        trees.py  vocab.py（词表）  configs/m4.json  tables/*.npz  MANIFEST.json
                        rare.py（乱码检测器 rc3）  rare_gb.py（学习打分）  rare_ref.py（参照表）  rare_vocab.py（乱码词表）
                        configs/rc3.json  tables/*_rare.npz  tables/{qwen3,glm}_rcref.npz  tables/rc3_gb.npz
                        serving/hooks.py（vLLM 钩子）  serving/worker.py（异步检测进程）  serving/collect.py（采样器里的 rc3 采集）
integrations/vllm_0_27/ patch_vllm.py（5 个文件的锚点补丁：--check / --apply / --revert）
bench/                  run_h20.sh（H20 验证与性能测试一键脚本）  cpu_bench.py（每 token 开销）
                        online_tpot.py（在线 TPOT / CPU / 内存）  smoke_intercept.py（拦截与 rc3 冒烟）
                        collect_gpu_check.py（采样器 Triton 核的 GPU 核对）  fake_vllm_server.py（无 GPU 演练）
                        server_tpot_metrics.py  summarize_tpot.py（结果摘要）
tests/                  合成数据测试：流式 = 参照、vLLM 规则等价、状态有界、钩子（含 rc3）、补丁与采集链路（test_vllm_patch.py）、
                        采集公式（test_collect.py）、MANIFEST、rc3（test_rare.py）
tools/                  verify_against_research.py（与研究仓逐条对比）  make_manifest.py
```

## 检查

```bash
python -m pytest -q        # 约 30 秒，合成数据，不含真实轨迹；需要 torch 或真实 vLLM 文件的测试在缺少时跳过
python tools/verify_against_research.py --research <研究仓 detector_v35 的路径>                 # m4：19,321 条逐条一致，约 80 秒
python tools/verify_against_research.py --research <研究仓 detector_v35 的路径> --detector rc3   # rc3：严格档 663 次、监控档 1,161 次报警逐条一致，约 40 秒
```

后两条需要研究仓及其语料，只读选型集。
