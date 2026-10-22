# v35_m4_detector — LLM 推理循环输出（m4）与乱码（rc3）的流式检测器

在生成过程中检测模型"陷入循环"（taxonomy v3.5 的 `repetition` core），每 64 个 token 判一次，命中后可以像 stop string 一样立刻结束请求。
0.2.0 起另含乱码检测器 rc3（taxonomy v3.5.1 的 `rare_chars`），只做监控：有离线接口，也已接入同一组 vLLM 钩子（`V35_RARE=shadow`），
报警只记日志，从不结束请求（见下文）。

- m4 **只用生成的 token id**，不需要 logprob、不需要 top-k。
- 每 token 约 3.4–3.6 μs（Mac 单线程）、6.7–14 μs（H20 服务器，含线上实测），每请求状态约 166 KB，有上界，不随输出长度增长。
- H20 单卡实测：开启检测后 TPOT 变化在 ±22 μs/token（≤ 0.14%）以内，小于测量分辨率；拦截冒烟通过（`docs/INTEGRATION.md` §7）。
- 召回与误报（研究仓库 `detector_v35` 的评测，每语料 FPR 上限 0.1%）：Qwen3 holdout core 召回 0.444、FPR 0.062%；vLLM 原生检查调参版 0.317。
- 本仓库的判定逻辑、配置与词表是研究仓库被评测版本的逐字节拷贝（`src/v35_detector/MANIFEST.json`），并在全部 19,321 条 dev 记录上逐条复现了它的报警（`tools/verify_against_research.py`）。
- **乱码监控 rc3**（`RareDetector`、`CONFIG_RC3`）：输入是 token id、top-1 logprob 与生成 token 的 logprob（vLLM `logprobs=1`）。
  - 严格档（误报率上限 0.1%）Qwen3 holdout core 召回 0.433，前一版 rc2 为 0.294；误报率 0.213%，**超过 0.1% 的上限**，所以只做监控、不拦截。
  - 离线每 token 约 0.76 μs（Mac），每请求状态约 1.3 KB，每进程常驻约 64 MB（参照表等）；与研究仓库逐条复现了 dev 上 663 次严格档、1,161 次监控档报警。
  - 在线：`V35_RARE=shadow`，只检查带 `logprobs=1`（或更多）的请求，其余请求 rc3 跳过、m4 照常。
    打开 logprobs 本身与 rc3 在 H20 上的开销还没有测，测试方案与交接包 r2 见 `docs/INTEGRATION.md` §10。

## 安装

```bash
pip install -e .            # 依赖只有 numpy；建新模型词表另需 tokenizers：pip install -e '.[vocab]'
```

在 macOS 的 exFAT 盘（如 T7）上原地构建 wheel 会被系统生成的 `._*` 文件打断。复制到服务器或本机磁盘时排除它们：
`rsync -a --exclude '._*' <仓库>/ <目标>/`。

## 用法

```python
from v35_detector import RepDetector, CONFIG_M4

det = RepDetector('qwen3', CONFIG_M4)        # 词表：qwen3 | glm | dsv4，或 vocab.py 生成的 .npz
st = det.start()                             # 每个请求一份状态
for ids in stream_of_new_token_ids:          # 任意块大小
    res = det.feed(st, ids)
    if res is not None:                      # 报警：res.fired_at（已消费 token 数）、res.reason（触发的组件）
        break
res = res or det.finish(st)                  # 输出结束时冲刷最后不满 64 的一块
```

乱码监控（离线）：

```python
from v35_detector import RareDetector, CONFIG_RC3

rare = RareDetector('qwen3', CONFIG_RC3)
st = rare.start()
res = rare.feed(st, ids, top1_logprobs, chosen_logprobs)   # 自然对数 logprob，任意块大小；严格档报警时返回 Result
res = res or rare.finish(st)
mon = rare.monitor(st)                                      # 监控档（误报上限 0.5%）的首次报警
```

接入 vLLM 0.27：`integrations/vllm_0_27/patch_output_processor.py` 在 OutputProcessor 里加 4 个钩子（共 17 行），
feed 钩子同时传入该步的 `new_logprobs`。m4 由 `V35_DETECTOR=off|shadow|intercept` 控制，rc3 由 `V35_RARE=off|shadow` 控制
（只监控，没有拦截模式），见 `docs/INTEGRATION.md` §4、§10。

## 文档

- `docs/INTERFACE.md`：接口说明（API、输入输出约定、配置格式、资源、服务钩子与事件日志）。
- `docs/INTEGRATION.md`：适配流程（vLLM 接入、两种运行方式、新模型上线清单、监控、性能测试步骤）。

## 目录

```text
src/v35_detector/       core.py（判据定义）  stream.py（流式检测器）  offline.py（整条记录参照实现）
                        trees.py  vocab.py（词表）  configs/m4.json  tables/*.npz  MANIFEST.json
                        rare.py（乱码检测器 rc3）  rare_gb.py（学习打分）  rare_ref.py（参照表）  rare_vocab.py（乱码词表）
                        configs/rc3.json  tables/*_rare.npz  tables/{qwen3,glm}_rcref.npz  tables/rc3_gb.npz
                        serving/hooks.py（vLLM 钩子）  serving/worker.py（异步检测进程）
integrations/vllm_0_27/ patch_output_processor.py（锚点补丁：--check / --apply / --revert）
bench/                  run_h20.sh（H20 验证与性能测试一键脚本）  cpu_bench.py（每 token 开销）
                        online_tpot.py（在线 TPOT / CPU / 内存）  smoke_intercept.py（拦截冒烟）
                        fake_vllm_server.py（无 GPU 演练）  server_tpot_metrics.py  summarize_tpot.py（结果摘要）
tests/                  合成数据测试：流式 = 参照、vLLM 规则等价、状态有界、钩子（含 rc3）、补丁、MANIFEST、rc3（test_rare.py）
tools/                  verify_against_research.py（与研究仓库逐条对比）  make_manifest.py
```

## 检查

```bash
python -m pytest -q                                   # 约 30 秒，不含真实数据
python tools/verify_against_research.py --research /Volumes/T7/work/inference_severe_error_monitor/detector_v35
                                                      # 需要研究仓库与语料；约 80 秒；m4 在 dev 19,321 条上逐条一致
python tools/verify_against_research.py --research /Volumes/T7/work/inference_severe_error_monitor/detector_v35 --detector rc3
                                                      # rc3：严格档 663 次、监控档 1,161 次报警逐条一致，约 40 秒
```
