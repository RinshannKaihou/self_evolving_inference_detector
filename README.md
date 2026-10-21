# v35_m4_detector — LLM 推理循环输出的流式检测器（m4）

在生成过程中检测模型"陷入循环"（taxonomy v3.5 的 `repetition` core），每 64 个 token 判一次，命中后可以像 stop string 一样立刻结束请求。

- **只用生成的 token id**，不需要 logprob、不需要 top-k。
- 每 token 约 3.4–3.6 μs（单线程 CPU），每请求状态约 166 KB，有上界，不随输出长度增长。
- 召回与误报（研究仓库 `detector_v35` 的评测，每语料 FPR 上限 0.1%）：Qwen3 holdout core 召回 0.444、FPR 0.062%；vLLM 原生检查调参版 0.317。
- 本仓库的判定逻辑、配置与词表是研究仓库被评测版本的逐字节拷贝（`src/v35_detector/MANIFEST.json`），并在全部 19,321 条 dev 记录上逐条复现了它的报警（`tools/verify_against_research.py`）。
- `rare_chars`（乱码 / 异常字符）检测器仍在迭代，暂未包含。

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

接入 vLLM 0.27：`integrations/vllm_0_27/patch_output_processor.py` 在 OutputProcessor 里加 4 个钩子（共 17 行），
由环境变量 `V35_DETECTOR=off|shadow|intercept` 控制，见 `docs/INTEGRATION.md`。

## 文档

- `docs/INTERFACE.md`：接口说明（API、输入输出约定、配置格式、资源、服务钩子与事件日志）。
- `docs/INTEGRATION.md`：适配流程（vLLM 接入、两种运行方式、新模型上线清单、监控、性能测试步骤）。

## 目录

```text
src/v35_detector/       core.py（判据定义）  stream.py（流式检测器）  offline.py（整条记录参照实现）
                        trees.py  vocab.py（词表）  configs/m4.json  tables/*.npz  MANIFEST.json
                        serving/hooks.py（vLLM 钩子）  serving/worker.py（异步检测进程）
integrations/vllm_0_27/ patch_output_processor.py（锚点补丁：--check / --apply / --revert）
bench/                  run_h20.sh（H20 验证与性能测试一键脚本）  cpu_bench.py（每 token 开销）
                        online_tpot.py（在线 TPOT / CPU / 内存）  smoke_intercept.py（拦截冒烟）
                        fake_vllm_server.py（无 GPU 演练）  server_tpot_metrics.py
tests/                  合成数据测试：流式 = 参照、vLLM 规则等价、状态有界、钩子、补丁、MANIFEST
tools/                  verify_against_research.py（与研究仓库逐条对比）  make_manifest.py
```

## 检查

```bash
python -m pytest -q                                   # 约 15 秒，不含真实数据
python tools/verify_against_research.py --research /Volumes/T7/work/inference_severe_error_monitor/detector_v35
                                                      # 需要研究仓库与语料；约 80 秒；dev 19,321 条逐条一致
```
