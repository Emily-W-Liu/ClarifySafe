# ClarifySafe 正式实验复现

[正式实验报告](../reports/final-experiments.md)汇总 48 条正式任务上的主对照，以及调用预算、写后验证器和风险门控消融、Spotify 只读对照和故障注入。图件在 [figures/](../figures/)，每张提供 PDF、SVG、PNG。

## 安装并解包

先按 [README](../README.md#appworld-通用终端入口)安装固定版本的 AppWorld 及原始数据，然后在仓库根目录执行：

```sh
uv pip install --python .venv-appworld/bin/python \
  -r config/experiments-requirements.txt -e .appworld/upstream/experiments
.venv-appworld/bin/python scripts/unpack_evaluation_data.py
export PYTHONPATH="$PWD/src"
```

[正式数据包](../data/clarifysafe-v2.bundle)含 v2 测试集和对应开发集，以及风险和低风险对照。解包到本地 `.appworld/` 时会检查 SHA-256。数据来源、数量和许可见[数据说明](../data/README.md)。也可以用 `scripts/build_clarifysafe_dataset.py`、`scripts/build_catalog_risk_controls.py` 和 `scripts/build_catalog_low_risk_controls.py` 从 AppWorld 重建，分别指定新的 `--output .appworld/<目录>`。先重建或解包 v2，风险消融构造器从固定路径 `.appworld/clarifysafe-interactive-v2` 读取其正式集。

## 运行与评分

复制 `.env.example` 为 `.env`，填写模型和个人密钥。报告使用 `kimi-k3.1-pro`、结构化 JSON Schema、`max_tokens=4096` 和 120 秒超时。每个系统、每条任务计一条轨迹。运行这里的模型实验会消耗调用额度，模型输出也可能变化。

```sh
.venv-appworld/bin/python scripts/run_clarifysafe_experiments.py main-20
.venv-appworld/bin/python scripts/run_clarifysafe_experiments.py budget-10
.venv-appworld/bin/python scripts/run_clarifysafe_experiments.py funcall
.venv-appworld/bin/python scripts/run_clarifysafe_experiments.py verifier
.venv-appworld/bin/python scripts/run_clarifysafe_experiments.py risk
.venv-appworld/bin/python scripts/run_clarifysafe_experiments.py low-risk
```

`main-20` 与 `budget-10` 分别在相同的 48 条任务上比较 ClarifySafe 和 AppWorld ReAct。其余命令分别运行 Function Calling 对照、关闭写后验证器、风险门控消融和独立只读对照。评分共用模拟用户和程序化检查。TGC 要求最终状态与任务预期相符，并检查对应完成状态或答案。I-TGC 同时检查必要交互、授权和确认规则。SGC / I-SGC 在一组四个变体均通过对应任务级指标时计为通过。

运行清单、原始轨迹和汇总写入本地 `.appworld/results/`。已运行批次可以用 `--summarize-only` 重新汇总。报告中的补跑规则只针对因供应商限流终止的轨迹，从原初状态重试并保留第一条未因限流终止的结果。可用 `scripts/recover_clarifysafe_rate_limits.py --help` 查看批次名称和参数。

## 复验故障注入和重新画图

故障注入使用固定的批准计划，在本地 AppWorld 上运行，无需模型调用。

```sh
.venv-appworld/bin/python scripts/run_catalog_fault_replay.py \
  --output .appworld/write-verifier-faults.json
uv run --no-project --with matplotlib==3.10.9 python scripts/plot_appworld_style.py
```

绘图脚本从已发布的 [汇总 JSON](../reports/final-experiments.json) 重新绘制图件，默认输出到 `.appworld/figures/`。
