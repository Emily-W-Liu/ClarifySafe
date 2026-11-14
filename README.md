# ClarifySafe

ClarifySafe 是一个处理模糊指令的工具调用 Agent。它先检索工具定义和应用记录，必要时询问用户，随后由运行时检查风险、授权和写入前状态。写入后重新读取状态验证结果。

## 本地演示与前端联调

需要 Python 3.11–3.13 和 [uv](https://docs.astral.sh/uv/)。

```sh
uv sync --locked --no-dev
uv run --no-dev clarifysafe --data-dir .clarifysafe/demo demo
uv run --no-dev clarifysafe --data-dir .clarifysafe/web serve --port 8000 --allow-origin http://localhost:5173
```

启动后打开 `http://127.0.0.1:8000/docs`。脚本模式无需密钥，示例指令为“把会议纪要发给 Alex”。HTTP 接口目前连接模拟消息环境。使用模型理解该环境中的自然语言指令时，复制 `.env.example` 为 `.env`，填写自己的模型和密钥，再用 `uv run --no-dev --env-file .env clarifysafe --mode llm serve` 启动。详见[接口说明](docs/ui-api.md)。

## AppWorld 通用终端入口

终端入口使用独立的 Python 3.11 环境，按本项目验证过的 AppWorld 提交安装。先安装 Git LFS。以下命令在项目根目录执行。

```sh
git lfs install
git clone https://github.com/StonyBrookNLP/appworld.git .appworld/upstream
git -C .appworld/upstream checkout 42b5bcf3cd334fee33f0c37c02070a9f5807add5
git -C .appworld/upstream lfs pull
uv venv --python 3.11 .venv-appworld
uv pip install --python .venv-appworld/bin/python -r config/appworld-requirements.txt -e .appworld/upstream
(
  cd .appworld/upstream
  ../../.venv-appworld/bin/python -m appworld.cli install --repo
)
.venv-appworld/bin/python -m appworld.cli download data --version 0.2.0 --root "$PWD/.appworld/world"
```

复制 `.env.example` 为 `.env`，填入模型 ID 和密钥。以下命令会接受自由文本指令，显示澄清问题和待批准的操作。

```sh
PYTHONPATH=src .venv-appworld/bin/python scripts/demo_tool_agent.py
```

可以用 `--world-root` 指定已有 AppWorld 安装，用 `--task-id` 选择初始数据。审计记录写入本地 `.appworld/results/`。AppWorld API 操作发生在本地模拟环境。

## 代码位置

| 模块 | 路径 |
|---|---|
| AppWorld Agent 循环和工具适配 | `src/clarifysafe/agent/tool_agent.py`、`src/clarifysafe/environment/appworld_tools.py` |
| 运行时的风险校验、逐步写入和台账 | `src/clarifysafe/agent/policy.py`、`batch.py`、`executor.py` |
| 模拟消息 Agent 的命令行和 HTTP 入口 | `src/clarifysafe/cli.py`、`src/clarifysafe/api.py` |
| 前端接口契约 | `examples/handoff/openapi.json` |

每位组员从 `main` 创建自己的分支，完成变更后通过 Pull Request 合并。前端对接以 OpenAPI 契约为准。AppWorld 由其官方仓库提供。引用可见 [AppWorld 论文](https://aclanthology.org/2024.acl-long.850/)。
