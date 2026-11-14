# ClarifySafe HTTP 接口

该服务连接本地模拟消息环境，供前端联调。AppWorld 使用独立的终端入口。

## 启动服务

在项目根目录安装锁定依赖，然后启动固定脚本模式。该模式支持“把会议纪要发给 Alex”。

```sh
uv sync --locked --no-dev
uv run --no-dev clarifysafe --data-dir .clarifysafe/web-demo serve --port 8000 --allow-origin http://localhost:5173
```

需要自然语言理解时，加载本地模型配置启动LLM模式。模型调用会产生API费用。

```sh
uv run --no-dev --env-file .env clarifysafe --data-dir .clarifysafe/web-llm --mode llm serve --port 8000 --allow-origin http://localhost:5173
```

服务监听 `http://127.0.0.1:8000`。浏览器打开 `/docs` 可以直接试用接口，`/openapi.json`提供机器可读契约。UI和服务在同一台电脑运行。允许来源必须包含前端实际协议、域名及端口，例如localhost与127.0.0.1属于不同来源，可以重复传入`--allow-origin`。

运行模式和策略由启动配置确定。`--policy config/policy.toml`放在serve前面。改变策略后，既有授权仍受运行时的策略摘要检查。

## 已实现的接口

| 接口 | 请求 | 成功响应 |
|---|---|---|
| GET /health | 无 | 服务版本、mode、environment=mock |
| POST /runs | `{"instruction":"把会议纪要发给 Alex"}` | 201，AgentResult |
| GET /runs/{run_id} | 路径中的run_id | 200，AgentResult |
| POST /runs/{run_id}/responses | UserResponse | 200，AgentResult |
| GET /runs/{run_id}/events?after=0 | after为最后收到的sequence | 200，`{"events":[...],"next_after":N}` |

运行结果沿用已有AgentResult schema。`pending.type=ask`时显示候选，仅accepts_text为true时提供自由输入。`pending.type=confirm`时显示action和risk，提交approved布尔值。每次都使用本次响应里的intervention_id。

事件接口返回对象，events字段是事件列表。下次查询传入next_after。sequence是数据库全局序号，在单任务中可能不连续。请求after为负数会被拒绝。

## 前端调用示例

```javascript
const base = "http://127.0.0.1:8000";
const response = await fetch(`${base}/runs`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ instruction: "把会议纪要发给 Alex" }),
});
const result = await response.json();
if (!response.ok) throw new Error(JSON.stringify(result.detail));
// 保存result.run_id并按照result.pending渲染交互。
```

用户选择候选后，把下面的对象POST到该run的responses接口。

```json
{"intervention_id":"替换为当前ask ID","choice":"alex-a"}
```

用户批准时使用当前confirm ID。

```json
{"intervention_id":"替换为当前confirm ID","approved":true}
```

拒绝使用false。LLM模式的Ask支持text字段。choice、text、approved每次只提交一种。

## 请求时序和错误处理

当前POST同步执行到下一次人工介入或任务结束。模型调用期间前端应显示处理中。服务对同一数据目录加跨进程文件锁，运行期间其他状态查询、事件查询或写请求返回409 runtime_busy，可以稍后查询。健康检查可继续响应。

| HTTP状态 | 含义 | UI处理 |
|---|---|---|
| 404 | detail.code=run_not_found | 提示任务不存在 |
| 409 | detail.code=runtime_busy，Retry-After=1 | 暂停轮询，稍后查询 |
| 409 | detail.code=stale_intervention | 刷新任务，展示当前待回答问题或结果 |
| 422 | 请求结构错误，或detail.code=invalid_response | 修正字段，保留界面输入 |
| 503 | model_unavailable或storage_unavailable | 展示服务错误，检查后端配置或存储 |

自定义错误采用`{"detail":{"code":"...","message":"..."}}`。FastAPI输入校验错误沿用`detail`数组格式，前端应兼容两种结构。

任务执行中发生的模型或工具失败通常返回HTTP成功和`status=stopped`，因为任务结果已经记录。应同时检查HTTP状态及AgentResult.status。

批准按钮提交后禁用，失败时先读取任务状态。相同批准重复提交返回409，执行器保留独立的重复执行保护。创建任务接口尚无请求去重键，收到网络超时后自动重发POST /runs可能创建第二个任务。客户端关闭连接也可能让已提交操作继续运行。

事件采用增量轮询，暂未提供执行中的实时SSE。服务重启后通过原run_id恢复，使用相同数据目录和策略。当前是本机演示服务，账号登录、跨用户权限隔离和公网部署尚未实现。

