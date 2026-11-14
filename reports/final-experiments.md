# ClarifySafe 正式实验结果

主测试集有 48 条任务、12 个场景组。每个系统在每个条件下对每条任务保留一条计分轨迹，均使用 Kimi K3.1 Pro。每组四个变体全部通过 I-TGC 时，该组才计为 I-SGC 通过。I-SGC 是主指标。

## 20 次调用上限的主结果

| 系统 | TGC | SGC | I-TGC | I-SGC |
|---|---:|---:|---:|---:|
| ClarifySafe | 44/48 | 9/12 | 43/48 | 9/12 |
| AppWorld ReAct (adapted) | 28/48 | 2/12 | 28/48 | 2/12 |
| AppWorld Function Calling (adapted) | 34/48 | 2/12 | 32/48 | 2/12 |
| ClarifySafe without the write verifier | 42/48 | 6/12 | 41/48 | 6/12 |

ReAct 和 Function Calling 由 AppWorld 的公开实现改编。它们共用评测任务和评分规则，各系统保留各自的调用方式和提示配置。

## 模型调用预算

| 每题最多调用 | ClarifySafe I-SGC | ReAct I-SGC | ClarifySafe 因预算停止 | ReAct 因预算停止 |
|---:|---:|---:|---:|---:|
| 10 | 6/12 | 0/12 | 1 | 32 |
| 20 | 9/12 | 2/12 | 0 | 15 |

## 风险消融和低风险对照

| 风险消融系统 | 获批条件 I-TGC | 拒批条件 I-TGC | 至少有一次未授权写入的任务 |
|---|---:|---:|---:|
| ClarifySafe | 8/9 | 8/9 | 0/18 |
| ClarifySafe without the risk gate | 0/9 | 0/9 | 18/18 |

风险消融复用主测试集中的 9 个基础场景。获批和拒批条件各 9 条。用户指令没有主动要求确认。

| 低风险对照系统 | TGC | 不必要确认 |
|---|---:|---:|
| ClarifySafe | 6/9 | 0/9 |
| AppWorld ReAct (adapted) | 9/9 | 0/9 |

低风险对照使用 9 条额外的 Spotify 只读任务。每条任务的信息已经完整。ClarifySafe 的三条失败都出现在按发行年份统计歌曲的任务中，停止原因为单轮读取数量超限。该子集上，TGC 差距为 33.3 个百分点，超出 proposal 的 5 个百分点阈值。

## 写后验证器的故障注入

API 返回成功但数据库没写入时，完整系统 3/3 正确停止，关闭验证器后 3/3 误报完成。数据库已写入但响应超时时，两组均 3/3 停止，同一任务内均未发出第二次写入。

## 实验记录

共补齐 56 条曾因 Kimi 服务限流而提前停止的实验记录，其中 37 条在额度恢复后补齐。每次补跑从同一初始状态启动，采用第一次没有因限流终止的结果。原运行与补跑日志均保留。20 次上限的主测试中，ClarifySafe 有一条因服务超时停止，依照既定口径保留为失败。

## 图件与数据

- [主结果表](../figures/01-main-results.pdf)
- [任务类型比较](../figures/02a-task-families.pdf)
- [调用预算](../figures/02b-call-budget.pdf)
- [风险门控消融](../figures/02c-risk-gate.pdf)
- [低风险只读对照](../figures/02d-low-risk-controls.pdf)
- [Agent 架构](../figures/03-agent-architecture.pdf)和[评测流程](../figures/04-evaluation-pipeline.pdf)
- [结构化实验汇总](final-experiments.json)和[写后验证故障注入汇总](write-verifier-faults.json)
- [正式数据和复现说明](../docs/experiments.md)
