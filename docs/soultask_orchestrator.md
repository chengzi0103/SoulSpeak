# SoulTask Orchestrator 框架说明

## 1. 总览

SoulTask Orchestrator（STO）是 SoulSpeak 体系内负责“把想法落地为行动”的任务枢纽。它把对话 LLM（Emilia）、任务调度、沙箱执行等模块串联在一起，让“提醒”“运行脚本”“安装软件”等需求能够被可靠执行、记录与追溯。

核心理念：

1. **前端对话轻量化** —— Emilia 专注于快速交流，不直接执行命令；
2. **调度智能模块化** —— 独立的调度 LLM（Orchestrator）负责解析意图、创建/更新任务；
3. **执行安全可控** —— Dispatcher + Executors 组合保证动作在可追踪、可审计的环境内完成；
4. **状态全程可观测** —— 任务与日志集中存储，Emilia 随时可以查询并向用户反馈。

### 1.1 组件架构

```mermaid
graph LR
    User((User)) -->|自然语言| Emilia[Emilia 前端 LLM]
    Emilia -->|request_task 工具| OrchestratorLLM[调度 LLM Orchestrator]
    OrchestratorLLM -->|写入/更新| TaskDB[(Task Store)]
    Emilia -->|query_tasks 工具| TaskDB

    TaskDB --> Dispatcher[Dispatcher 调度服务]
    Dispatcher -->|pending 任务| ExecutorPool{Executors}

    ExecutorPool --> SandboxExecutor[Sandbox Executor]
    ExecutorPool --> HostExecutor[Host Runner]
    ExecutorPool --> ReminderExecutor[Reminder/TTS]
    ExecutorPool --> AgentExecutor[Multi-step Agent]

    SandboxExecutor --> Logs[(Task Logs)]
    HostExecutor --> Logs
    ReminderExecutor --> Logs
    AgentExecutor --> Logs

    Logs --> Emilia
```

## 2. 组件职责

| 组件 | 角色 | 职责 |
| --- | --- | --- |
| Emilia | 对话前端 | 识别用户意图，调用工具 `request_task` / `list_tasks` / `task_detail`，同步记忆与反馈 |
| 调度 LLM Orchestrator | 任务规划 | 将自然语言需求转为结构化任务（`task_type`、`payload`、`manual_required` 等），写入 Task Store |
| Task Store | 真相来源 | SQLite/Postgres 等，存储任务主体、状态、时间戳、结果摘要；另含 `task_logs` 记录执行细节 |
| Dispatcher | 常驻调度器 | 轮询/订阅 Task Store，挑选 `status=pending` 的任务，锁定后分发给合适的 Executor |
| Executors | 执行器集合 | 根据 `task_type` 执行动作，采集 stdout/stderr/产物，更新任务状态与日志 |
| Task Logs | 审计与追踪 | 保存状态流转、stdout、stderr、人工确认记录，供 Emilia/用户查询 |

### 2.1 Executor 分类示例

| Executor | 适用任务 | 行为 |
| --- | --- | --- |
| ReminderExecutor | `reminder` / `notification` | 到点调用 `play_sentences` 或其他通知渠道 |
| SandboxExecutor | `run_code` / `data_analysis` | 通过 gVisor/Docker+Jupyter 执行代码，返回 stdout/图像 |
| HostExecutor | `install_package` / 系统操作 | 需要人工确认，使用白名单命令或模板化脚本执行，严格记录结果 |
| AgentExecutor | `plan_and_execute` / 复杂流程 | 触发子 LLM + Sandbox 的多步执行（例如生成计划、逐步运行、验证） |

## 3. 任务生命周期

### 3.1 状态机

```mermaid
stateDiagram-v2
    [*] --> pending_manual: 创建任务（需人工确认）
    pending_manual --> pending: 用户确认
    pending_manual --> cancelled: 用户拒绝/撤销
    pending --> running: Dispatcher 锁定并执行
    running --> success: 执行成功
    running --> failed: 执行失败（可提供错误信息）
    failed --> blocked: 需要人工干预/备注后再试
    blocked --> pending: 补救后重新执行
    cancelled --> [*]
    success --> [*]
```

### 3.2 表结构建议

`tasks` 表核心字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | TEXT/UUID | 任务 ID |
| `type` | TEXT | 任务类型（如 `reminder`、`run_code`、`install_package`） |
| `payload` | JSON | 参数（命令、时间、脚本等） |
| `status` | TEXT | `pending_manual` → `pending` → `running` → `success/failed/blocked/cancelled` |
| `manual_required` | BOOLEAN | 是否需要人工确认 |
| `scheduled_for` | DATETIME | 定时执行时间（提醒类任务用） |
| `created_at` / `updated_at` / `executed_at` | DATETIME | 生命周期时间戳 |
| `result` | JSON/TEXT | 摘要结果（stdout 关键行、文件路径等） |
| `error` | JSON/TEXT | 失败原因（包含 stderr 片段） |
| `confirmation_note` | TEXT | 用户确认/拒绝理由 |
| `attempts` | INTEGER | 执行次数（重试用） |

`task_logs` 表用于记录细粒度事件（状态变更、stdout、stderr、人工确认等），字段包含 `task_id`、`event`、`timestamp`、`data`（JSON/text）。

## 4. 请求与执行流程

### 4.1 任务创建 & 执行顺序图

```mermaid
sequenceDiagram
    participant U as User
    participant E as Emilia
    participant O as Orchestrator LLM
    participant T as Task Store
    participant D as Dispatcher
    participant X as Executor

    U->>E: “帮我安装 htop”
    E->>O: 调用 request_task 工具
    O->>T: 写入任务（type=install_package, status=pending_manual, manual_required=true）
    E->>U: 告知任务创建并等待确认
    U->>E: “确认安装”
    E->>T: 调用 confirm_task（status=pending）

    loop 调度周期
        D->>T: 查询 pending 任务
        T-->>D: 返回任务详情
        D->>T: 标记 status=running
        D->>X: 调用对应 Executor（HostExecutor）
        X->>X: 执行命令（可能在 sandbox 或宿主机）
        X->>T: 写状态 success/failed + 日志
    end

    E->>T: 查询任务状态
    T-->>E: 返回 success/failed + 日志
    E->>U: 反馈结果
```

## 5. 人工确认与安全策略

1. **高危任务默认 `manual_required=true`**：如 `install_package`、`modify_system`。
2. **确认操作**：通过工具或界面将 `status` 从 `pending_manual` 改为 `pending` 并写 `confirmation_note`；拒绝则设为 `cancelled`。
3. **执行白名单**：HostExecutor 仅接受经审核的命令模板，如 `brew install ...`、`apt-get install ...`，并记录完整 stdout/stderr。
4. **失败处理**：若执行失败，Executor 将状态设为 `failed` 并写 `error`；调度器可根据策略自动重试，或将任务转为 `blocked` 以等待人工处理。
5. **安全隔离**：沙箱执行器使用 gVisor/Docker + Jupyter Kernel；HostExecutor 在确认后执行，并可结合 sudo 权限分级或独立服务守护。

## 6. 任务查询与反馈

- Emilia 可通过工具 `list_tasks(status=...)`、`task_detail(task_id)` 获取任务列表与详细日志；
- `Task Logs` 可以提供 stdout/stderr 摘要、文件产物链接等，保证每次执行可追踪、可复盘；
- 对于长期任务或失败任务，Emilia 会主动提醒用户处理（例如 `blocked` 状态）。

## 7. 落地路线建议

1. **最小可行产品**
   - 建立 Task Store（SQLite）与 `tasks`/`task_logs` 表；
   - 实现 Dispatcher（APScheduler 轮询）+ `ReminderExecutor`、`SandboxExecutor`；
   - 接入调度 LLM 工具（FastMCP）让 Emilia 能创建/确认/查询任务。

2. **扩展阶段**
   - 增加 HostExecutor（高权限操作，需人工确认和白名单）；
   - 引入 AgentExecutor 处理复杂多步骤任务（子 LLM + sandbox）；
   - 设计简单的 Web/CLI 面板，方便人工查看和确认任务。

3. **高级优化**
   - 将 Task Store 迁移至 Postgres，利用 LISTEN/NOTIFY 降低延迟；
   - 引入队列（如 Huey/RQ）处理执行压力与重试；
   - 建立统一的审计与通知系统，对 task 结果触发 TTS/消息推送。

## 8. 命名约定与目录

- 名称：**SoulTask Orchestrator（STO）**
- 文档路径：`docs/soultask_orchestrator.md`
- 相关模块建议放置于：
  - `soul_speak/tasks/` —— 任务模型、调度器、执行器实现
  - `soul_speak/tools/` —— FastMCP 工具（与 LLM 交互）
  - `scripts/` —— Host Runner/Sandbox Manager 启动脚本

---

通过 STO 架构，SoulSpeak 可以在保持对话体验流畅的同时，实现“会说话、能办事、记得住”的全链路能力。随着任务类型增多，只需增加对应 Executor 与规则即可逐步扩展。

## 9. 落地进度计划

| 步骤 | 项目 | 状态 | 说明 |
| --- | --- | --- | --- |
| 1 | 设计 TaskStore 与任务模型 | ✅ | 定义 tasks/task_logs 结构，封装 CRUD 与日志接口 |
| 2 | 实现 ReminderExecutor + SandboxExecutor | - | 验证基础闭环（任务入库 → 执行 → 写回）|
| 3 | 接入 HostExecutor（需人工确认） | - | 支持高权限任务，加入白名单与确认流程 |
| 4 | 集成 AgentExecutor（多步 LLM 执行） | - | 处理复杂任务/规划，多轮执行 |
| 5 | 调度器 + 工具一体化 | - | APScheduler/Huey 等调度 + FastMCP 工具，让 Emilia/Orchestrator 调用 |
| 6 | 前端任务面板（可选） | - | 提供任务列表/确认/日志查看 UI |

## 10. 示例任务（提醒）

```json
{
  "id": "reminder-2025-01-08-21",
  "type": "reminder",
  "status": "pending",
  "manual_required": false,
  "scheduled_for": "2025-01-08T21:00:00+08:00",
  "payload": {
    "message": "晚上9点喝水",
    "tts_voice": "emilia"
  },
  "created_at": "2025-01-08T12:05:33+08:00",
  "updated_at": "2025-01-08T12:05:33+08:00"
}
```
