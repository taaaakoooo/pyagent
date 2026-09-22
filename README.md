# pyagent

一个可自托管的编码 Agent：终端 UI + 工具沙箱 + 技能路由 + MCP 接入。

默认启动全屏 TUI（Elm 架构：Model / Update / View / Cmd），也可以 `--cli` 退回行式交互。
所有文件操作都被限制在 `agent.workspace_root` 之内；会改动系统或文件的工具在执行前需要你逐次审批。

---

## 目录

- [特性](#特性)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [运行方式](#运行方式)
- [TUI 键位](#tui-键位)
- [行式 CLI 命令](#行式-cli-命令)
- [配置](#配置)
- [内置工具](#内置工具)
- [技能（Skill）](#技能skill)
- [MCP 服务器](#mcp-服务器)
- [权限审批](#权限审批)
- [会话与落盘数据](#会话与落盘数据)
- [运行测试](#运行测试)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

---

## 特性

- **双界面**：默认全屏 TUI；`--cli` 使用行式交互（适合脚本或没有终端能力的环境）。
- **路径沙箱**：`read_file` / `write_file` / `replace_line` / `run_shell` 等全部相对工作区解析，越界路径直接拒绝。
- **权限审批**：改文件的工具先给出 diff 预览，`run_shell` 额外给出风险标签（递归强删、提权、格式化磁盘、远程代码执行等）。只读工具免审批，MCP 工具默认必须审批。
- **技能路由**：用关键词命中决定这一轮放行哪些工具，避免把所有工具都塞给模型。
- **MCP 接入**：`stdio` 与 `http` 两种传输，自动发现工具、处理重名冲突、并发调用。
- **会话持久化**：JSONL 追加写入，支持多会话，进程重启后自动恢复历史。
- **大输出落盘**：超长工具输出写入磁盘，由 `read_disk_data` 按需回读，避免撑爆上下文。
- **上下文压缩**：token 剩余量低于上限的 10% 时，自动从最旧的非 system 消息开始丢弃，直到用量回到 90% 以下；被丢弃的消息归档到磁盘而不是直接消失。用量达到 `warning_threshold` 时另有提醒。

---

## 环境要求

| 项目 | 要求 |
|---|---|
| Python | 3.9 或更高（本项目在 3.9.12 上开发与测试） |
| 依赖 | `PyYAML>=6.0`、`rich>=13.0`（见 `requirements.txt`） |
| LLM | 默认接入 DeepSeek（OpenAI 兼容接口） |
| 终端 | TUI 需要真实终端（Windows Terminal / PowerShell / iTerm 等），无法在管道或重定向下运行 |

`PyYAML` 是可选的：没装时会退回到内置的极简 YAML 解析器，但仍建议安装。

---

## 快速开始

### 1. 进入项目并创建虚拟环境

```powershell
cd E:\pyagent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux：

```bash
cd /path/to/pyagent
python3 -m venv .venv
source .venv/bin/activate
```

> PowerShell 若提示脚本被禁止执行，先运行：
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

程序从**环境变量**读取密钥，默认变量名是 `DEEPSEEK_API_KEY`（可在 `config.yaml` 的 `llm.api_key_env` 改）。

临时设置（只对当前终端有效）：

```powershell
$env:DEEPSEEK_API_KEY = "sk-your-key-here"
```

```bash
export DEEPSEEK_API_KEY="sk-your-key-here"
```

永久设置（Windows，写入用户环境变量，新开终端生效）：

```powershell
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "sk-your-key-here", "User")
```

> **注意**：项目**不会**自动读取 `.env` 文件。`.gitignore` 里列了 `.env` 只是防止你误提交密钥，程序本身没有 dotenv 加载逻辑。请使用上面真正的环境变量。

### 4. 启动

```bash
python main.py
```

看到带边框的输入框就说明启动成功了（下面是一个全新会话的首屏）：

```
 pyagent  session:default  status:idle  turn:0/15  tokens:0/128000  skills:on
╭──────────────────────────────────── Chat ────────────────────────────────────╮
│ Ask something to get started.                                                │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────  > 输入  ──────────────────────────────────╮
│ > █                                                                          │
╰─────────── Enter send  Esc commands  PgUp/PgDn scroll  Tab panels ───────────╯
```

---

## 运行方式

```bash
python main.py                 # 启动 TUI（默认）
python main.py --cli           # 启动行式 CLI
python -m tui                  # 等价于 TUI，可多传一个 --fps
```

### 命令行参数

| 参数 | 作用 | 适用 |
|---|---|---|
| `--config PATH` | 指定配置文件，默认 `config/config.yaml` | 两者 |
| `--session NAME` | 打开指定会话，默认取 `session.default_id` | 两者 |
| `--cli` | 用行式 CLI 代替 TUI | — |
| `--fps N` | 空闲时的重绘频率（默认 12） | 仅 TUI |

示例：

```bash
python main.py --session work            # 打开名为 work 的会话
python main.py --config my-config.yaml   # 用另一份配置
python -m tui --fps 20                   # 提高空闲刷新率
```

---

## TUI 键位

界面文案全部为中文（帮助面板、页脚提示、面板标题与表头、空状态、状态标签、审批浮层、权限选项、操作提示、行式 CLI 的输出）。唯一保留英文的是 `pyagent` 品牌名、`tokens:` 计数、命令名（`/skills`、`/mcp`、`/session`、`exit`）和技术标识（`stdio`/`http`、工具名、服务名、会话 id）。

完整的快捷键表就写在 TUI 里：`Tab` 切到 `帮助` 面板即可看到，和本节内容一致。

输入框有**两个层**，边框颜色和标题会告诉你当前在哪一层。这么设计是因为终端**无法区分「Shift+字母」和「输入大写字母」**——两者收到的都是同一个字符。所以命令都要在命令层里按，保证命令永远不会吃掉你正在打的字。

| 输入框状态 | 边框 | 标题 |
|---|---|---|
| 输入层（默认） | 青色 | `> 输入` |
| 命令层 | 品红 | `命令层` |
| 正在回复（输入锁定） | 黄色 | `输入已锁定（agent 回复中）` |
| 焦点在其他面板 | 灰色 | `输入 · 按 Tab 回到对话` |

### 对话页 · 输入层

| 按键 | 作用 |
|---|---|
| 任意字符 | 输入（大小写、中英文都正常） |
| `Enter` | 发送 |
| `Esc` | 有内容 → 清空输入；正在回复 → 取消本轮；输入框为空 → 进入命令层 |
| `↑` / `↓` | 输入框为空时：滚动聊天记录；正在输入时：翻输入历史 |
| **鼠标滚轮** | **上下滚动聊天记录**（终端不转发鼠标时等同 `↑`/`↓`） |
| `PgUp` / `PgDn` | 上下翻页聊天记录 |
| `Backspace` | 删除最后一个字符 |
| `Delete` / `Home` | 清空输入框 |
| `Tab` / `Shift+Tab` | 切换面板 |

> **鼠标被 TUI 接管了。** 为了能识别滚轮，程序启动时会开启鼠标上报，退出时自动关闭。代价是**拖拽选中文字需要按住 `Shift`**（个别终端用 `Option`/`Alt`）——这是滚轮滚动无法避免的取舍。程序退出后会恢复成普通终端，不影响后续使用。

### 命令层（在空输入框按 `Esc` 进入）

| 按键 | 作用 |
|---|---|
| `Shift+Q` | 退出程序（agent 正在回复时需再按一次确认） |
| `Shift+C` | 正在回复时取消本轮；空闲时退出 |
| `i` | 回到输入层 |
| `Esc` | 回到输入层 |
| 其他字母 | 回到输入层，并把这个字母输进去 |

> 想在提问里以大写 `Q` 开头（比如 `Q&A`），按 `i` 回到输入层再打即可。平时正常打字完全不受影响。

### 退出方式

三条路径，任选其一：

| 方式 | 操作 | 说明 |
|---|---|---|
| 输入指令 | 在输入框键入 `/exit`（或 `/quit`、`/q`）后回车 | 唯一**不需要切层**的方式，正在打字时直接可用 |
| 命令层 | `Esc` → `Shift+Q` | 输入框非空时先按 `Esc` 清空，再按 `Esc` 进命令层 |
| 信号 | `Ctrl+C` | 等价于 `Shift+C`：回复中取消本轮，空闲时退出 |

行为约定：

- **正在回复时不会直接退出。** 第一次 `Shift+Q`（或 `/exit`）只会在输入框下方提示确认，再按一次才真的退出；按其他任意键即取消确认。避免手滑丢掉正在生成的回答。
- **退出一定恢复终端。** 鼠标上报、光标可见性、字符属性都会在退出时复位；程序还会注册 `atexit` 兜底，即使主循环异常退出也不会把终端留在鼠标上报状态。
- **`Ctrl+C` 是优雅中断，不是强杀。** 第一次 `Ctrl+C` 走 `InterruptMsg`，和 `Shift+C` 完全同一条路径；**卡住不退出时再按一次 `Ctrl+C`** 会恢复默认信号处理并强杀，不会出现"Ctrl+C 按了没反应"。
- **退出后打印一行摘要**，因为聊天记录会随备用屏幕一起消失：

```
pyagent: 已退出 · 会话 default · 12 条消息 · 本轮 3 次 · 历史 .sessions/default.jsonl
```

### 面板页（技能 / MCP / 会话）

| 按键 | 作用 |
|---|---|
| `Tab` / `Shift+Tab` | 切换面板 |
| `↑` / `↓` | 移动选择 |
| `Space` / `Enter` | 开关技能、启用/停用 MCP 服务器、切换会话 |
| `Shift+R`（或 `r`） | 重新加载当前面板 |
| **鼠标滚轮** | 上下移动高亮项 |
| `Shift+N`（或 `n`） | 新建会话（会话面板） |
| `Shift+D`（或 `d`） | 删除会话（会话面板） |

### 审批浮层

| 按键 | 作用 |
|---|---|
| `a` / `d` | 本次允许 / 本次拒绝 |
| `A` / `D` | 会话内始终允许 / 会话内始终拒绝 |
| `↑` `↓` `←` `→` | 移动选项 |
| `Enter` / `Space` | 确认当前选项 |
| `Esc` | 本次拒绝 |

> 审批浮层会拦截按键，需要先做出选择（或按 `Esc` 拒绝）才能继续使用其他快捷键。

---

## 行式 CLI 命令

`python main.py --cli` 之后，直接输入内容即可对话；以 `/` 开头的是内置命令（命令名保持英文，输出提示为中文）：

| 命令 | 作用 |
|---|---|
| `/skills list` | 列出已加载的技能与关键词 |
| `/skills reload` | 重新扫描技能目录 |
| `/skills on` / `off` | 开启 / 关闭技能路由（关闭后所有工具都可见） |
| `/skills status` | 查看本轮命中的技能与放行的工具 |
| `/mcp list` | 列出 MCP 服务器及连接状态 |
| `/mcp reload` | 重连所有 MCP 服务器 |
| `/mcp enable <name>` / `disable <name>` | 启用 / 停用某个服务器 |
| `/mcp tools [name]` | 列出（某个服务器的）MCP 工具 |
| `/session list` | 列出所有会话 |
| `/session new [name]` | 新建并切换会话 |
| `/session use <name>` | 切换到已有会话 |
| `/session current` | 显示当前会话与归档目录 |
| `/session delete <name>` | 删除会话 |
| `/help` | 显示帮助 |
| `exit` / `quit` | 退出 |

---

## 配置

配置文件默认是 `config/config.yaml`，用 `--config` 可指定其他路径。缺失的字段会自动用默认值补齐。

```yaml
llm:
  provider: deepseek
  api_base: "https://api.deepseek.com/v1"
  model: "deepseek-v4-flash"
  api_key_env: "DEEPSEEK_API_KEY"
  fallback_api_key_envs: []      # 保持为空，除非确实有可互换的密钥
  timeout_seconds: 180
  stream: true

agent:
  max_turns: 15                  # 单轮对话内最多的工具调用轮次
  workspace_root: "."            # 所有文件工具的沙箱根目录

token_usage:
  context_window: 128000
  max_session_tokens: null       # 设为整数可覆盖上面的上下文窗口
  warning_threshold: 0.7         # 用到这里比例时提醒
  warn_once: true

session:
  storage_dir: ".sessions"
  default_id: "default"
  isolate_tool_output: true      # 每个会话用独立的 .tool_outputs/<会话名>/

skills:
  enabled: true
  dir: "skills"
  files: []                      # 额外的技能文件（可选）
  inject_system_prompt: true     # 把技能说明注入 system prompt
  unmatched_tools: "readonly"    # none | readonly | all
  always_visible_sources:
    - "mcp"

mcp:
  enabled: true
  autostart: true                # 启动时自动连接
  servers: []                    # 见「MCP 服务器」
```

### 关键字段说明

| 字段 | 说明 |
|---|---|
| `llm.model` | `deepseek-v4-flash` 是服务端别名，会解析到 `deepseek-flash`。若某天别名失效，改成规范名即可。 |
| `llm.api_key_env` | 读取密钥的环境变量名；找不到时会报错并提示变量名。 |
| `agent.workspace_root` | 沙箱边界。设成 `"."` 就是项目根目录；工具无法读写这个目录之外的文件。 |
| `skills.unmatched_tools` | 没有关键词命中时的策略。见[技能](#技能skill)。 |
| `session.isolate_tool_output` | 开启时工具输出按会话隔离到 `.tool_outputs/<会话名>/`。 |

---

## 内置工具

工具顺序固定为下列 7 个（注册表按固定顺序暴露，便于维护）：

| 工具 | 作用 | 是否需要审批 |
|---|---|---|
| `read_file` | 读取工作区内文本文件（支持 `offset` / `limit`） | 否 |
| `list_dir` | 列出工作区目录（可递归） | 否 |
| `read_regex` | 用正则搜索文件并返回命中行 | 否 |
| `read_disk_data` | 回读落盘的大体积工具输出 | 否 |
| `write_file` | 创建或覆盖文件 | **是**（附 diff 预览） |
| `replace_line` | 按行号精确替换某一行 | **是**（附 diff 预览） |
| `run_shell` | 在沙箱内执行 shell 命令 | **是**（附风险标签） |

只读工具（`read_*` / `list_*` / `get_*`）自动放行；需要审批的工具在每次执行前弹浮层。若审批回调不可用，系统会**安全兜底**：只允许只读工具。

`run_shell` 会匹配下列风险模式并展示给用户：

`destructive_delete`（递归强删）、`force_delete`、`elevated_privilege`（sudo 等）、`permission_change`（chmod 777）、`disk_format`、`raw_disk_write`（dd）、`remote_code_exec`（`curl | sh`）、`device_redirect`、`obfuscated_exec`（编码命令）、Windows 下的 `rmdir /s` 等。

---

## 技能（Skill）

技能把**关键词**和**工具**绑在一起：只有当你这句话命中了关键词，对应工具才会暴露给模型。这样模型不会因为看到一堆用不上的工具而产生幻觉调用。

技能是带 YAML front matter 的 Markdown 文件，放在 `skills/` 目录里会被自动发现：

```markdown
---
name: workspace-files
description: Reading and editing files inside the workspace
keywords:
  - 文件
  - read file
  - edit file
tools:
  - read_file
  - list_dir
  - write_file
  - replace_line
---

When working with files in the workspace:

1. Always read the current contents before editing.
2. Prefer `replace_line` over `write_file` for small changes.
3. Use workspace-relative paths. Never escape the workspace root.
```

| front matter 字段 | 必填 | 说明 |
|---|---|---|
| `name` | 是 | 技能名，需唯一 |
| `keywords` | 是 | 命中关键词，至少一个；中英文皆可 |
| `tools` | 否 | 命中后放行的工具名 |
| `description` | 否 | 一句话说明，会写进 system prompt |

front matter 之后的内容是**指令正文**，会注入 system prompt（受 `inject_system_prompt` 控制）。

### `unmatched_tools` 策略

| 取值 | 没命中关键词时 |
|---|---|
| `none` | 一个工具都不给。**不建议**：模型会转而用纯文本编造工具调用语法。 |
| `readonly` | 只读工具仍然可用，改文件的工具仍需命中关键词。**默认值。** |
| `all` | 所有工具都可用（等于忽略技能路由）。 |

`always_visible_sources` 指定哪类工具始终可见，默认 `["mcp"]`，即 MCP 工具不受技能路由限制。

加载失败不会中断启动，错误可以通过 CLI 的 `/skills list` 或 TUI 的技能面板查看。

---

## MCP 服务器

在 `config.yaml` 的 `mcp.servers` 下配置。MCP 工具会以**远端工具名原样**注册；若与其他工具（本地或另一个服务器）重名，会自动加数字后缀消歧，例如先注册的占用 `read_file`，后来的变成 `read_file_1`。**MCP 工具默认都需要审批**，因为它们运行在本地沙箱之外。

连接时会先尝试 `server/discover` 握手（要求服务端 `supportedVersions` 里含 `2026-07-28`），失败则回退到标准的 `initialize` + `initialized` 握手。

### stdio（本地子进程）

```yaml
mcp:
  enabled: true
  autostart: true
  servers:
    - name: "filesystem"
      transport: "stdio"
      command: "npx"
      args:
        - "-y"
        - "@modelcontextprotocol/server-filesystem"
        - "."
      env: {}
      timeout_seconds: 60
      enabled: true
```

`command` 和 `args` 是子进程的启动命令，双方通过 stdin/stdout 交换 JSON-RPC 消息；stderr 单独接管并写日志，避免管道堵塞。

### HTTP

```yaml
    - name: "remote"
      transport: "http"
      url: "http://127.0.0.1:8080/mcp"
      headers: {}
      timeout_seconds: 60
      enabled: true
```

`transport` 也可写 `streamable-http`、`streamable_http` 或 `sse`——这四个值都会归一化成同一种基于 POST 的 HTTP 传输。

> **已知限制**：该 HTTP 传输**不支持 SSE 流式响应**。如果服务端返回 `Content-Type: text/event-stream`，会直接报错并提示改用 stdio 服务器。

HTTP 模式会先探测 `/mcp` 路径，失败再回退到根路径，成功后记住可用的端点。单次 `tools/call` 的等待上限是 60 秒（`MCP_DEFAULT_TIMEOUT_SECONDS`，也可由 `timeout_seconds` 覆盖），超时即失败。

配置校验会检查：名称非空且不重复、`stdio` 必须有 `command`、非 `stdio` 必须有 `url`、`timeout_seconds > 0`。校验不过会直接拒绝启动并给出具体位置。


---

## 权限审批

一次完整的审批流程分三步：

1. **先看会话记忆**——这个工具本轮是否已被「始终允许/始终拒绝」。
2. **需要交互时**，构建一个带 **diff 预览**（改文件类工具）和**风险标签**（`run_shell`）的审批请求交给界面。
3. **没有审批回调时安全兜底**——只放行只读工具。

四种结果：

| 结果 | 快捷键 | 效果 |
|---|---|---|
| 本次允许 | `a` | 只放行这一次 |
| 本次拒绝 | `d` | 只拒绝这一次 |
| 会话内始终允许 | `A` | 本会话内后续同名工具直接放行 |
| 会话内始终拒绝 | `D` | 本会话内后续同名工具直接拒绝 |

> **注意**：「会话内」指的是**当前会话**。切换到别的会话时，已记住的允许/拒绝项会被自动清空，避免在会话 A 里选的「始终允许 `run_shell`」悄悄授权了会话 B。重新选中当前会话属于空操作，不会清空。

---

## 会话与落盘数据

### 会话

对话历史以 JSONL 逐行追加，进程重启后会从上次的会话文件恢复（包括工具调用记录）。若恢复时发现缺少 system prompt，会自动补回。

会话管理有三种入口：

```bash
python main.py --session work        # 启动时直接指定
```

```bash
# 行式 CLI 内
/session new work
/session use work
/session list
```

TUI 里切到 Sessions 面板，用 `Shift+N` 新建、`Enter` 切换、`Shift+D` 删除。

会话 ID 规则：以字母或数字开头，只能包含字母、数字、`.`、`_`、`-`，最长 64 字符，不含空格和路径分隔符。

### 目录布局

```
.sessions/
  default.jsonl                 # 每个会话一个 JSONL 文件
  work.jsonl
.tool_outputs/
  <会话名>/                      # isolate_tool_output: true 时按会话隔离
    tool_results/               # 超长工具输出的归档，供 read_disk_data 回读
    turns/                      # 每轮 LLM 原始响应（turn_001.json…）
    cold/                       # 被上下文压缩丢弃的旧消息（turn_001.jsonl…）
```

两个目录都在 `.gitignore` 里，不会误提交。

---

## 运行测试

```bash
python -m unittest discover -s tests
```

当前 380 个测试通过（1 个跳过）。测试不需要真实 LLM，也不需要真实终端。

---

## 项目结构

```
main.py                    入口：默认 TUI，--cli 走行式 CLI
config/
  config.py                配置加载、API Key 解析、token 预算
  config.yaml              默认配置
internal/
  agent/                   Agent 主循环、会话恢复、技能/MCP 装配
  llm/                     LLM 客户端（流式、重试、错误处理）
  tools/                   工具定义、执行器、注册表、路径沙箱
  permission/              审批流程、风险分析、diff 预览
  diff/                    文件差异生成
  session/                 JSONL 会话存储
  skills/                  技能解析、关键词匹配、prompt 注入
  mcp/                     MCP 传输 / 客户端 / 管理器 / 工厂
  compact/                 上下文压缩
  types/                   共享数据类型
tui/                       Elm 架构终端界面
  model.py  update.py  view.py  msgs.py
  runtime.py               输入线程、消息队列、批量渲染循环
  agent_bridge.py          Agent 与 TUI 之间的异步桥接
skills/                    技能 Markdown 文件（自动发现）
tests/                     单元测试
```

---

## 常见问题

**启动报 `API key not found`**

没有设置环境变量。确认变量名与 `llm.api_key_env` 一致，并注意环境变量是**在启动终端之前**设置的：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
python main.py
```

**快捷键没反应 / 被 Cursor 抢走**

终端宿主（Cursor、VS Code、tmux）会拦截 `Ctrl` 组合键，所以本项目**不使用任何 Ctrl 快捷键**。退出请用 `Shift+Q`（或直接输入 `/exit`），取消请用 `Esc`。

**`Ctrl+C` 为什么没有直接杀掉进程？**

`Ctrl+C` 被接成了应用内中断（`InterruptMsg`），和 `Shift+C` 走同一条路径：正在回复就取消本轮，空闲就正常退出并恢复终端。这是有意为之——强杀会让备用屏幕、鼠标上报、光标状态来不及复位。如果退出流程真的卡住（worker 线程不结束、MCP 阻塞读取），**再按一次 `Ctrl+C`** 会恢复默认信号处理并立即强杀。

**滚轮滚不动 / 滚轮往输入框里打出字母**

滚轮依赖宿主终端主动上报鼠标事件（SGR `?1006`，退化为 X10 `?1000` 也能识别）。程序启动时会请求开启，退出时关闭。

**宿主是否转发鼠标由它自己决定，这一点实测过：** 在 Cursor 内置终端面板（xterm.js）里，即使我们发了 `?1000h ?1006h`，整个诊断日志里**一条鼠标报文都没有** —— 滚轮被宿主自己吞掉，然后重新发出上下方向键（Windows 下是 `\xe0` + 扫描码 `H`/`P`）。

所以 `↑`/`↓` 在输入框为空时用来滚动聊天记录 —— 在没转发鼠标的终端里，滚轮打出来的就是这两个键，这样滚轮依然能翻聊天记录；而在你正打字（输入框有内容）时，它们仍然是输入历史。

怎么判断自己在哪种情况：滚一下滚轮，看诊断日志里有没有 `\x1b` 开头的行。有 → 宿主转发了鼠标；没有，只有 `'\xe0'` + `'H'`/`'P'` → 没转发，滚轮是方向键（此时也能滚，靠上面的 Up/Down 绑定）。

滚动曾经会把 `H`、`P`、`>`、`~` 之类的字符打进输入框。根因不是终端，是读取层：终端输入没有消息边界，一条鼠标报文可能分几次到达。旧的解码器"拿到多少算多少"，半截序列的头部被丢掉后，剩下的字节就按普通字符解码了——而 X10 报文的载荷字节是 `32 + 数值`，全是可打印字符，于是变成乱码输入。

现在有三道防线，任意一条单独生效都不会漏字符：

1. **未完成的序列挂起**（`KeyStreamDecoder`）——要么等补齐，要么超时后整体丢弃，绝不会当作文字提交；
2. **认领报文的尾巴**——鼠标报文的前三个字节就能认出编码（X10 定长 3 字节载荷，SGR 到 `M`/`m` 结束）。若报文被切断，剩余字节会被"认领"下来而不是交给输入框，认领完还能还原成一次滚动；SGR 的参数位只可能是数字和 `;`，遇到别的字符立即收手，不会误吞你打的字；
3. **孤儿扫描码认领**——Windows 下方向键是 `\x00`/`\xe0` 前缀 + 扫描码两字节。若 `kbhit()` 一直报告"没东西"而扫描码其实在路上，等待再久也没用，所以读取线程会记下"欠一个扫描码"，把下一个字符认领成扫描码（`H`=上、`P`=下），而不是当成你打的字。

如果滚轮仍然异常，可以用诊断开关看宿主到底发了什么：

```powershell
$env:PYAGENT_INPUT_DEBUG = "input.log"
python -m tui
```

日志里每一行是读取线程收到的一个原始字符。滚一下滚轮，再看 `input.log`：

- 出现 `'\x1b'`、`'['`、`'<'`、`'M'` —— 宿主支持鼠标上报，格式是 SGR；
- 出现 `'\x1b'`、`'['`、`'M'` 后跟三个奇怪字符 —— 宿主用的是 X10 旧格式（同样支持）；
- 完全没有 `\x1b` 开头的行，只有 `'\xe0'` + `'H'`/`'P'` —— 宿主没转发鼠标，把滚轮变成了方向键；
- 出现 `'\x00'` 或 `'\xe0'` —— 走的是 Windows 控制台扫描码路径。若紧跟 `<scan=None>` 再跟一个 `<orphan scan 'H'>`，说明扫描码比读取窗口晚到，此时按第 3 道防线被认领，不会变成文字。
- `'<flush pending=... expecting=...>'` 表示终端在此处"卡住"过：`pending` 是被切断的报文头，`expecting` 是还没还的字节数。这就是第 2 道防线接手的时刻。

第 3 道防线的窗口是量出来的，不是猜的：`ORPHAN_SCAN_WINDOW_SECONDS = 2.0`。实测 99 次扫描码迟到里，平均 138ms、最快 0ms、4 次落在 500–800ms、最慢一次 3.2s —— 原来的 0.5s 太紧，才漏出了那几个 `H`/`P`。窗口放宽不会误吞打字，因为只要有别的字符到达，认领立即结束。

如果鼠标上报在你的宿主里反而碍事（比如想直接拖拽选中文字），加 `--no-mouse` 关掉：

```powershell
python -m tui --no-mouse
```

**想复制聊天记录里的文字**

鼠标被 TUI 接管了，按住 `Shift` 再拖拽即可选中（部分终端用 `Option`/`Alt`）。退出程序后终端恢复正常选择行为；也可以用 `--no-mouse` 全程不接管鼠标。

**想以大写 Q 开头提问**

终端里 Shift+Q 和打字的大写 Q 是同一个字符。按 `i`（或 `Esc`）回到输入层再打即可。

**方框、光标显示成乱码或 `?`**

终端编码不是 UTF-8。Windows 上可以：

```powershell
chcp 65001
```

或直接改用 Windows Terminal。程序本身已经对无法编码的字符做了替换处理，不会因此崩溃，但字形会退化。注意像 `✎` 这类字符在 GBK 代码页下根本编不出来，因此界面刻意只用了兼容性更好的字形。

**`python main.py` 报终端相关错误**

TUI 需要真实终端。管道、重定向、CI 环境下请改用 `python main.py --cli`。

**MCP 服务器连不上**

先看错误：

```bash
/mcp list
```

`stdio` 模式确认 `command` 在 PATH 里（例如 `npx`、`uvx`）；HTTP 模式确认 URL 可达、且服务端路径是 `/mcp` 或根路径。也可以临时设 `enabled: false` 跳过某个服务器。

**模型说自己调用了工具，但工具没执行**

检查技能路由：如果这句话没命中任何技能关键词，而 `unmatched_tools` 是 `none`，模型会拿不到任何工具转而编造调用。把策略改成 `readonly`（默认值）或 `all`，或在技能里补上对应关键词。用 `/skills status` 可以看本轮实际放行了哪些工具。

**工具输出太长看不清**

超过阈值的输出会落盘到 `.tool_outputs/<会话名>/tool_results/`，让模型用 `read_disk_data` 按需回读，而不是把全文塞进上下文。

---

## 许可

本仓库未附带开源许可证。如需使用或分发，请先与仓库所有者确认。
