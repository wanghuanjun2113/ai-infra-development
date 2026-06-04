# AGENTS

## 工程原则

- 遵循尽量简单原则。先使用清晰的脚本和文档，只有在真实重复使用后再增加抽象。
- `development.yaml` 是配置文件，只放当前工作框架、框架 git 信息、模型路径、数据集路径、部署参数、性能测试参数和精度测试参数。
- 不要把项目介绍、README 式说明、目录说明放进 `development.yaml`。
- 工程目录是约定，不鼓励通过配置修改：`code/`、`runs/`、`profiling/`、`logs/`。

## Agent 执行原则


### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 脚本原则

- 单一脚本完成单一功能，避免一个脚本生成另一个脚本再执行。
- 具体框架的启动流程放在对应的专用脚本中，不放进通用辅助脚本。
- `scripts/dev.py` 只做通用辅助操作，例如检查配置、同步代码、生成通用 run 目录；不要承载具体框架的启动流程。
- xllm 启动只使用 `bash scripts/launch_xllm.sh`。
- `scripts/launch_xllm.sh` 自己读取 `development.yaml`、创建 `runs/deploy/<run_id>/`、选择空闲 NPU、设置 `ASCEND_RT_VISIBLE_DEVICES`、写入部署产物并启动服务。

## 目录约定

- `code/` 放本地框架源码，不提交到当前仓库。
- `runs/` 放临时运行产物和报告，不提交到当前仓库。
- `profiling/` 放本地 profiling 产物，不提交到当前仓库。
- `logs/` 放本地日志，不提交到当前仓库。
- `frameworks/` 下一个框架一个适配文档，例如 `frameworks/xllm.md`。
- `skills/` 只写工作流和规则，不复制脚本实现逻辑。

## xllm 工作流

- 编译 xllm 时参考 `frameworks/xllm.md` 的 Build 部分。
- 启动 xllm 时运行：

```bash
bash scripts/launch_xllm.sh
```

- 默认启动脚本会读取 `development.yaml` 中的模型和部署参数。
- 默认启动脚本会检查 `npu-smi info`，选择空闲卡，并把实际可见卡写入 run 目录下的 `visible_devices.txt`。
- 启动日志、PID、manifest 和报告写入 `runs/deploy/<run_id>/`。

## 修改要求

- 修改配置结构时，同步更新 `shared/devkit.py`、相关脚本、README、framework 文档和测试。
- 修改 skill 时保持 `SKILL.md` 简洁；框架差异写入 `frameworks/<framework>.md`，确定性动作写入脚本。
- 不要提交 `code/`、`runs/`、`profiling/`、`logs/` 中的本地内容。
