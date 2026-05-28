# ai-infra-development

`ai-infra-development` 是面向 AI 推理基础设施开发维护的多框架工作台。

它把 Codex skills、框架适配文档、一个统一脚本入口和本地运行产物组织在同一个工程里，用于辅助 `xllm`、`vllm-ascend`、`sglang` 等推理框架的编译、安装、部署、性能测试、精度测试、代码 review、问题修复和性能优化。

## 快速开始

```bash
git clone git@github.com:your-org/ai-infra-development.git
cd ai-infra-development
bash scripts/bootstrap.sh
```

首次执行前，先修改根目录下的 `development.yaml`：

- 配置各框架的 Git 仓库地址和默认分支。
- 配置当前工作框架。
- 配置测试和部署使用的模型路径、数据集路径。
- 配置部署参数、性能测试 case、精度测试 case。

## 目录结构

```text
development.yaml  唯一顶层配置文件，保存框架、路径、默认测试参数等团队配置。
skills/           Codex skills，每个子目录是一个独立 skill。
frameworks/       框架适配文档，一个框架一个文件。
shared/           共享 Python 工具，目前只保留 devkit.py。
scripts/          用户直接执行的入口脚本，主要使用 scripts/dev.py。
tests/            轻量级测试，覆盖最小公共能力。
code/             本地源码目录，由 bootstrap 拉取，不提交到当前仓库。
runs/             标准化运行产物，如 build/deploy/perf/accuracy 报告，不提交。
profiling/        本地 profiling 产物，不提交。
logs/             本地日志，不提交。
```

## 常用命令

```bash
bash scripts/bootstrap.sh
python3 scripts/dev.py check
python3 scripts/dev.py sync status
python3 scripts/dev.py sync pull --framework xllm
python3 scripts/dev.py run build --framework xllm
bash scripts/launch_xllm.sh
python3 scripts/dev.py run perf --framework xllm --case smoke
python3 scripts/dev.py run accuracy --framework xllm --case ceval
```

## 典型流程

```text
1. 从 GitHub clone 本工程。
2. 修改 development.yaml。
3. 执行 scripts/bootstrap.sh 拉取各框架源码。
4. 使用 Codex skills 或 scripts/ 执行编译、部署、性能测试、精度测试。
5. 在 runs/ 和 profiling/ 下查看标准化产物和报告。
```

## 设计原则

```text
development.yaml  只保留一个配置入口。
scripts/          单一脚本完成单一功能，避免一个脚本生成另一个脚本再执行。
scripts/dev.py    只做通用辅助操作，不承载具体框架的启动流程。
frameworks/       一个框架一个适配文档。
skills/           只写流程，不复制脚本逻辑。
runs/             保存可复查的执行结果。
```

第一版工程刻意保持简单：不要提前拆配置、不要提前做复杂 schema、不要为每个动作维护一套脚本。等真实使用中出现稳定重复需求，再把逻辑沉淀为更严格的脚本。
