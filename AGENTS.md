# 仓库协作指南

## 沟通与更新语言
默认使用中文与用户协作，包括进度更新、状态说明、问题定位、变更说明和最终答复。除非用户明确要求使用其他语言，否则不要切换到英文。

当修改仓库配置、说明文档或协作约定时，优先同步维护中文内容；如果需要保留英文术语，使用中文解释其含义与用途。

## 项目结构与模块组织
`code/` 目录包含主要工作流脚本：`CAM_S.py` 用于模型训练，`MFCCnew.py` 用于生成 MFCC 特征，`VAL_WIN.py` 用于偏向 Windows 环境的推理与导出。

`code_new/` 目录保存较新的或替代性的实验脚本，包括 Linux 风格的训练脚本 `CAM_S.py`、较旧版 MFCC 提取脚本 `MFCC.py`，以及完整的验证/对比流水线 `val_accuracy_analysis_concrete_full.py`。

每个配置好的数据集根目录应包含 `Audio/`、`MFCC_Output/` 和 `Label/`。运行过程中产生的目录与文件，例如 `runs/`、`logs_*`、`Audio_44100/` 以及导出的 `.xlsx` 结果文件，属于运行产物，不应提交到版本库。

## 构建、测试与开发命令
当前仓库没有包级封装或 Makefile。修改代码后，请直接运行脚本，并先检查各文件底部附近硬编码的路径配置，如 `data_dir`、`output_dir`、`pretrained_weights` 和 `log_dir`。

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch pandas numpy librosa openpyxl tensorboardX torchlibrosa soundfile matplotlib pillow
python code\MFCCnew.py
python code\CAM_S.py
python code\VAL_WIN.py
```

只有在你明确希望走替代实验流程时，才使用 `code_new\*.py` 下的脚本。

## 编码风格与命名约定
使用 Python，统一 4 空格缩进，导入顺序按 `stdlib`、第三方库、本地模块分组。函数名和变量名使用 `snake_case`，模型类和数据集类使用 `CamelCase`。

现有入口文件名如 `CAM_S.py` 和 `VAL_WIN.py` 可能被其他脚本直接引用，不要轻易重命名。

仓库中未提供格式化器或 linter 配置。请保持改动兼容 `black` 风格，路径处理尽量显式，并优先做局部、小范围修改，避免无必要的大型重构。

## 测试指南
当前工作区未包含自动化 `tests/` 目录，也没有覆盖率门禁。验证改动时，优先只重跑受影响的阶段，例如：MFCC 提取、检查点加载、训练启动或推理导出。

如果你新增测试，请将其放在 `tests/` 目录下，并重点覆盖数据集加载、张量形状以及 Excel I/O 边界情况。

## 提交与 Pull Request 指南
当前工作区无法推断出既有 Git 提交规范，因此建议使用简短、祈使句式的提交信息，最好采用 Conventional Commit 风格，例如：`fix: handle missing MFCC files in validation`。

Pull Request 应说明以下内容：修改了哪些脚本路径、是否引入了新依赖、是否需要调整路径配置，以及通过什么方式完成了验证。如果运行会生成表格或模型检查点，也应注明输出文件位置。

## 安全与配置建议
不要提交模型权重、原始音频、生成的 MFCC 表格，或机器相关的绝对路径。若后续需要扩展项目，优先将可复用路径迁移到环境变量或独立配置模块中，而不是继续散落在脚本内部。
