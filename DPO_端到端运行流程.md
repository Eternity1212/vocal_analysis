# DPO 端到端运行流程

## 1. 总体目标

这套流程的目标是把现有仓库拆成三层来协作：

1. `Client/` 继续承担前端在线评分链路，不改现有实现。
2. 新增的 `configs/ + src/ + scripts/` 承担 DPO 数据构造、训练和导出。
3. `sft/` 继续作为参考模型思路与旧脚本来源，必要时只借鉴逻辑，不直接侵入修改。

整个流程可以概括为：

`前端音频/老师评分 -> Client 产出在线模型评分结果 -> 新脚本整理 teacher/model manifest -> 生成 DPO 数据集 -> 训练 DPO 策略模型 -> 导出前端可消费结果`

---

## 2. 目录职责

- `Client/`
  负责从前端任务里下载音频、执行当前在线评分脚本、回传 `scores_data`。
- `sft/`
  提供现有 MFCC、标签读取、参考模型输出格式的历史实现依据。
- `data/external/client_downloads/`
  存放前端或客户端下载下来的老师评分文件。
- `data/processed/manifests/`
  存放标准化后的 `teacher_manifest.jsonl` 与 `model_manifest.jsonl`。
- `data/processed/dpo/`
  存放最终 `train_dpo.jsonl`、`val_dpo.jsonl`、`test_dpo.jsonl`。
- `outputs/frontend/`
  存放训练后导回前端的 payload。

---

## 3. 前端到 DPO 数据

### 3.1 前端到 Client

当前在线评分流程仍然走原有 `Client`：

1. 前端创建评分任务。
2. `Client/src/api/client.py` 拉取待处理任务。
3. `Client/src/processor/task_manager.py` 下载音频。
4. `Client/src/processor/model_runner.py` 调用 `Client/scripts/inference_score_file.py`。
5. 脚本产出 `predictions.xlsx`。
6. `model_runner.py` 将其解析成前端需要的 `scores_data` 并提交回后端。

这一步不直接产出 DPO 数据，但会留下可复用的模型评分结果来源。

### 3.2 老师评分标准化

如果老师评分来自前端下载的 Excel 文件，执行：

```powershell
python scripts/prepare_teacher_manifest.py `
  --source-type client_excel_dir `
  --input-dir data/external/client_downloads/teacher_scores `
  --audio-dir data/raw/audio `
  --mfcc-dir data/processed/mfcc `
  --voice-part sopran `
  --output-path data/processed/manifests/teacher_manifest.jsonl
```

如果老师评分来自 `sft` 风格的 `Label/*.xlsx`，执行：

```powershell
python scripts/prepare_teacher_manifest.py `
  --source-type label_dir `
  --input-dir data/raw/labels_teacher `
  --audio-dir data/raw/audio `
  --mfcc-dir data/processed/mfcc `
  --voice-part sopran `
  --score-dims 10 `
  --output-path data/processed/manifests/teacher_manifest.jsonl
```

### 3.3 参考模型评分标准化

如果模型评分来自 `Client/outputs/task_*/predictions.xlsx`，执行：

```powershell
python scripts/prepare_model_manifest.py `
  --source-type client_task_dir `
  --input-path Client/outputs `
  --output-path data/processed/manifests/model_manifest.jsonl
```

如果模型评分来自 `sft` 批量预测结果 Excel，执行：

```powershell
python scripts/prepare_model_manifest.py `
  --source-type sft_prediction_excel `
  --input-path path/to/results.xlsx `
  --output-path data/processed/manifests/model_manifest.jsonl
```

### 3.4 构造 DPO 数据集

```powershell
python scripts/build_dpo_dataset.py `
  --teacher-manifest data/processed/manifests/teacher_manifest.jsonl `
  --model-manifest data/processed/manifests/model_manifest.jsonl `
  --output-dir data/processed/dpo
```

---

## 4. DPO 训练

### 4.1 训练参考模型

如果你要走新框架下的参考训练：

```powershell
python scripts/train_reference.py --config configs/train_reference.yaml
```

如果你已经有历史参考权重，也可以跳过这步，直接把权重放到：

```text
models/reference/reference_model.pt
```

### 4.2 训练 DPO 策略模型

```powershell
python scripts/train_dpo.py --config configs/train_dpo.yaml
```

训练完成后会得到新的策略模型权重，例如：

```text
outputs/dpo/policy_model_dpo.pt
```

---

## 5. DPO 结果回到前端

### 5.1 导出新模型预测

```powershell
python scripts/export_predictions.py `
  --config configs/train_dpo.yaml `
  --checkpoint outputs/dpo/policy_model_dpo.pt `
  --input-jsonl data/processed/dpo/test_dpo.jsonl `
  --output-jsonl outputs/frontend/predictions.jsonl
```

### 5.2 转换为前端兼容 payload

```powershell
python scripts/export_frontend_payload.py `
  --input-jsonl outputs/frontend/predictions.jsonl `
  --output-jsonl outputs/frontend/frontend_payload.jsonl
```

输出结构会对齐当前前端/客户端使用的 `scores_data` 风格，例如：

```json
{
  "sample_id": "000001",
  "voice_part": "sopran",
  "scores_data": {
    "Vibrato": 4,
    "Throat": 3,
    "Position": 5,
    "Open": 2,
    "Clean": 4,
    "Resonate": 3,
    "Unify": 4,
    "Falsetto": 2,
    "Chest": 5,
    "Nasal": 3
  }
}
```

这样你后续无论是继续走接口回传，还是给前端做离线展示，都可以直接复用同一套字段。

---

## 6. 推荐的实际落地顺序

1. 先固定老师评分来源，到底是前端下载 Excel，还是 `Label/*.xlsx`。
2. 再跑 `prepare_teacher_manifest.py` 和 `prepare_model_manifest.py`。
3. 确认 `teacher_manifest.jsonl` 与 `model_manifest.jsonl` 的 `sample_id` 能对上。
4. 构造 `train/val/test_dpo.jsonl`。
5. 准备或迁移参考模型权重。
6. 跑 `train_dpo.py`。
7. 跑 `export_predictions.py` 和 `export_frontend_payload.py`。
8. 最后再决定是否把新策略模型接回现有 `Client` 在线链路。

---

## 7. 当前边界

当前新增框架已经把“接真实数据”的入口打通，但仍保留以下边界：

- 不修改 `Client/` 和 `sft/` 的现有代码。
- 新框架默认以统一的 10 维评分顺序为准。
- 新框架默认把 `Client` 结果当成“数据来源之一”，不是直接把 `Client` 代码改造成 DPO Trainer。
- 如果后续要真正复用 `sft` 的原始模型结构和权重，还需要在 `src/models/` 里补一层适配，而不是改旧脚本。
