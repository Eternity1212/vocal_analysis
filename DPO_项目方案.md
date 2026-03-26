# 基于参考模型与老师偏好的音频评分 DPO 项目方案

## 1. 项目目标

本项目的目标是：

1. 先使用原来的 6000 条数据训练一个基础音频评分模型，作为后续 DPO 训练的参考模型（Reference Model）。
2. 在参考模型的基础上，构造偏好数据集。
3. 每条偏好样本都来自同一段音频，并且包含两个评分结果：
   - 一个是参考模型的打分
   - 一个是老师的打分
4. 在 DPO 训练阶段，让新模型相对于参考模型，更偏好老师的打分结果。
5. 最终得到一个新的评分模型，使其在相同音频输入下，更倾向于输出接近老师偏好的评分。

这个项目的核心思想可以概括为：

`输入音频 -> MFCC 特征 -> 参考模型打分 / 老师打分 -> 构造偏好对 -> DPO 训练 -> 新策略模型更偏好老师评分`

---

## 2. 任务定义

### 2.1 输入

每条训练样本的原始输入是音频。

训练时的实际输入流程为：

1. 读取音频
2. 提取 MFCC 特征
3. 将 MFCC 特征输入模型
4. 得到评分结果

因此，从建模角度看：

- 原始输入是 `audio`
- 模型输入是 `mfcc`

可以定义为：

- `x_audio`: 原始音频
- `x_mfcc`: 由音频提取出的 MFCC 特征

---

### 2.2 输出

每条音频最终对应一个评分向量。

需要特别说明的是：你的原始评分体系可以是 40 维，但当前这个 DPO 实验阶段只使用其中固定的 10 个维度。因此，本文档中所有 `score`、`评分向量`、`chosen/rejected score` 的定义，都默认指这 10 个被选中的评分维度，而不是完整的 40 维原始标注。

结合你现有代码的结构，建议将进入当前 DPO 流程的评分统一定义为长度为 10 的整数向量，每个位置对应一个评分维度，每个分值范围为 `1~5`。

例如：

```json
[4, 3, 5, 2, 4, 3, 4, 2, 5, 3]
```

其中 10 个位置分别表示 10 个评分维度，例如：

1. vibrato
2. throat
3. position
4. open
5. clean
6. resonate
7. unify
8. falsetto
9. chest
10. nasal

如果后续你有新的维度命名，可以直接替换，但训练格式建议保持固定。

如果原始标注文件里仍然保存的是完整的 40 维评分，那么在构造 DPO 数据前，需要先按固定顺序筛选出这 10 个目标维度，再生成本文档中的 `model_score` 和 `teacher_score`。

结合你当前仓库里的代码，建议额外固定下面这条实现约定：

- `sft` 目录下的训练/验证脚本作为 DPO 参考模型的主线来源
- `Client` 目录主要负责从前后端流程中下载音频、收集老师评分文件或线上评分结果，不直接作为 DPO 训练格式本身

也就是说，当前文档里的 DPO 方案应优先对齐 `sft\_1_MFCC.py`、`sft\_2_CAM_S.py` 和 `sft\_3_val_accuracy_analysis_concrete_full.py` 这条离线训练链路；`Client` 更适合被视为“老师数据采集/在线任务入口”，而不是 DPO Trainer 直接依赖的数据格式。

---

## 3. 模型角色定义

本项目中需要明确区分两个模型角色。

### 3.1 参考模型 Reference Model

参考模型由原来的 6000 条数据训练得到。

它的作用不是最终上线，而是在 DPO 阶段作为固定参考项使用。参考模型需要满足以下要求：

1. 训练完成后参数冻结，不再更新。
2. 对每条音频都能输出一组评分。
3. 对每条音频都能输出每个评分维度上的概率分布或 logits。
4. 在 DPO 训练时，用它来计算参考对数概率 `log p_ref(y | x)`。

简单说，参考模型负责回答：

`对于同一段音频，旧模型更倾向于哪个评分结果。`

---

### 3.2 策略模型 Policy Model

策略模型是要通过 DPO 训练得到的新模型。

它的初始化方式建议为：

- 直接从参考模型拷贝权重初始化

这样做的原因是：

1. 新模型的起点稳定
2. DPO 训练更容易收敛
3. 新模型相对于旧模型的偏移更可控

策略模型在训练后负责回答：

`对于同一段音频，新模型是否比参考模型更偏好老师的评分。`

---

## 4. 你当前要使用的 DPO 数据形式

你已经明确了当前考虑的数据形式是：

1. 输入音频
2. MFCC 特征
3. 模型打分
4. 老师打分
5. 选择老师打分位置的 flag

基于你的要求，建议将每条 DPO 样本定义为如下结构。

### 4.1 单条样本字段定义

这里的 `model_score` 与 `teacher_score` 都表示“筛选后的 10 维评分子集”，不是完整的 40 维原始评分。

另外，结合当前代码现状，建议把 `voice_part` 也作为样本字段之一保留下来。因为 `Client/scripts/inference_score_file.py` 目前是按声部加载不同权重文件的，后续如果你继续沿用这套多声部模型，就不能把不同声部样本完全混成一个没有标记的数据集。

```json
{
  "sample_id": "000001",
  "audio_path": "data/audio/000001.wav",
  "mfcc_path": "data/mfcc/000001_MFCC.xlsx",
  "voice_part": "sopran",
  "model_score": [3, 2, 4, 3, 3, 2, 4, 2, 3, 2],
  "teacher_score": [4, 3, 4, 4, 4, 3, 4, 2, 3, 2],
  "teacher_preferred_flag": 1
}
```

其中：

- `sample_id`：样本唯一标识
- `audio_path`：原始音频路径
- `mfcc_path`：该音频提取后的 MFCC 特征路径
- `voice_part`：声部信息，例如 `sopran`、`mezzo`、`tenor`、`baritone`、`bass`
- `model_score`：参考模型对该音频的打分结果
- `teacher_score`：老师对该音频的打分结果
- `teacher_preferred_flag`：老师打分是否为偏好项

这里把 `mfcc_path` 写成 `_MFCC.xlsx`，是为了和你当前 `sft` 与 `Client` 代码的实际产物保持一致。若后续你在新框架里改成 `.npy` 或 `.pt` 缓存格式，也可以，但那时需要同步修改数据加载代码，不能再说“直接复用当前脚本”。

---

### 4.2 flag 的定义

由于你当前的目标是“希望结果能够偏好老师的打分一点”，因此在 DPO 阶段，这个 flag 可以定义为：

- `1`：老师打分是偏好答案
- `0`：老师打分不是偏好答案

在你当前这个项目里，绝大多数情况下应当使用：

```json
"teacher_preferred_flag": 1
```

因为你的训练目标本身就是让模型偏向老师评分。

也就是说，当前阶段你可以把它理解为：

`teacher_score = chosen`
`model_score = rejected`

---

### 4.3 DPO 训练时的 chosen / rejected 对应关系

基于你的数据格式，可以直接映射为：

- `x = mfcc`
- `y_chosen = teacher_score`
- `y_rejected = model_score`

如果 `teacher_preferred_flag = 1`，则：

- `teacher_score` 作为 chosen
- `model_score` 作为 rejected

如果未来你需要支持更复杂的偏好关系，也可以扩展这个 flag 逻辑，但当前项目不需要复杂化，按照上面的定义即可。

---

## 5. 推荐的数据存储格式

为了后续训练方便，建议偏好数据集使用 `jsonl` 格式保存。

原因是：

1. 一行一条样本，易于流式读取
2. 字段结构清晰
3. 方便后续增加新字段
4. 很适合构造 DPO 数据集

推荐文件名：

```text
train_dpo.jsonl
val_dpo.jsonl
test_dpo.jsonl
```

每行示例：

```json
{"sample_id":"000001","audio_path":"data/audio/000001.wav","mfcc_path":"data/mfcc/000001_MFCC.xlsx","voice_part":"sopran","model_score":[3,2,4,3,3,2,4,2,3,2],"teacher_score":[4,3,4,4,4,3,4,2,3,2],"teacher_preferred_flag":1}
{"sample_id":"000002","audio_path":"data/audio/000002.wav","mfcc_path":"data/mfcc/000002_MFCC.xlsx","voice_part":"mezzo","model_score":[2,2,3,2,3,2,3,2,2,2],"teacher_score":[3,3,4,3,4,3,4,2,3,2],"teacher_preferred_flag":1}
```

---

## 6. MFCC 特征层的设计要求

由于你的整个项目都建立在音频评分上，所以 MFCC 处理必须固定下来，不要在参考模型和 DPO 模型之间随意变化。

建议固定以下内容：

1. 采样率固定
2. `n_mfcc` 固定
3. 帧长固定
4. 帧移固定
5. 补零或截断策略固定

你当前代码中已经在做统一重采样和定长截断，这个思路可以保留，但建议正式项目中把 MFCC 配置写成独立配置项。

如果你的目标是“尽量复用当前 `sft` 代码和已有权重”，建议统一采用下面这一套 MFCC 规范：

- `mfcc.shape = [40, 128]`

对应含义为：

- 40 个 MFCC 维度
- 128 个时间帧

这里的“40”指的是声学特征里的 MFCC 维度数量，不是评分维度数量；当前文档中的评分向量维度仍然固定为 10。

这也与当前 `sft\_1_MFCC.py` 中的 `n_mfcc=40`、`sft\_2_CAM_S.py` 中的 `BatchNorm2d(40)` 保持一致。

需要特别注意：你当前 `Client/scripts/inference_score_file.py` 与 `Client/scripts/inference_scores.py` 里使用的是 `48000 Hz + 128 维 MFCC`，这与 `sft` 主链路并不一致。因此在正式做 DPO 之前，必须先二选一：

1. 继续复用 `sft` 现有参考模型和权重，那么 DPO 全流程统一使用 `40 x 128` MFCC
2. 全面切换到 `Client` 当前的 `128 x 128` MFCC 方案，那么参考模型、策略模型、训练数据和推理脚本都要一起重训或重构

基于你现有仓库的可复用程度，本文档推荐优先采用第 1 条，也就是以 `sft` 这条链路为准。

然后模型输入张量形状统一为：

- `[1, 40, 128]`

如果后续要改成别的形状，也必须保证参考模型和策略模型完全一致。

---

## 7. DPO 训练所需的真正信息

虽然你当前计划保存的数据格式是：

- `model_score`
- `teacher_score`
- `teacher_preferred_flag`

但是在真正训练 DPO 时，仅仅保存“最终分数”还不够。

你还必须能够得到：

1. 策略模型对 `teacher_score` 的对数概率
2. 策略模型对 `model_score` 的对数概率
3. 参考模型对 `teacher_score` 的对数概率
4. 参考模型对 `model_score` 的对数概率

因此，训练时必须满足下面这个条件：

对于同一条 `mfcc`，模型不能只输出最终 argmax 分数，还必须输出 logits 或 softmax 前的分布信息。

这一点和你当前代码的关系是：

- `sft` 训练与验证代码里，模型前向本身已经能拿到原始输出张量，因此可以继续扩展成 DPO 所需的 `log p(y|x)` 计算
- `Client` 目前写入 `predictions.xlsx` 的是 argmax 后的离散分数，只适合展示或回传，不足以直接拿来训练 DPO

所以，后续真正实现 DPO Trainer 时，不应从 `predictions.xlsx` 反推 logits，而应直接在训练代码里调用参考模型和策略模型做前向计算。

---

### 7.1 单条评分向量的对数概率计算方式

假设模型输出是 10 个维度、每个维度 5 个类别的 logits，形状为：

```text
[10, 5]
```

则一个具体评分向量，例如：

```json
[4, 3, 5, 2, 4, 3, 4, 2, 5, 3]
```

可以先转成类别索引：

```json
[3, 2, 4, 1, 3, 2, 3, 1, 4, 2]
```

然后该评分向量在模型下的总对数概率定义为：

```text
log p(y | x) = sum_i log p(y_i | x)
```

这里的 `i` 表示第 `i` 个评分维度。

也就是说：

1. 对每个维度做 `log_softmax`
2. 取出目标分数对应的对数概率
3. 对 10 个维度求和

这个定义非常重要，因为 DPO 的训练目标不是比较最终整数分数本身，而是比较模型对两个评分向量的偏好程度。

---

## 8. DPO 训练目标在本项目中的映射

标准 DPO 的核心思想是：

对于同一个输入 `x`，让策略模型更偏好 `chosen`，而不是 `rejected`，同时参考模型作为一个基准约束。

映射到你的项目中就是：

- 输入 `x`：音频对应的 MFCC 特征
- `chosen`：老师打分 `teacher_score`
- `rejected`：参考模型打分 `model_score`

因此，本项目的 DPO 样本对就是：

```text
(x_mfcc, teacher_score, model_score)
```

对应的 DPO 损失可以写成：

```text
L_dpo = -log sigmoid(beta * ((log pi_theta(y_chosen | x) - log pi_ref(y_chosen | x))
                           - (log pi_theta(y_rejected | x) - log pi_ref(y_rejected | x)))))
```

在你的场景中，代入后就是：

```text
L_dpo = -log sigmoid(beta * ((log pi_theta(teacher_score | mfcc) - log pi_ref(teacher_score | mfcc))
                           - (log pi_theta(model_score | mfcc) - log pi_ref(model_score | mfcc))))
```

这里：

- `pi_ref` 是冻结的参考模型
- `pi_theta` 是当前训练中的策略模型
- `beta` 是 DPO 超参数，用来控制偏好强度

---

## 9. 完整训练流程

### 阶段一：训练参考模型

1. 使用原来的 6000 条数据训练基础模型
2. 保存最优 checkpoint
3. 将该 checkpoint 作为 `reference_model`
4. 后续 DPO 阶段冻结该模型参数

输出产物：

- `reference_model.pth`

---

### 阶段二：构造 DPO 数据

对每条带老师评分的音频样本，执行下面流程：

1. 读取原始音频
2. 提取 MFCC
3. 用参考模型推理，得到 `model_score`
4. 读取老师标注，得到 `teacher_score`
5. 设置 `teacher_preferred_flag = 1`
6. 保存为一条 DPO 样本

其中第 4 步在你当前仓库里有两种可能来源：

- 离线标签文件：即 `Label/*.xlsx`
- 前后端链路下载到本地后的老师评分文件，需先解析并映射到固定的 10 维顺序

如果老师评分最初来自 `Client` 侧下载文件，那么建议单独增加一个“老师评分标准化脚本”，先把原始下载结果转换成统一的 10 维 `teacher_score`，再进入 DPO 数据构造。

输出产物：

- `train_dpo.jsonl`
- `val_dpo.jsonl`

---

### 阶段三：初始化策略模型

1. 创建一个新模型，结构与参考模型相同
2. 加载参考模型参数
3. 将其作为 `policy_model`

这样做的好处是：

1. 训练稳定
2. 不会一开始偏移太大
3. 更容易体现“在旧模型基础上偏好老师评分”

---

### 阶段四：执行 DPO 训练

每个 batch 的训练步骤如下：

1. 读取 batch 内的 MFCC 特征
2. 读取 `teacher_score`
3. 读取 `model_score`
4. 用策略模型计算：
   - `log pi_theta(teacher_score | mfcc)`
   - `log pi_theta(model_score | mfcc)`
5. 用参考模型计算：
   - `log pi_ref(teacher_score | mfcc)`
   - `log pi_ref(model_score | mfcc)`
6. 根据 DPO 公式计算 loss
7. 只更新策略模型参数
8. 参考模型始终冻结

输出产物：

- `policy_model_dpo_best.pth`

---

### 阶段五：验证与对比

训练完成后，需要至少做三类对比：

1. 参考模型与老师评分的一致性
2. DPO 后策略模型与老师评分的一致性
3. DPO 后策略模型相对于参考模型是否更接近老师评分

建议验证指标：

1. 每个评分维度的准确率
2. 每个评分维度与老师分数的平均绝对误差
3. 全部 10 个维度上的平均绝对误差
4. 新模型相对于参考模型的改进幅度

你真正想看到的结论应该是：

`DPO 后模型在同一批带老师评分的数据上，比参考模型更接近老师评分。`

---

## 10. 推荐的数据目录结构

建议项目目录按下面方式组织：

```text
project/
├── data/
│   ├── raw/
│   │   ├── audio/
│   │   └── labels_teacher/
│   ├── processed/
│   │   ├── mfcc/
│   │   └── dpo/
│   │       ├── train_dpo.jsonl
│   │       ├── val_dpo.jsonl
│   │       └── test_dpo.jsonl
│   └── external/
│       └── client_downloads/
├── models/
│   ├── reference/
│   └── dpo/
├── logs/
├── configs/
│   ├── mfcc.yaml
│   ├── train_reference.yaml
│   └── train_dpo.yaml
├── src/
│   ├── data/
│   ├── feature/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── scripts/
│   ├── build_dpo_dataset.py
│   ├── train_reference.py
│   ├── train_dpo.py
│   └── export_predictions.py
├── tests/
└── README.md
```

这个目录结构是按你提供的 `Python代码与项目规范.md` 做过对齐后的版本，比当前仓库里“脚本散放在根目录”的形式更适合作为新 DPO 框架的落地目标。

---

## 11. 推荐的最小字段集合

如果你想先用最小代价把整个项目跑通，那么每条 DPO 样本最少需要下面这些字段：

```json
{
  "sample_id": "000001",
  "audio_path": "data/audio/000001.wav",
  "mfcc_path": "data/mfcc/000001_MFCC.xlsx",
  "voice_part": "sopran",
  "model_score": [3, 2, 4, 3, 3, 2, 4, 2, 3, 2],
  "teacher_score": [4, 3, 4, 4, 4, 3, 4, 2, 3, 2],
  "teacher_preferred_flag": 1
}
```

这是你当前项目可以直接采用的核心数据格式。

---

## 12. 建议额外保存的字段

虽然不是最小必需，但为了后续调试和实验，建议额外保存下面这些字段：

```json
{
  "sample_rate": 44100,
  "duration": 3.84,
  "reference_model_version": "ref_v1",
  "teacher_id": "teacher_01",
  "created_at": "2026-03-22T10:00:00",
  "task_type": "singing_score"
}
```

这些字段的作用是：

1. 方便排查不同版本模型的训练效果
2. 方便后续做老师间差异分析
3. 方便做数据切分和数据回溯

---

## 13. 项目中的关键约束

为了让整个 DPO 项目顺利落地，必须遵守以下约束：

1. 参考模型训练完后必须冻结
2. 策略模型和参考模型必须使用完全一致的 MFCC 提取参数
3. 评分向量的维度和顺序必须固定
   这里固定的是“选出的 10 个评分维度”的维度和顺序；如果原始标签是 40 维，必须先完成同一套筛选映射，再进入训练
4. 训练 DPO 时必须使用 logits 计算对数概率，不能只用 argmax 分数
5. `teacher_preferred_flag` 在当前项目里应默认为 1
6. 如果 `model_score` 与 `teacher_score` 完全相同，该样本对 DPO 价值较低，可以选择跳过
7. 训练集和验证集必须严格分开，避免误判 DPO 效果
8. 如果当前仍沿用多声部独立权重方案，则 `voice_part` 必须保留，并在数据构造、参考模型加载和策略模型训练时保持一致
9. `Client` 导出的 `predictions.xlsx` 只能作为展示或中间结果，不能替代 DPO 训练阶段所需的 logits

---

## 14. 最终结论

基于你当前的设想，最清晰、最直接、最适合落地的方案就是：

1. 用原来的 6000 条数据训练一个参考模型
2. 对每条新音频提取 MFCC 特征
3. 对同一条音频同时收集：
   - 参考模型打分 `model_score`
   - 老师打分 `teacher_score`
4. 令：
   - `teacher_score` 为 chosen
   - `model_score` 为 rejected
5. 使用 DPO 目标训练新的策略模型
6. 让新模型在相同音频输入下，相对于参考模型，更偏好老师评分

因此，你当前最核心的数据格式可以直接定为：

```json
{
  "sample_id": "000001",
  "audio_path": "data/audio/000001.wav",
  "mfcc_path": "data/mfcc/000001_MFCC.xlsx",
  "voice_part": "sopran",
  "model_score": [3, 2, 4, 3, 3, 2, 4, 2, 3, 2],
  "teacher_score": [4, 3, 4, 4, 4, 3, 4, 2, 3, 2],
  "teacher_preferred_flag": 1
}
```

这份格式已经足够作为你整个项目的数据主线。

如果后面你继续往下做，下一步最自然的工作顺序就是：

1. 先固定 MFCC 配置
2. 先训练参考模型
3. 再构造 DPO 偏好数据
4. 最后实现 DPO Trainer
