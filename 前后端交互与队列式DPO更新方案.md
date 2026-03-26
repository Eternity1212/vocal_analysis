# 前后端交互与队列式 DPO 更新方案

## 1. 文档目的

这份文档用于明确当前项目中“前端上传数据、后端切分音频、提取 MFCC、生成单条 JSON、进入队列、达到阈值后触发一次 DPO 更新、更新完成后切换新模型”的完整设计。

这份设计严格围绕你当前的想法展开：

1. 前端上传 `.wav` 音频文件
2. 前端同时上传一份 `json`
3. `json` 中包含：
   - 模型打分
   - 老师打分
   - 偏好 flag
4. 后端对音频进行切分
5. 对每个切片做 MFCC 特征提取
6. 将同一条音频的多个 MFCC 结果拼接在一起
7. 用一个索引记录每个切片在拼接结果中的位置
8. 每条样本单独保存一个 `json` 文件
9. 这些 `json` 文件以队列形式存储
10. 当队列数量达到设定阈值时，例如 10 条，触发一次 DPO 更新
11. DPO 更新后产生新的模型权重
12. 如果新模型通过检查，则设置模型切换标志位，前端通过轮询感知并完成切换

---

## 2. 整体流程

整体流程可以概括为：

`前端上传 wav + json -> 后端切分音频 -> 提取 MFCC -> 拼接特征并建立索引 -> 生成单条样本 json -> 放入队列 -> 达到阈值 -> 触发一次 DPO 更新 -> 生成 new_model -> 更新 change_flag -> 前端轮询感知模型变化`

当前文档包含两条可并行存在的更新路线：

1. 实时队列更新路线
   - 前端持续上传单条样本
   - 后端自动入队
   - 达到阈值后自动触发一次 DPO 更新

2. 离线手动更新路线
   - 人工手动下载和汇总所有评分文件
   - 人工发起统一处理
   - 人工触发一次批量 DPO 训练
   - 训练完成后再决定是否切换到新模型

---

## 3. 流程图

为了更清晰展示整个系统，本节将流程拆成 4 张图：

1. 总览流程图
2. 实时队列更新分支图
3. 离线手动更新分支图
4. 模型切换与前端轮询图

---

## 3.1 总览流程图

这张图用于展示整个系统的主流程，以及“实时自动更新”和“离线手动更新”两条分支。

图片文件：`pic/01_overview_flow.svg`

![总览流程图](pic/01_overview_flow.svg)

---

## 3.2 实时队列更新分支图

这张图只展示“实时自动更新”分支，重点是样本如何入队、达到阈值后如何自动训练。

图片文件：`pic/02_realtime_queue_flow.svg`

![实时队列更新分支图](pic/02_realtime_queue_flow.svg)

---

## 3.3 离线手动更新分支图

这张图只展示“人工汇总后统一训练”的分支，重点是人工下载、人工清洗、人工触发训练。

图片文件：`pic/03_offline_manual_flow.svg`

![离线手动更新分支图](pic/03_offline_manual_flow.svg)

---

## 3.4 模型切换与前端轮询图

这张图只展示“训练完成以后如何切换模型”，重点是 `change_flag` 和前端轮询逻辑。

图片文件：`pic/04_model_switch_flow.svg`

![模型切换与前端轮询图](pic/04_model_switch_flow.svg)

---

## 4. 前端上传内容设计

## 4.1 前端每次上传的内容

前端每次上传一条完整样本，包含两部分：

1. 一个 `.wav` 音频文件
2. 一个与该音频对应的 `json`

建议上传方式使用：

- `multipart/form-data`

这样可以在一个请求中同时传文件和结构化信息。

---

## 4.2 前端上传接口

建议设计一个统一接口：

```text
POST /api/v1/upload-sample
```

请求包含两个字段：

1. `audio_file`
2. `meta_json`

其中：

- `audio_file` 是前端上传的 `.wav`
- `meta_json` 是前端生成的一条 JSON 信息

---

## 4.3 前端上传的 JSON 结构

当前版本中，前端上传的 JSON 应至少包含以下字段：

```json
{
  "sample_id": "20260323_000001",
  "user_id": "user_001",
  "model_score": [3, 2, 4, 3, 3, 2, 4, 2, 3, 2],
  "teacher_score": [4, 3, 4, 4, 4, 3, 4, 2, 3, 2],
  "teacher_preferred_flag": 1,
  "client_time": "2026-03-23T10:00:00",
  "score_dim": 10
}
```

字段说明如下：

1. `sample_id`
   - 当前样本唯一标识

2. `user_id`
   - 当前音频对应的用户或任务来源

3. `model_score`
   - 当前模型对这条音频的十类评分结果

4. `teacher_score`
   - 老师对同一条音频的十类评分结果

5. `teacher_preferred_flag`
   - 偏好标记
   - 当前设计固定为：
     - `1` 表示老师分数为 preferred
     - `0` 表示当前样本不参与本次偏好更新或不切换模型

6. `client_time`
   - 前端生成该条数据的时间

7. `score_dim`
   - 评分维度数量，当前固定为 10

---

## 5. 后端接收后的处理逻辑

后端收到前端上传的一条样本后，需要执行以下步骤：

1. 保存原始 `.wav`
2. 保存原始 `json`
3. 对音频进行切分
4. 对每个切片提取 MFCC
5. 将该条音频的全部 MFCC 特征拼接在一起
6. 建立一个 `segment_index`
7. 生成新的“训练样本 JSON”
8. 将该样本 JSON 放入队列

---

## 6. 音频切分设计

## 6.1 切分目标

由于前端上传的音频长度可能不一致，因此后端不直接把整段音频作为单一输入保存，而是先切分，再提取特征。

这样做有三个作用：

1. 统一特征处理长度
2. 方便后续批量训练
3. 方便将同一条音频拆成多个片段做聚合

---

## 6.2 切分方式

当前设计建议采用固定窗口切分。

例如：

- 切片长度：`2s`
- 步长：`2s`

也就是不重叠切分。

举例：

如果原音频长度为 `8.4s`，则可以切成：

1. `0s ~ 2s`
2. `2s ~ 4s`
3. `4s ~ 6s`
4. `6s ~ 8s`

末尾不足部分可以选择：

1. 丢弃
2. 补零

当前方案建议：

- 末尾不足一个完整窗口时，补零到完整窗口

这样每个切片都能得到统一形状的 MFCC。

---

## 7. MFCC 特征提取与拼接设计

## 7.1 单个切片的 MFCC 输出

每个音频切片提取一个固定大小的 MFCC 特征。

建议统一输出形状为：

```text
[128, 128]
```

即：

1. 128 个 MFCC 维度
2. 128 个时间帧

---

## 7.2 同一条音频的 MFCC 拼接方式

假设一条音频被切成 `N` 个片段，则每个片段都会产生一个 `MFCC[i]`。

当前设计中，“拼接到一起”建议定义为：

```text
mfcc_concat.shape = [N, 128, 128]
```

也就是说，不是把所有值直接拉平成一个长向量，而是把多个切片的 MFCC 按第一维堆叠起来。

例如：

```text
第 0 段 -> mfcc_concat[0]
第 1 段 -> mfcc_concat[1]
第 2 段 -> mfcc_concat[2]
...
第 N-1 段 -> mfcc_concat[N-1]
```

这种方式最适合后续做：

1. 分段推理
2. 分段聚合
3. 批量 DPO 训练

---

## 7.3 segment_index 的定义

为了表示“拼接后的每个位置对应哪一个音频片段”，必须保存一个索引。

建议每条样本都保存一个 `segment_index` 字段，格式如下：

```json
[
  {"segment_id": 0, "start_ms": 0, "end_ms": 2000, "concat_pos": 0},
  {"segment_id": 1, "start_ms": 2000, "end_ms": 4000, "concat_pos": 1},
  {"segment_id": 2, "start_ms": 4000, "end_ms": 6000, "concat_pos": 2}
]
```

字段含义如下：

1. `segment_id`
   - 切片编号

2. `start_ms`
   - 原音频中该切片的起始时间

3. `end_ms`
   - 原音频中该切片的结束时间

4. `concat_pos`
   - 该切片在 `mfcc_concat` 中的位置

这样后续无论是训练还是分析，都能准确知道：

- 某一段 MFCC 来自原始音频的哪个时间区间
- 在拼接张量中对应哪个位置

---

## 8. 单条样本文件设计

## 8.1 每条样本单独保存一个 JSON

当前设计要求每次前端返回的结果单独存在一个 JSON 文件，这个思路可以直接保留。

因此，后端处理完一条样本后，会生成：

1. 原始音频文件
2. 原始前端 JSON
3. 处理后的 MFCC 拼接文件
4. 处理后的训练样本 JSON

---

## 8.2 推荐的单条训练样本 JSON 格式

```json
{
  "sample_id": "20260323_000001",
  "audio_path": "runtime/audio/20260323_000001.wav",
  "raw_front_json_path": "runtime/raw_json/20260323_000001.json",
  "mfcc_concat_path": "runtime/mfcc/20260323_000001_mfcc_concat.npy",
  "segment_count": 4,
  "segment_index": [
    {"segment_id": 0, "start_ms": 0, "end_ms": 2000, "concat_pos": 0},
    {"segment_id": 1, "start_ms": 2000, "end_ms": 4000, "concat_pos": 1},
    {"segment_id": 2, "start_ms": 4000, "end_ms": 6000, "concat_pos": 2},
    {"segment_id": 3, "start_ms": 6000, "end_ms": 8000, "concat_pos": 3}
  ],
  "model_score": [3, 2, 4, 3, 3, 2, 4, 2, 3, 2],
  "teacher_score": [4, 3, 4, 4, 4, 3, 4, 2, 3, 2],
  "teacher_preferred_flag": 1,
  "status": "pending",
  "create_time": "2026-03-23T10:00:00"
}
```

这个 JSON 就是后续进入队列的核心对象。

---

## 9. 队列设计

## 9.1 设计目标

队列的作用是：

1. 暂存单条样本 JSON
2. 等待样本累计到阈值
3. 达到阈值后抽取一个批次
4. 用这个批次做一次 DPO 更新

当前版本按你的想法，使用“样本条数阈值”触发更新。

示例阈值：

```text
threshold = 10
```

也就是说：

- 当队列中累计到 10 条 JSON 样本时，触发一次 DPO 更新

---

## 9.2 队列推荐实现方式

为了贴合你“每条结果单独一个 JSON 文件”的设计，第一版建议直接使用“文件夹队列”。

目录如下：

```text
runtime/
├── audio/
├── raw_json/
├── mfcc/
├── queue/
│   ├── pending/
│   ├── processing/
│   ├── done/
│   └── failed/
├── batches/
├── checkpoints/
└── state/
```

其中：

1. `queue/pending/`
   - 等待参与训练的样本 JSON

2. `queue/processing/`
   - 正在被当前 DPO 批次使用的样本 JSON

3. `queue/done/`
   - 已完成训练的样本 JSON

4. `queue/failed/`
   - 处理或训练失败的样本 JSON

---

## 9.3 入队规则

每当后端处理完一条新样本：

1. 生成训练样本 JSON
2. 将该 JSON 放入 `queue/pending/`
3. 状态设为 `pending`

队列可以按文件创建时间排序，默认采用先进先出，也就是 FIFO。

---

## 9.4 触发训练规则

当前版本的规则非常明确：

1. 后台服务定期检查 `queue/pending/`
2. 如果 `pending` 中的样本数 `< 10`
   - 不触发训练
3. 如果 `pending` 中的样本数 `>= 10`
   - 按时间顺序取前 10 条
   - 放入一个训练批次
   - 触发一次 DPO 更新

举例：

如果当前 `pending` 目录里有 13 条样本，那么：

1. 先抽取前 10 条
2. 组成 `batch_000001`
3. 执行一次 DPO 更新
4. 剩余 3 条继续留在 `pending`

---

## 10. 训练批次设计

## 10.1 批次文件

每次从队列里抽出 10 条样本后，建议生成一个批次描述文件，例如：

```text
runtime/batches/batch_000001.json
```

示例内容如下：

```json
{
  "batch_id": "batch_000001",
  "sample_count": 10,
  "sample_json_list": [
    "runtime/queue/processing/20260323_000001.json",
    "runtime/queue/processing/20260323_000002.json",
    "runtime/queue/processing/20260323_000003.json"
  ],
  "status": "processing",
  "create_time": "2026-03-23T10:30:00"
}
```

这个批次文件的作用是：

1. 标记本次训练到底用了哪 10 条数据
2. 方便追踪某次 DPO 更新对应的数据来源
3. 方便后续排查训练问题

---

## 10.2 批次训练输入

每次 DPO 更新时，训练器读取这 10 条样本 JSON，并进一步读取每条样本中的：

1. `mfcc_concat_path`
2. `segment_index`
3. `model_score`
4. `teacher_score`
5. `teacher_preferred_flag`

对于每条样本：

1. 读取 `mfcc_concat`
2. 按切片逐段送入模型
3. 得到每一段的 logits
4. 将所有段的 logits 做聚合
5. 形成该条样本的最终样本级 logits
6. 用样本级 logits 计算该条样本的 DPO loss

---

## 11. 分段聚合方式

由于一条音频现在被切成多个片段，所以模型最终用于 DPO 的分数不再直接来自单一切片，而来自整条样本的聚合结果。

当前推荐最简单的聚合方式是：

1. 每个切片独立推理
2. 得到每个切片的 logits
3. 对所有切片的 logits 取平均
4. 平均后的结果作为该条音频的最终 logits

即：

```text
sample_logits = mean(segment_logits_1, segment_logits_2, ..., segment_logits_N)
```

这样做的优点是：

1. 实现简单
2. 与当前模型结构兼容
3. 适合第一版项目落地

---

## 12. 一次 DPO 更新的定义

当前版本中，“一次强化学习更新”在工程上定义为：

1. 从 `queue/pending/` 中取出 10 条样本
2. 构造一次 DPO 训练批次
3. 基于这 10 条数据执行一次训练过程
4. 得到一次新的模型权重

这一次训练输出的权重文件建议命名为：

```text
policy_model_dpo_best.pth
```

或者更明确一些：

```text
policy_model_dpo_iter_000001.pth
```

如果你希望保留每一次更新后的权重，建议使用第二种命名。

---

## 13. 新模型生成与切换逻辑

## 13.1 new_model 的定义

每次 DPO 更新完成后，都会得到一个新的模型权重。

这个权重在当前文档中记为：

```text
new_model
```

它本质上就是：

```text
policy_model_dpo_best.pth
```

或者：

```text
policy_model_dpo_iter_xxxxxx.pth
```

也就是“本轮训练完成后的新权重”。

---

## 13.2 change_flag 的作用

根据你的图，DPO 完成以后，需要通过一个 `change_flag` 来控制前端是否感知模型切换。

当前建议将 `change_flag` 定义为：

1. `0`
   - 前端不变
   - 当前仍然使用旧模型

2. `1`
   - 表示新的模型已经训练完成并通过检查
   - 前端或服务网关应切换到新的模型版本

---

## 13.3 什么时候把 change_flag 置为 1

不建议只要训练完成就立刻置为 `1`。

更合理的规则是：

1. 完成一次 DPO 更新
2. 得到 `new_model`
3. 对 `new_model` 做基本检查
4. 如果检查通过，则：
   - 更新 `active_model`
   - 设置 `change_flag = 1`
5. 如果检查不通过，则：
   - 保持旧模型
   - 设置 `change_flag = 0`

这样可以保证前端不会因为一次失败训练而切换到坏模型。

---

## 13.4 模型状态文件

建议用一个状态文件记录当前模型情况：

```text
runtime/state/model_state.json
```

示例内容：

```json
{
  "active_model_version": "policy_model_dpo_iter_000005",
  "active_model_path": "runtime/checkpoints/policy_model_dpo_iter_000005.pth",
  "change_flag": 1,
  "last_batch_id": "batch_000005",
  "update_time": "2026-03-23T11:00:00"
}
```

---

## 14. 前端轮询设计

## 14.1 轮询目标

前端需要知道：

1. 当前模型是否发生了更新
2. 是否需要切换到新的模型版本

因此建议前端定期调用一个轮询接口：

```text
GET /api/v1/model/update-status
```

---

## 14.2 返回格式

```json
{
  "change_flag": 1,
  "active_model_version": "policy_model_dpo_iter_000005",
  "update_time": "2026-03-23T11:00:00"
}
```

规则如下：

1. 如果 `change_flag = 0`
   - 前端不做任何动作

2. 如果 `change_flag = 1`
   - 前端感知到有新模型
   - 前端触发模型版本刷新流程

---

## 14.3 前端收到 change_flag = 1 之后的动作

当前方案建议前端这样处理：

1. 检测到 `change_flag = 1`
2. 前端短暂进入“模型切换中”状态
3. 等待服务端完成新模型加载
4. 新模型加载成功后，前端恢复正常请求
5. 服务端将 `change_flag` 重置为 `0`

如果你希望逻辑更接近你图中的“锁定后再放行”，可以这样理解：

1. `change_flag = 1`
   - 进入切换流程
2. 服务端切换到 `new_model`
3. 切换成功后放行
4. 前端继续正常使用

---

## 15. 推荐的目录结构

```text
runtime/
├── audio/
│   ├── 20260323_000001.wav
│   └── 20260323_000002.wav
├── raw_json/
│   ├── 20260323_000001.json
│   └── 20260323_000002.json
├── mfcc/
│   ├── 20260323_000001_mfcc_concat.npy
│   └── 20260323_000002_mfcc_concat.npy
├── queue/
│   ├── pending/
│   ├── processing/
│   ├── done/
│   └── failed/
├── batches/
│   ├── batch_000001.json
│   └── batch_000002.json
├── checkpoints/
│   ├── reference_model.pth
│   ├── policy_model_dpo_iter_000001.pth
│   └── policy_model_dpo_best.pth
└── state/
    └── model_state.json
```

---

## 16. 核心字段总结

为了保证整个流程稳定，当前版本最重要的字段如下：

### 16.1 前端原始 JSON 核心字段

```json
{
  "sample_id": "20260323_000001",
  "model_score": [3, 2, 4, 3, 3, 2, 4, 2, 3, 2],
  "teacher_score": [4, 3, 4, 4, 4, 3, 4, 2, 3, 2],
  "teacher_preferred_flag": 1
}
```

### 16.2 后端训练样本 JSON 核心字段

```json
{
  "sample_id": "20260323_000001",
  "audio_path": "runtime/audio/20260323_000001.wav",
  "mfcc_concat_path": "runtime/mfcc/20260323_000001_mfcc_concat.npy",
  "segment_index": [
    {"segment_id": 0, "start_ms": 0, "end_ms": 2000, "concat_pos": 0},
    {"segment_id": 1, "start_ms": 2000, "end_ms": 4000, "concat_pos": 1}
  ],
  "model_score": [3, 2, 4, 3, 3, 2, 4, 2, 3, 2],
  "teacher_score": [4, 3, 4, 4, 4, 3, 4, 2, 3, 2],
  "teacher_preferred_flag": 1,
  "status": "pending"
}
```

### 16.3 模型状态文件核心字段

```json
{
  "active_model_version": "policy_model_dpo_iter_000005",
  "active_model_path": "runtime/checkpoints/policy_model_dpo_iter_000005.pth",
  "change_flag": 1
}
```

---

## 17. 一次完整示例

这里用“累计 10 条数据触发一次更新”举一个完整例子。

### 步骤 1

前端上传：

1. `20260323_000001.wav`
2. `20260323_000001.json`

其中 JSON 中包含：

- `model_score`
- `teacher_score`
- `teacher_preferred_flag`

---

### 步骤 2

后端收到后执行：

1. 保存原始音频
2. 切分音频
3. 对每段计算 MFCC
4. 生成 `20260323_000001_mfcc_concat.npy`
5. 生成该样本的训练 JSON
6. 将该 JSON 放入 `queue/pending/`

---

### 步骤 3

当 `queue/pending/` 中累计到第 10 条样本时：

1. 抽取这 10 条 JSON
2. 移动到 `queue/processing/`
3. 生成 `batch_000001.json`
4. 启动一次 DPO 训练

---

### 步骤 4

DPO 训练完成后：

1. 生成 `policy_model_dpo_iter_000001.pth`
2. 检查该模型是否可用

如果可用：

1. 更新 `active_model_version`
2. 更新 `active_model_path`
3. 设置 `change_flag = 1`

如果不可用：

1. 保持旧模型
2. `change_flag = 0`

---

### 步骤 5

前端定期轮询：

```text
GET /api/v1/model/update-status
```

如果发现：

```json
{
  "change_flag": 1
}
```

则表示：

- 有新模型已经准备好
- 前端可以进入切换流程

完成切换后，后端将 `change_flag` 重置为 `0`。

---

## 18. 当前版本的最终工程定义

基于你当前的想法，这个系统的工程定义可以直接固定为：

1. 前端每次上传一条 `wav + json`
2. 每条样本单独生成一个训练样本 JSON
3. 后端切分音频并提取 MFCC
4. 同一条音频的多个 MFCC 结果堆叠为一个 `mfcc_concat`
5. 使用 `segment_index` 记录每段特征的位置
6. 所有训练样本 JSON 进入 `queue/pending`
7. 当样本数达到 `10` 时，触发一次 DPO 更新
8. 每次更新都生成一个新的 `new_model`
9. 新模型通过检查后，更新 `active_model`
10. 用 `change_flag` 通知前端模型已变更
11. 前端通过轮询感知并完成模型切换

---

## 19. 最终结论

这套设计的核心不是“前端直接参与训练”，而是：

1. 前端负责持续上传 `wav + json`
2. 后端负责将每条样本标准化为统一训练格式
3. 后端用文件队列管理样本
4. 达到阈值后进行一次批量 DPO 更新
5. 更新后的模型通过 `change_flag` 和轮询机制切换到新版本

因此，你这个项目当前最关键的三个核心对象就是：

1. 单条样本 JSON
2. 队列目录 `queue/pending`
3. 模型状态文件 `model_state.json`

这三个对象一旦固定，后面的前后端交互、训练触发和模型更新流程就都能串起来。

---

## 20. 第二条路线：离线手动汇总打分文件并训练

除了上面的“实时队列触发更新”路线以外，还需要保留一条“离线手动更新”路线。

这条路线的核心思想是：

1. 前端先正常产生音频和打分文件
2. 但后端不要求实时触发训练
3. 由人工在某个时间点统一下载、汇总、处理这些打分文件
4. 再一次性构造训练数据并执行 DPO 训练
5. 训练完成后人工决定是否切换模型

这条路线适合以下场景：

1. 当前阶段不希望系统自动更新模型
2. 需要人工筛查老师打分质量
3. 需要先手动清洗数据再训练
4. 需要把一批数据攒够后统一训练
5. 需要避免线上服务与训练流程强耦合

---

## 20.1 离线手动更新路线的整体流程

离线手动更新路线可以概括为：

`前端产生 wav + json -> 人工统一下载全部样本 -> 后端离线切分音频 -> 离线提取 MFCC -> 生成训练样本 json -> 人工汇总为一批训练数据 -> 手动触发 DPO 训练 -> 生成 new_model -> 人工检查 -> 手动切换 active_model`

与实时路线相比，最大的区别是：

1. 不依赖 `queue/pending` 达到阈值后自动训练
2. 不依赖后台常驻 worker 自动触发训练
3. 训练的开始时间由人工控制
4. 模型切换也可以由人工控制，而不是自动放行

---

## 20.2 离线手动更新路线的数据来源

离线分支中，前端仍然产生和实时路线相同格式的原始数据：

1. `.wav` 音频文件
2. 与音频对应的 `json`

推荐保留和实时路线一致的前端 JSON 结构：

```json
{
  "sample_id": "20260323_000001",
  "user_id": "user_001",
  "model_score": [3, 2, 4, 3, 3, 2, 4, 2, 3, 2],
  "teacher_score": [4, 3, 4, 4, 4, 3, 4, 2, 3, 2],
  "teacher_preferred_flag": 1,
  "client_time": "2026-03-23T10:00:00",
  "score_dim": 10
}
```

区别只在于：

1. 这些文件不立即入实时训练队列
2. 而是先由人工统一收集
3. 后续再批量导入训练流程

---

## 20.3 离线手动更新路线的目录建议

建议在现有 `runtime/` 下面增加一套手动更新目录：

```text
runtime/
├── manual_upload/
│   ├── audio/
│   └── raw_json/
├── manual_processed/
│   ├── mfcc/
│   ├── sample_json/
│   └── batches/
├── checkpoints/
└── state/
```

其中：

1. `manual_upload/audio/`
   - 人工下载回来的原始音频

2. `manual_upload/raw_json/`
   - 人工下载回来的原始评分 JSON

3. `manual_processed/mfcc/`
   - 离线切分和 MFCC 提取之后得到的拼接特征文件

4. `manual_processed/sample_json/`
   - 离线生成的训练样本 JSON

5. `manual_processed/batches/`
   - 人工整理好的批次文件

---

## 20.4 离线手动更新路线的处理步骤

离线路线建议按下面的固定步骤执行。

### 步骤 1：人工下载全部样本

人工从前端、管理后台或存储系统中统一下载：

1. 一批 `.wav`
2. 一批对应的 `json`

下载后放到：

```text
runtime/manual_upload/audio/
runtime/manual_upload/raw_json/
```

---

### 步骤 2：人工执行离线预处理

人工启动一个离线处理脚本，对下载回来的样本统一处理。

离线预处理包含：

1. 读取音频
2. 切分音频
3. 对每个切片提取 MFCC
4. 生成 `mfcc_concat`
5. 生成 `segment_index`
6. 生成训练样本 JSON

这一步完成后，会得到：

1. `manual_processed/mfcc/*.npy`
2. `manual_processed/sample_json/*.json`

---

### 步骤 3：人工筛查和清洗样本

离线路线的一个重要优点是，训练前可以人工筛查样本。

建议人工重点检查：

1. 音频是否损坏
2. JSON 是否缺字段
3. `model_score` 和 `teacher_score` 是否维度一致
4. `teacher_preferred_flag` 是否合理
5. 某些明显错误标注是否需要剔除

如果发现问题，可以：

1. 删除对应样本
2. 手动修正 JSON
3. 将异常样本移到单独目录

---

### 步骤 4：人工构造训练批次

筛查完成后，人工将需要训练的样本汇总成一个批次。

例如生成：

```text
runtime/manual_processed/batches/manual_batch_20260323.json
```

示例内容：

```json
{
  "batch_id": "manual_batch_20260323",
  "sample_count": 86,
  "sample_json_list": [
    "runtime/manual_processed/sample_json/20260323_000001.json",
    "runtime/manual_processed/sample_json/20260323_000002.json",
    "runtime/manual_processed/sample_json/20260323_000003.json"
  ],
  "create_mode": "manual",
  "status": "ready"
}
```

这表示：

1. 这次不是按 10 条自动触发
2. 而是人工指定一整批样本进行训练

---

### 步骤 5：人工触发 DPO 训练

实时路线中，训练由队列阈值触发。

离线路线中，训练应由人工显式触发。

例如可以约定一个命令：

```text
python train_dpo_manual.py --batch-file runtime/manual_processed/batches/manual_batch_20260323.json
```

或者由管理后台触发一个接口：

```text
POST /api/v1/manual-dpo-train
```

请求示例：

```json
{
  "batch_id": "manual_batch_20260323"
}
```

---

## 20.5 离线手动更新路线中的训练输入

手动训练时，训练器读取批次文件中的每条 `sample_json`，并进一步读取：

1. `mfcc_concat_path`
2. `segment_index`
3. `model_score`
4. `teacher_score`
5. `teacher_preferred_flag`

也就是说，训练样本格式和实时路线保持一致，只是“谁来触发训练”不同。

这一点非常重要，因为这样可以保证：

1. 两条路线共用同一套数据格式
2. 两条路线共用同一套 DPO Trainer
3. 只需要切换“数据进入训练器的方式”
4. 不需要维护两套训练逻辑

---

## 20.6 离线手动更新路线中的模型切换

离线路线训练完成后，也会产出一个新的模型权重：

```text
policy_model_dpo_manual_best.pth
```

或者：

```text
policy_model_dpo_manual_20260323.pth
```

训练完成后建议执行以下步骤：

1. 人工检查训练日志
2. 人工检查评估指标
3. 人工确认该模型是否优于当前线上模型
4. 如果确认可用，再手动更新 `active_model`
5. 再决定是否把 `change_flag` 置为 `1`

因此，离线路线中的 `change_flag` 更适合定义为：

1. `0`
   - 暂不切换

2. `1`
   - 人工确认后允许切换到新模型

也就是说，在离线路线中：

- `change_flag` 是“人工审核通过后才生效”的标志

---

## 20.7 离线手动更新路线与实时队列路线的关系

这两条路线应被视为同一个系统里的两种训练入口，而不是两个完全独立的系统。

推荐关系如下：

1. 实时队列路线
   - 用于持续、小批量、自动积累和自动更新

2. 离线手动路线
   - 用于人工汇总、人工清洗、人工控制训练时机

3. 两条路线共享：
   - 相同的前端 JSON 格式
   - 相同的后端样本 JSON 格式
   - 相同的 MFCC 处理逻辑
   - 相同的 DPO 训练器
   - 相同的模型状态文件

4. 两条路线的主要差异：
   - 实时路线按阈值自动触发
   - 离线路线按人工指令触发

---

## 20.8 推荐的最终双分支工程定义

基于当前需求，整个系统建议明确支持两种更新模式。

### 模式 A：实时队列更新

流程如下：

1. 前端上传单条 `wav + json`
2. 后端处理成训练样本 JSON
3. 放入 `queue/pending`
4. 达到阈值后自动触发 DPO 更新
5. 训练成功后自动准备新模型
6. 通过 `change_flag` 通知前端切换

### 模式 B：离线手动更新

流程如下：

1. 前端先持续产生 `wav + json`
2. 人工统一下载所有样本
3. 人工统一做切分、MFCC、样本构造
4. 人工整理成一个训练批次
5. 人工触发 DPO 训练
6. 人工检查训练结果
7. 人工确认后更新 `active_model`
8. 再决定是否设置 `change_flag = 1`

---

## 20.9 最终结论补充

因此，这个项目不应该只被定义成“实时更新系统”，而应该被定义成：

`一个同时支持实时队列更新和离线手动更新的 DPO 训练与模型切换系统`

当前最稳妥的工程做法是：

1. 先把数据格式统一
2. 先把 MFCC 和样本 JSON 处理逻辑统一
3. 再在训练入口层面支持两种模式：
   - 自动队列触发
   - 手动批量触发

这样后续无论你是要走线上自动更新，还是走人工收集后统一训练，都不需要重做数据结构和训练主流程。
