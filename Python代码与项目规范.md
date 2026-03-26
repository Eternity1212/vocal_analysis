# 机器学习和深度学习Python代码与项目规范

## 2. 代码风格与规范

以下的大部分规范可使用 Pycharm IDE 的“重新设置代码格式”功能自动应用，或使用 Black 和 isort 等工具自动格式化和排序代码。

### 2.1 PEP 8 遵守

- 尽可能遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 编码风格。
  - 缩进使用4个空格，不使用制表符。
  - 每行代码不超过79个字符，文档字符串或注释不超过72个字符。
    ——实际上考虑到一些较长的URL或表达式，该限制可以适当放宽，如88个字符（Black默认设置）。
  - 函数和类之间用两个空行分隔，方法之间用一个空行分隔。
  - 文件末尾保留一个空行。
  - 使用空格围绕运算符和逗号，但不要在函数参数列表、索引或切片中使用空格。
    - 正确：`x = 1`, `func(a, b)`, `my_list[1:5]`
    - 错误：`x=1`, `func( a, b )`, `my_list[1 : 5]`
  - 避免在行尾添加多余的空格。
  - 使用反斜杠（`\`）进行行连接时，确保后面没有任何字符（包括空格）。
  - 对于长表达式，建议使用括号进行隐式行连接，而不是反斜杠。
    - 推荐：
      ```python
      result = (
          some_function(arg1, arg2)
          + another_function(arg3, arg4)
          - yet_another_function(arg5)
      )
      ```
    - 不推荐：
      ```python
      result = some_function(arg1, arg2) + \
               another_function(arg3, arg4) - \
               yet_another_function(arg5)
      ```
  - 注释符号前面保留两个空格，后面保留一个空格。

### 2.2 命名约定

- **模块/文件名**：小写字母，单词间用下划线分隔。例如：`data_loader.py`, `model_utils.py`。
- **变量名**：小写字母，单词间用下划线分隔（snake_case）。例如：`batch_size`, `learning_rate`。
- **常量名**：大写字母，单词间用下划线分隔。例如：`MAX_EPOCHS`, `DEFAULT_SEED`。
- **函数名**：小写字母，动词开头，snake_case。例如：`train_model()`, `load_data()`。
- **类名**：驼峰命名法（CamelCase），首字母大写。例如：`DataLoader`, `ConvBlock`。
- **私有成员**：以单下划线开头，表示内部使用。例如：`_internal_helper()`。

### 2.3 类型提示

使用Python类型注解（Type Hints）提高代码可读性和IDE支持。对函数参数和返回值添加类型注解。
- 示例：
  ```python
  from typing import List, Tuple, Optional
  
  def preprocess_data(
      data: List[float],
      normalize: bool = True,  # 冒号右边和等号两边保持一个空格。
      max_len: Optional[int] = None  # 如无类型注解，则参数与默认值之间不加空格。
  ) -> Tuple[np.ndarray, np.ndarray]:
      ...
  ```

### 2.5 文档字符串 Docstring

- 使用 **Google风格** 或 **NumPy风格** 的文档字符串。这里推荐Google风格。
- 必须包含：功能描述、参数、返回值、可能引发的异常。
- 示例（Google风格）：
  ```python
  def train_step(model, batch, optimizer):
      """执行单个训练步骤。
  
      该函数接受一个模型、一个包含输入和标签的批次，以及一个优化器实例，执行前向传播、计算损失、反向传播和参数更新。
  
      Args:
          model: 待训练的PyTorch模型。
          batch: 包含输入和标签的元组 (inputs, labels)。
          optimizer: 优化器实例。
  
      Returns:
          float: 当前步骤的损失值。
  
      Raises:
          ValueError: 如果batch格式不正确。
      """
  ```

## 3. 项目结构

### 3.1 典型目录布局

采用类似 Cookiecutter Data Science 的结构，并根据深度学习项目特点调整：

```
project/
├── .gitignore
├── README.md
├── LICENSE
├── pyproject.toml           # 项目元数据、构建工具配置
├── requirements.txt         # 依赖包
├── environment.yml          # Conda环境文件（可选）
├── setup.py                 # 可安装本地包（可选）
├── configs/                 # 配置文件（YAML/JSON）
│   ├── data.yaml
│   ├── training.yaml
│   └── model.yaml
├── data/                    # 数据目录（通常被.gitignore忽略）
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/               # 探索性分析notebook
│   ├── 01-eda.ipynb
│   └── 02-model-prototype.ipynb
├── src/                     # 源代码（可安装的包）
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── layers.py
│   │   └── architecture.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/                   # 单元测试
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_models.py
│   └── test_training.py
├── scripts/                 # 可执行脚本（训练、评估、部署）
│   ├── train.py
│   ├── evaluate.py
│   └── serve.py
├── outputs/                 # 训练日志、评估结果等输出（通常被.gitignore忽略）
│   └── logs/
├── models/                  # 保存训练好的模型（通常被.gitignore忽略）
│   └── experiments/         # 按实验组织
├── reports/                 # 报告、图表、指标
│   └── figures/
└── docker/                  # Dockerfile和相关资源
    └── Dockerfile
```

### 3.2 模块化设计

- 将代码按功能分离到不同模块（数据、模型、训练、评估、工具）。
- 避免在notebook中写大量不可复用的代码，关键功能迁移到`src/`中。
- 每个模块应有清晰的接口，便于测试和复用。例如，`src/data/dataset.py`中定义数据加载器类，`src/models/architecture.py`中定义模型结构。
- 使用`__init__.py`文件组织模块导出，简化导入路径。例如，在`src/data/__init__.py`中：
  ```python
  from .dataset import create_dataloader
  from .preprocessing import preprocess_data
  
  __all__ = ['create_dataloader', 'preprocess_data']
  ```
  这样在其他模块中可以直接`from src.data import create_dataloader`。
- 依赖使用绝对导入，避免相对导入带来的混淆。例如：`from src.models.layers import ConvBlock`。

### 3.3 配置文件管理

- 使用YAML或JSON存储超参数、路径等配置。
- 通过配置类或字典加载配置，便于实验切换。
- 示例配置文件（`configs/training.yaml`）：
  ```yaml
  training:
    batch_size: 32
    epochs: 100
    learning_rate: 0.001
    optimizer: Adam
    loss: CrossEntropy
  data:
    train_path: data/processed/train.csv
    val_path: data/processed/val.csv
  ```
- 使用工具如`hydra`、`omegaconf`或`dataclasses`管理配置。这里推荐'omegaconf'。

---

## 4. 依赖管理与环境

### 4.1 虚拟环境

- 使用 **conda** 或 **venv** 创建隔离环境。
- 明确指定Python版本（如3.8+）。

### 4.2 依赖文件

- **requirements.txt**：列出所有直接依赖，并固定版本号（`package==x.y.z`）。
- **environment.yml**（若使用conda）：包含完整环境信息，便于跨平台复现。
- 使用`pip freeze > requirements.txt`前需清理无关包，建议使用`pipreqs`生成。

### 4.3 容器化

- 提供 **Dockerfile** 以标准化运行环境。
- 使用多阶段构建减小镜像体积。
- 示例Dockerfile片段：
  ```dockerfile
  FROM python:3.9-slim as builder
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --user -r requirements.txt
  
  FROM python:3.9-slim
  WORKDIR /app
  COPY --from=builder /root/.local /root/.local
  COPY src/ ./src/
  COPY scripts/ ./scripts/
  ENV PATH=/root/.local/bin:$PATH
  CMD ["python", "scripts/train.py"]
  ```

## 5. 版本控制

### 5.1 Git 使用规范

- 使用 `.gitignore` 忽略临时文件、数据、模型、虚拟环境、IDE配置等。
- 提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/)：`feat: add data augmentation`, `fix: correct learning rate scheduler`。
- 分支策略：`main`为主干，`develop`为开发分支，功能分支如`feature/data-pipeline`。

### 5.2 数据版本控制

- 对于大数据集，使用 **DVC (Data Version Control)** 或 **Git LFS**。
- DVC 将数据文件指针存储在Git中，实际数据可存在云存储（S3, GCS）或本地共享存储。
- 工作流：`dvc add data/raw` → `dvc push` → 提交`.dvc`文件到Git。

## 6. 测试

### 6.1 单元测试

- 使用 **pytest** 编写测试，测试文件置于`tests/`目录。
- 测试数据生成器或mock对象以避免依赖真实数据。
- 示例测试：
  ```python
  # tests/test_data.py
  import pytest
  from src.data.dataset import create_dataloader
  
  def test_dataloader_batch_shape():
      loader = create_dataloader(batch_size=4)
      batch = next(iter(loader))
      assert batch[0].shape[0] == 4
  ```

### 6.2 集成测试

- 测试端到端流程（如训练一小部分数据，验证模型不崩溃）。
- 可在CI流程中运行。

### 6.3 测试数据管理

- 使用小型合成数据集或固定样本子集进行测试。
- 将测试数据纳入版本控制（或通过脚本生成）。

## 7. 文档

### 7.1 代码注释与docstring

- 对公共API编写详细文档字符串（参照2.5）。
- 复杂算法需添加行内注释解释“为什么”而不是“是什么”。
  ——“是什么”还是需要的，但“为什么”更重要，尤其是对于复杂的模型设计或训练策略。

### 7.2 项目README

- 必须包含：项目简介、安装步骤、快速开始、目录结构、依赖、如何贡献、许可证。
- 使用Markdown格式。

### 7.3 自动化文档

- 使用 **Sphinx** 或 **MkDocs** 从docstring生成HTML文档。
- 配置自动API文档生成。

## 8. 实验跟踪与模型管理

### 8.1 实验记录

- 使用 **MLflow**、**Weights & Biases** 或 **TensorBoard** 记录每次实验的：
  - 超参数
  - 指标（损失、准确率等）
  - 模型权重
  - 代码版本（Git commit）
  - 环境信息
- 示例（MLflow）：
  ```python
  import mlflow
  
  with mlflow.start_run():
      mlflow.log_params({"lr": 0.01, "batch_size": 32})
      mlflow.log_metric("accuracy", 0.95)
      mlflow.pytorch.log_model(model, "model")
  ```

### 8.2 模型序列化与版本化

- 使用框架原生格式保存模型（PyTorch: `.pt`, TensorFlow: SavedModel）。
- 模型文件命名包含实验ID和时间戳：`model_20250101_123456.pt`。
- 将模型文件与实验记录关联，存储在`models/experiments/`下。

### 8.3 超参数管理

- 使用配置文件和命令行参数（如`argparse`或`click`）覆盖默认值。
- 推荐使用**Hydra**实现分层配置和组合。

## 9. 数据处理

### 9.1 数据获取与存储

- 原始数据存放于`data/raw/`，只读不修改。
- 处理后的数据存放于`data/processed/`，可通过脚本生成。

### 9.2 数据验证

- 使用 **Great Expectations** 或 **Pandera** 对数据质量进行验证（列存在、类型、值范围）。
- 在数据预处理前执行验证。

### 9.3 数据预处理pipeline

- 将预处理步骤封装为可组合的组件（如`sklearn`的`Pipeline`或自定义类）。
- 确保预处理逻辑在训练和推理时一致（保存预处理对象）。
- 示例（使用`sklearn`）：
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  
  preprocess_pipe = Pipeline([
      ('scaler', StandardScaler()),
      # ...
  ])
  ```

## 10. 训练与评估

### 10.1 训练脚本设计

- 训练脚本（`scripts/train.py`）应包含：
  - 解析配置
  - 初始化数据加载器
  - 构建模型
  - 设置优化器、损失函数
  - 训练循环（包括验证、早停、学习率调度）
  - 保存最佳模型
  - 记录指标
- 支持从检查点恢复训练。

### 10.2 分布式训练考虑

- 对于多GPU训练，使用框架内置工具（PyTorch DDP, TensorFlow MirroredStrategy）。
- 配置文件中设置分布式参数。

### 10.3 评估指标

- 实现常用指标（准确率、F1、IoU等）并支持扩展。
- 在验证集上定期计算，并在测试集上最终评估。
- 使用 `sklearn.metrics` 或自定义实现。

## 11. 模型部署

### 11.1 模型导出格式

- 根据部署环境导出模型：
  - **ONNX**：跨平台、高性能推理。
  - **TorchScript**（PyTorch）或 **SavedModel**（TensorFlow）。
  - **TensorRT**：NVIDIA GPU优化。
- 导出脚本单独存放（如`scripts/export_model.py`）。

### 11.2 服务化

- 使用 **REST API** 提供推理服务（Flask, FastAPI）。
- 或使用 **gRPC** 实现高性能通信。
- 示例（FastAPI）：
  ```python
  from fastapi import FastAPI
  from pydantic import BaseModel
  
  app = FastAPI()
  
  class InputData(BaseModel):
      features: List[float]
  
  @app.post("/predict")
  def predict(data: InputData):
      result = model.predict(data.features)
      return {"prediction": result}
  ```

### 11.3 CI/CD

- 配置CI流水线（GitHub Actions, GitLab CI）执行：
  - 代码格式化检查
  - 单元测试
  - 构建Docker镜像
  - 部署到测试环境
- 模型更新可通过CI/CD自动部署。

## 12. 安全与隐私

### 12.1 敏感数据处理
- 不要将包含个人身份信息（PII）的数据提交到Git。
- 使用环境变量或密钥管理服务存储API密钥、数据库密码，
- 对敏感数据加密存储。