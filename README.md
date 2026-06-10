# FEDQANG 代码归档说明

- **项目名称**：FEDQANG
- **作者**：陈睿齐
- **编写时间**：2026.06.10
- **归档目的**：整理毕业论文实验代码，说明联邦学习训练、区块链/P2P 通信、数据贡献博弈、日志记录和论文图片复现所需的关键路径与部署方式，便于后续复现实验、检查数据和继续扩展。

## 1. 项目目的

FEDQANG 面向去中心化联邦学习场景，研究在 MNIST、CIFAR-10、MedMNIST/PathMNIST 等数据集上的数据质量评估、激励分配与非合作博弈式数据贡献决策。代码主要完成以下任务：

1. 启动多个 P2P/gRPC 训练节点，模拟去中心化联邦学习参与方。
2. 对 MNIST、CIFAR-10、MedMNIST 数据进行客户端切分，支持 IID、Non-IID、类别偏置和实验专用划分。
3. 运行 CNN 本地训练、模型更新交换、质量分数计算、成本计算、收益/转移支付计算。
4. 记录每轮训练的 loss、test error、quality score、payoff、cost、lambda、transfer、datasize 等日志。
5. 运行非合作博弈仿真，分析不同参与方规模、不同质量场景下的收敛轮数与数据贡献。
6. 读取实验日志并生成论文图片，包括准确率/损失对比、质量分数曲线、转移支付柱状图、数据贡献图、数据分布图等。

## 2. 代码组成与文件说明

### 2.1 主运行入口

| 文件 | 作用 |
| --- | --- |
| `main.py` | 联邦学习节点启动入口。读取 `ip port` 参数，初始化 P2P/gRPC 网络后调用训练脚本。当前默认调用 `CIFAR10_CNN.run(0)`，可按实验需要切换为 `MNIST_CNN.run(0)` 或 `MedMNIST_CNN.run(0)`。 |
| `MNIST_CNN.py` | MNIST 联邦 CNN 实验主流程：模型初始化、本地训练、梯度/模型聚合、质量评价、激励记录。 |
| `CIFAR10_CNN.py` | CIFAR-10 联邦 CNN 实验主流程，包含实验配置、日志路径创建、客户端数据选择、训练与激励计算。 |
| `MedMNIST_CNN.py` | MedMNIST/PathMNIST 联邦 CNN 实验主流程。 |
| `*_CNN_path.py` | 对应数据集实验日志输出路径配置文件，例如 `MNIST_CNN_path.py`、`CIFAR10_CNN_path.py`、`MedMNIST_CNN_path.py`。运行不同实验前应先确认其中的 `path` 变量。 |

### 2.2 客户端、模型与数据集封装

| 文件 | 作用 |
| --- | --- |
| `client_MNIST.py` | MNIST 客户端逻辑：加载本地数据、训练本地模型、计算本地更新、测试、记录数据规模等。 |
| `client_CIFAR.py` | CIFAR-10 客户端逻辑。 |
| `client_MEDMNIST.py` | MedMNIST/PathMNIST 客户端逻辑。 |
| `mnist_cnn_model.py` | MNIST CNN 模型结构。 |
| `cifar_cnn_model.py` | CIFAR-10 CNN 模型结构。 |
| `medmnist_cnn_model.py` | MedMNIST/PathMNIST CNN 模型结构。 |
| `datasets.py` | 数据集注册与工具函数，统一返回 MNIST、CIFAR、MedMNIST 数据集类、特征数和类别数。 |
| `mnist_dataset.py` | 读取 `.npy` 格式 MNIST 客户端数据，返回 PyTorch Dataset。 |
| `cifar_dataset.py` | 读取 `.npy` 格式 CIFAR-10 客户端数据，返回 PyTorch Dataset。 |
| `medmnist_dataset.py` | 读取 `.npy` 格式 PathMNIST 客户端数据，返回 PyTorch Dataset。 |

### 2.3 P2P、区块链与激励机制

| 文件 | 作用 |
| --- | --- |
| `p2p.py` | P2P/gRPC 网络节点实现；读取 `ipport.txt` 中的节点地址；维护节点列表、模型列表、质量分数字典等全局网络状态。 |
| `grpc_pb2.py`、`grpc_pb2_grpc.py` | gRPC 通信代码。若修改 proto，需要重新生成。 |
| `blockchain.py` | 区块链数据结构、区块广播和链式记录逻辑。 |
| `core.py` | 全局区块链实例等共享对象。 |
| `transaction.py` | 交易结构与交易相关工具。 |
| `Hashring.py` | 哈希环工具，用于节点/委员会等映射逻辑。 |
| `synchronization.py` | 同步相关逻辑。 |
| `bc_enum.py` | 区块链/消息类型枚举。 |
| `transfer.py` | 激励转移支付相关函数。 |
| `cost_compute.py` | 成本计算。 |
| `pay_off_compute.py` | 收益/效用计算。 |
| `game.py`、`game_process.py` | 数据贡献博弈相关计算流程。 |

### 2.4 数据生成与解析脚本

| 文件 | 作用 |
| --- | --- |
| `mnist_parser.py` | 从 MNIST 原始数据生成客户端 `.npy` 文件，包含 uniform、按类别切片、Dirichlet Non-IID、实验二专用划分等函数。 |
| `cifar10_parser.py` | CIFAR-10 数据切分和 `.npy` 生成脚本。 |
| `medmnist_parser.py` | MedMNIST/PathMNIST 数据切分和 `.npy` 生成脚本。 |
| `dataset_view.py` | 数据查看辅助脚本。 |
| `visualize_distribution.py` | 读取客户端 `.npy` 数据并绘制类别分布堆叠柱状图。 |
| `tools/generate_non_coop_mock_data.py` | 生成非合作博弈仿真所需的质量分数模拟数据。 |

### 2.5 非合作博弈仿真

| 文件 | 作用 |
| --- | --- |
| `simulate_non_coop_game.py` | 单次非合作数据贡献博弈仿真入口；读取质量分数，求解每个参与方最优数据贡献，输出收敛轮数和明细日志。 |
| `run_non_coop_scenario_batch.py` | 批量运行多个质量场景和参与方规模，汇总 CSV 并绘制总轮数柱状/折线图。 |
| `quality_data/` | 非合作博弈质量场景输入数据。 |
| `game_log/` | 非合作博弈仿真输出日志、CSV 和图片。 |

### 2.6 画图程序

| 文件 | 主要用途 | 默认输入 | 默认输出 |
| --- | --- | --- | --- |
| `plot.py` | 通用训练日志曲线绘制。 | `log/` | `result/` |
| `plot_accuracy_compare.py` | 多方案 loss/error/accuracy 曲线对比。 | `log/cnn/...` | `result/cnn/...` |
| `plot_transfer.py` | 转移支付相关柱状图/曲线。 | `log/cnn/.../transfer` 或 `pay_off` | `result/cnn/...` |
| `plot_datasize.py` | 数据贡献/训练数据量变化图。 | `log/cnn/.../datasize` | `result/cnn/...` |
| `plot_federated_logs.py` | 联邦训练日志综合绘制。 | `log/cnn/...` | `result/cnn/...` |
| `plot_exp1.py` ~ `plot_exp5.py` | 英文实验图绘制脚本。 | 各脚本顶部 `LOG_DIR`/配置块 | `result/exp*` 或 `result/exp_v2/*` |
| `cn_plot_exp1.py` ~ `cn_plot_exp6.py` | 中文论文图绘制脚本。 | 各脚本顶部 `LOG_DIR`/配置块 | `result/chinese/exp*` |
| `visualize_distribution.py` | 客户端类别分布堆叠柱状图。 | `mnist/`、`cifar-10-batches-py/cifar10/`、`medmnist/` | 默认当前目录或脚本内 `save_path`，已有分布图也集中存于 `save/img/` |
| `print_transfer_totals.py` | 汇总转移支付数值，便于核对图表。 | `log/cnn/...` | 终端输出 |

> 注意：多数画图脚本将输入路径、输出路径、客户端端口等写在文件顶部配置区。复现实验图前，请先修改脚本顶部的 `LOG_DIR`、`OUTPUT_DIR`、`CONFIG`、`CLIENT_PORTS` 等变量。

## 3. Python 环境与依赖库

建议使用 Python 3.9 或 3.10，并优先创建独立虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

本项目用到的主要 Python 库如下：

```bash
pip install numpy pandas scipy scikit-learn scikit-image matplotlib pillow torch torchvision grpcio grpcio-tools protobuf cryptography pycryptodome python-mnist medmnist
```

依赖说明：

| 库 | 用途 |
| --- | --- |
| `torch`, `torchvision` | CNN 模型、本地训练、DataLoader、图像 transform。 |
| `numpy` | 数据切分、`.npy` 文件读写、向量计算。 |
| `pandas` | 部分数据/日志处理。 |
| `scipy` | 部分数值计算或优化辅助。 |
| `scikit-learn` | 传统模型、评价指标、实验辅助。 |
| `scikit-image` | 图像读取/处理辅助。 |
| `matplotlib` | 所有实验图绘制。 |
| `pillow` | 将数组转换为图像对象。 |
| `grpcio`, `grpcio-tools`, `protobuf` | P2P/gRPC 通信。 |
| `cryptography`, `pycryptodome` | 区块链/签名/加密相关逻辑。 |
| `python-mnist` | 读取 MNIST 原始 `idx` 文件。 |
| `medmnist` | 下载或处理 MedMNIST/PathMNIST 数据。 |

如果使用 GPU 训练，请根据本机 CUDA 版本从 PyTorch 官网安装对应版本的 `torch` 和 `torchvision`。

## 4. 关键路径说明

### 4.1 节点配置

- `ipport.txt`：P2P 节点列表，每行一个节点地址，格式通常为：

```text
127.0.0.1:50051
127.0.0.1:50053
127.0.0.1:50055
```

运行节点前请确保 `ipport.txt` 包含所有计划启动的节点地址。训练代码通常使用端口号推算客户端编号，例如 `(port - 50051) // 2`。

### 4.2 数据存储位置

项目使用 `.npy` 文件作为训练脚本直接读取的数据格式，通常最后一列为标签，前面列为展平后的图像像素。

| 数据集 | 默认/常用路径 | 文件命名示例 | 说明 |
| --- | --- | --- | --- |
| MNIST 原始数据 | `MNIST/raw/` | `train-images-idx3-ubyte` 等 | `mnist_parser.py` 读取的原始 MNIST 路径。 |
| MNIST 客户端数据 | `mnist/` 或仓库根目录 | `mnist0.npy`、`mnist_unif0.npy`、`mnist_noniid_0.1_client_0.npy`、`mnist_exp2_client_0.npy` | 由 `mnist_parser.py` 生成。 |
| CIFAR-10 原始/客户端数据 | `cifar-10-batches-py/cifar10/` | `cifar10_exp5_client_0.npy`、`cifar10_test.npy` | 由 `cifar10_parser.py` 生成或整理。 |
| MedMNIST/PathMNIST 客户端数据 | `medmnist/` | `pathmnist_exp1_client_0.npy`、`pathmnist_exp5_client_0.npy`、`medmnist_test.npy` | 由 `medmnist_parser.py` 生成或整理。 |
| 非合作博弈质量数据 | `quality_data/` | `non_coop_quality_scenarios/...` | `simulate_non_coop_game.py`、`run_non_coop_scenario_batch.py` 的输入。 |

> 大型原始数据集通常不建议提交到 Git；归档时应单独保存数据压缩包，并保持上述目录结构不变。

### 4.3 训练日志存储位置

训练日志集中在 `log/` 下，典型结构如下：

```text
log/cnn/<DATASET>/<experiment>/<scheme>/
├── loss/loss_<port>.txt
├── error/Test_error_<port>.txt
├── quality_score/Quality_score_<port>.txt
├── pay_off/pay_off<port>.txt 或 Transfer_<port>.txt
├── cost/cost_<port>.txt
├── lambda/lambda_<port>.txt
├── transfer/Transfer_<port>.txt
└── datasize/traindata_<port>.txt
```

示例路径：

- `log/cnn/MNIST/test/1_low_quality/our_scheme/quality_score/Quality_score_50051.txt`
- `log/cnn/CIFAR10/compare/1_low_quality/optimal/loss/loss_50051.txt`
- `log/cnn/MEDMNIST/exp_A/class/quality_score/Quality_score_50051.txt`

### 4.4 图片与结果存储位置

| 目录 | 内容 |
| --- | --- |
| `result/cnn/` | MNIST、CIFAR-10 等联邦训练对比图，常见格式为 `.png`、`.svg`、`.eps`。 |
| `result/chinese/` | 中文论文图，如实验 1~6 的中文标注图。 |
| `result/exp1` ~ `result/exp5` | 英文实验图或早期实验输出。 |
| `result/exp_v2/` | 新版实验图输出目录。 |
| `save/img/` | 数据分布可视化图片，如客户端类别堆叠柱状图。 |
| `game_log/` | 非合作博弈批量仿真的 CSV、日志和汇总图。 |

常见图片类型：

- `accuracy_compare.png/.svg/.eps`：准确率对比图。
- `loss_compare.png/.svg/.eps`：损失对比图。
- `quality_line.png/.svg/.eps`：质量分数变化图。
- `transfer_bar.png/.svg/.eps`：转移支付柱状图。
- `datasize_scheme_bar.png/.svg/.eps`：数据贡献/数据量对比图。
- `stacked_client_distribution_labels*.png`：客户端标签分布堆叠图。

## 5. 部署与运行流程

### 5.1 克隆代码并安装依赖

```bash
git clone <FEDQANG 仓库地址>
cd FEDQANG
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas scipy scikit-learn scikit-image matplotlib pillow torch torchvision grpcio grpcio-tools protobuf cryptography pycryptodome python-mnist medmnist
```

### 5.2 准备数据

1. 按第 4.2 节准备原始数据目录。
2. 根据实验数据集运行对应 parser，例如：

```bash
python mnist_parser.py
python cifar10_parser.py
python medmnist_parser.py
```

3. 检查输出的 `.npy` 文件是否位于训练脚本配置的目录中：
   - MNIST：`./mnist/` 或脚本中指定目录。
   - CIFAR-10：`./cifar-10-batches-py/cifar10/`。
   - MedMNIST：`./medmnist/`。

> 由于 parser 文件底部通常只启用某一个函数，运行前请打开对应 parser，确认 `if __name__ == "__main__"` 中调用的是你需要的数据划分函数。

### 5.3 配置实验路径

运行训练前检查以下文件：

1. `main.py`：选择要运行的数据集主流程：
   - MNIST：取消 `MNIST_CNN.run(0)` 注释。
   - CIFAR-10：使用 `CIFAR10_CNN.run(0)`。
   - MedMNIST：取消 `MedMNIST_CNN.run(0)` 注释。
2. `MNIST_CNN_path.py`、`CIFAR10_CNN_path.py`、`MedMNIST_CNN_path.py`：确认日志输出路径 `path`。
3. 对应 `*_CNN.py` 中的 `ExperimentConfig`：确认迭代轮数、batch size、数据文件前缀、数据目录、随机种子等参数。
4. `ipport.txt`：写入所有节点地址。

### 5.4 启动联邦学习节点

单机多进程模拟时，可打开多个终端，分别运行不同端口，例如：

```bash
python main.py 127.0.0.1 50051
python main.py 127.0.0.1 50053
python main.py 127.0.0.1 50055
```

如果需要启动更多参与方，继续按 `50057`、`50059`、`50061` 等端口启动，并同步更新 `ipport.txt`。所有节点启动后会通过 gRPC 通信，并在配置的 `log/` 路径下写入训练与激励日志。

### 5.5 运行非合作博弈仿真

单次仿真示例：

```bash
python simulate_non_coop_game.py --scenario-profile legacy --participant-counts 10 --total-datasize 500000
```

批量仿真示例：

```bash
python run_non_coop_scenario_batch.py --participant-counts 5,10,15,20,25 --total-datasize 500000
```

输出通常位于 `game_log/non_coop_batch/`，包括：

- `non_coop_total_rounds_by_scenario.csv`
- `non_coop_avg_total_rounds.csv`
- `non_coop_total_rounds_bar_with_fluctuation.png`

### 5.6 绘制论文图片

根据需要运行对应画图脚本：

```bash
python plot_accuracy_compare.py
python plot_datasize.py
python plot_transfer.py
python cn_plot_exp1.py
python cn_plot_exp2.py
python cn_plot_exp3.py
python cn_plot_exp4.py
python cn_plot_exp5.py
python cn_plot_exp6.py
python visualize_distribution.py
```

运行前请先检查脚本顶部的输入输出路径。图片会保存到 `result/`、`result/chinese/`、`result/exp_v2/`、`save/img/` 或脚本配置的其他目录。

## 6. 复现实验时的检查清单

1. Python 虚拟环境已激活，依赖库已安装。
2. 数据集原始文件和 `.npy` 客户端数据已放在脚本配置路径下。
3. `ipport.txt` 中节点数量与计划启动的进程数量一致。
4. `main.py` 中选择了正确的数据集运行入口。
5. `*_CNN_path.py` 的 `path` 指向本次实验的日志目录。
6. `*_CNN.py` 的 `ExperimentConfig` 中数据前缀、数据目录、迭代次数、batch size 与实验设计一致。
7. 训练结束后检查 `log/` 中各端口日志是否完整。
8. 画图前检查画图脚本顶部 `LOG_DIR`、`OUTPUT_DIR`、`CLIENT_PORTS` 等配置。
9. 图片输出后同时保存 `.png` 和论文常用的 `.eps`/`.svg`/`.pdf` 版本。

## 7. 归档建议

建议最终归档结构如下：

```text
FEDQANG/
├── README.md
├── *.py
├── tools/
├── ipport.txt
├── MNIST/raw/                 # 如需复现实验，单独随数据包保存
├── mnist/                     # MNIST 客户端 .npy 数据
├── cifar-10-batches-py/cifar10/ # CIFAR-10 客户端 .npy 数据
├── medmnist/                  # PathMNIST 客户端 .npy 数据
├── quality_data/              # 非合作博弈质量输入数据
├── log/                       # 联邦学习训练日志
├── game_log/                  # 非合作博弈输出
├── result/                    # 论文实验图
└── save/img/                  # 数据分布图
```

如果压缩包体积过大，可将 `log/`、`result/`、`game_log/`、原始数据集和 `.npy` 数据分别打包，并在本 README 中保留路径说明，确保解压后目录结构与代码配置一致。
