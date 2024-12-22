# Jittor Geometric

`jittor_geometric` 是一个基于 Jittor 的几何深度学习库，主要用于图神经网络（GNN）的开发和训练。本项目提供了多种工具、数据集和神经网络模块，帮助用户构建和训练图神经网络模型。

## 项目结构

### 目录说明

- **data**：包含与数据处理相关的工具和类，例如数据加载器和数据预处理器，用于简化图数据的管理和处理。

- **datasets**：数据集loader。

- **io**：负责输入输出操作，例如从不同文件格式读取或写入数据。

- **loader**：提供多种加载器，用于管理图或批数据的加载，便于模型训练和评估。

- **nn**：包括专门为图任务设计的神经网络模块，可能包含用于构建 GNN 的层和架构。

- **ops**：包含自定义操作或函数，用于图数据的处理，例如邻接矩阵操作、采样等。

- **utils**：包含各种辅助函数和工具，如日志记录、配置解析和其他常用工具。

## 快速开始

请确保已安装所有必要的依赖项。可以查看示例脚本以快速了解不同的 GNN 模型和数据集的使用方法。

### 安装

```bash
conda env create -f jittor_env.yml -n myGML
```

### 使用方法

运行任意示例脚本可以使用以下命令格式：


```bash
# 异配图
python gcn_example.py
```

```bash
# 动态图
python tgn_example.py
```


```bash
python train.py --data REDDIT     --num_neighbors 10 --use_cached_subgraph --use_onehot_node_feats
python train.py --data WIKI       --num_neighbors 30 --use_cached_subgraph --use_onehot_node_feats
python train.py --data MOOC       --num_neighbors 20 --use_cached_subgraph --use_onehot_node_feats
python train.py --data LASTFM     --num_neighbors 10 --use_cached_subgraph --use_onehot_node_feats
```