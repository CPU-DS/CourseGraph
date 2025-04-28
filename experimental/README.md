## 实验部分

### 目录结构

```bash
experimental/
├── data/               
│   ├── *.json          # 所有可用数据
│   ├── example/        # 示例数据
│   │   └── *.json
│   └── origin_data/    # 原始数据
│       ├── *.json
│       ├── pipeline.py # 原始数据处理脚本
├── scripts/
│   ├── ke/             # 知识抽取模型
│   ├── prompt/         # 提示工程
│   ├── evaluate/       # 评估
│   └── overview.py     # 数据概览
├── pre_trained/        # 预训练模型
├── results/            # 结果
└── README.md

```

### 实验准备

#### Swanlab

登录 [Swanlab](https://swanlab.cn)， 获取 API Key 并配置到环境变量中
```bash
export SWANLAB_API_KEY=
```

#### 评估指标

#### 数据

### 数据格式说明
