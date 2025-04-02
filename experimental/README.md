## 实验部分

### 目录结构

```bash
experimental/
├── data/               
│   ├── *.json          # 所有可用数据
│   ├── example/  
│   │   └── *.json      # 示例数据
│   └── origin_data/
│       ├── *.json      # 原始数据
│       ├── pipeline.py # 原始数据处理脚本
│       └── txt/
│           └── *.txt   # 原始纯文本数据
├── scripts/
│   ├── ner/            # 实体识别模型
│   ├── overview.py     # 数据概览
│   └── pre_trained/    # 预训练模型
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
