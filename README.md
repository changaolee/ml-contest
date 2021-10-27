### 目录结构

```
.
├── README.md
├── base
│   ├── __init__.py
│   ├── data_loader_base.py
│   ├── infer_base.py
│   ├── model_base.py
│   └── trainer_base.py
├── config
│   └── data_fountain_529.json
├── data
│   └── data_fountain_529
├── datasets
│   ├── __init__.py
│   └── data_fountain_529.py
├── doc
├── experiments
│   └── data_fountain_529
├── infer
│   └── __init__.py
├── model
│   └── __init__.py
├── run
│   ├── test.py
│   └── train.py
├── test
│   └── test.py
├── trainer
│   └── __init__.py
└── utils
    ├── __init__.py
    ├── config_utils.py
    └── utils.py
```

### 配置信息
#### data fountain 529

```
// 情感分类任务：config/data_fountain_529_senta.json
{
  "exp_name": "data_fountain_529_senta",  // 任务名称
  "train_filename": "train_data_public.csv",  // 原始训练集文件名
  "test_filename": "test_public.csv",  // 原始测试集文件名
  "dev_prop": 0.2,  // 开发集划分比例
  
  /**
   * 模型参数配置
   */
  "max_seq_len": 256,
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "pretrained_model_name": "bert-wwm-chinese",
  "num_classes": 3
  
   /**
   * 训练参数配置
   */
  "train_batch_size": 64,
  "train_epochs":3,
  "learning_rate": 2e-5
}
```

