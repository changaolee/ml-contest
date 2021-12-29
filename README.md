> 基于 PaddlePaddle 的机器学习比赛

### 已完成比赛

| 完成日期 | 赛题 | 方案详情 |
| --- | --- | --- |
| 2021-11-23 | [产品评论观点提取](https://www.datafountain.cn/competitions/529) | [情感分类](./notebook/data_fountain_529_senta)、[命名实体识别](./notebook/data_fountain_529_ner) |

### 运行

```bash
// 初始化环境变量
$ source env-gpu.sh

// 开始训练
$ python3 run/{exp_name}/train.py

// 执行预测
$ python3 run/{exp_name}/predict.py
```

### 目录结构

```
.
├── config                                              // 配置目录
│   ├── data_fountain_529_ner.json                      // DataFountain 529 NER 任务
│   └── data_fountain_529_senta.json                    // DataFountain 529 SENTA 任务
├── data                                                // 数据目录
│   ├── data_fountain_529_ner
│   │   └── README.md
│   └── data_fountain_529_senta
│       ├── data_augmentation                           // 数据增强目录
│       │   └── 929e3b6b06b3df8ce5bfce403e48c820        // 基于配置参数生成的唯一 key
│       │       ├── test.csv                            // 增强后的测试数据
│       │       └── train.csv                           // 增强后的训练
│       ├── data_difficulty_assessment                  // 数据难度评估目录
│       │   ├── 929e3b6b06b3df8ce5bfce403e48c820
│       │   │   ├── assessed_data.csv                   // 难度评估后数据标签（原始标签+难度标签）
│       │   └── └── data_difficulty_score.json          // 原始数据难度打分
│       ├── processed                                   // 处理后数据（直接输入模型）
│       │   └── 929e3b6b06b3df8ce5bfce403e48c820
│       │       ├── dev_0.csv                           // 开发集 K 折划分
│       │       ├── dev_1.csv                           // ...
│       │       ├── dev_2.csv                           // ...
│       │       ├── test.csv                            // 测试集
│       │       ├── train_0.csv                         // 训练集 K 折划分
│       │       ├── train_1.csv                         // ...
│       │       └── train_2.csv                         // ...
│       ├── README.md
│       ├── submit_example.csv                          // 赛题原始数据文件
│       ├── test_public.csv                             // ...
│       └── train_data_public.csv                       // ...
├── data_process                                        // 数据处理
│   ├── data_fountain_529_ner.py
│   ├── data_fountain_529_senta.py
│   └── __init__.py
├── dataset                                             // 模型数据集构建
│   ├── data_fountain_529_ner.py
│   ├── data_fountain_529_senta.py
│   └── __init__.py
├── env-gpu.sh                                          // 环境初始化脚本（GPU）
├── env.sh                                              // 环境初始化脚本（CPU）
├── experiment                                          // 运行中数据存储目录
│   └── data_fountain_529_senta
│       ├── checkpoint                                  // 模型 checkpoint 参数
│       ├── log                                         // 运行日志
│       │   └── data_fountain_529_senta.log
│       ├── result                                      // 预测结果
│       └── visual_log                                  // 训练可视化
├── infer                                               // 推断器
│   ├── data_fountain_529_ner.py
│   ├── data_fountain_529_senta.py
│   └── __init__.py
├── model                                               // 模型
│   ├── data_fountain_529_ner.py
│   ├── data_fountain_529_senta.py
│   └── __init__.py
├── notebook                                            // notebook（数据探查等）
│   └── data_fountain_529_senta
│       └── 情感分类：数据探索.ipynb
├── README.md
├── requirements-gpu.txt                                // 项目依赖（GPU）
├── requirements.txt                                    // 项目依赖（CPU）
├── resource                                            // 一些其他资源文件
│   ├── font
│   │   └── SimHei.ttf
│   └── stopwords
│       ├── baidu_stopwords.txt
│       ├── cn_stopwords.txt
│       ├── hit_stopwords.txt
│       └── scu_stopwords.txt
├── run                                                 // 执行入口（脚本）
│   └── data_fountain_529_senta
│       ├── predict.py                                  // 结果预测
│       ├── scripts                                     // 一些定制化脚本
│       │   └── gen_entity.py
│       └── train.py                                    // 模型训练
├── trainer                                             // 训练器
│   ├── data_fountain_529_ner.py
│   ├── data_fountain_529_senta.py
│   └── __init__.py
└── utils                                               // 一些公共方法
    ├── adversarial.py                                  // 对抗训练（FGM）
    ├── config_utils.py                                 // 配置相关
    ├── __init__.py
    ├── loss.py                                         // 自定义损失函数
    ├── metric.py                                       // 自定义评价指标
    ├── nlp_da.py                                       // NLP 数据增强方法
    ├── sdk                                             // 第三方 SDK
    │   ├── baidu_translate.py                          // 百度翻译
    │   └── __init__.py
    └── utils.py                                        // 一些工具方法
```