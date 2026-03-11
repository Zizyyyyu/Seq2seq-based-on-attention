# Seq2Seq Attention Machine Translation

基于PyTorch实现的序列到序列（Seq2Seq）注意力机制机器翻译模型，用于西班牙语到英语的翻译任务。

## 项目简介

本项目实现了一个带有注意力机制的Seq2Seq模型，使用GRU作为编码器和解码器，能够将西班牙语句子翻译成英语。项目包含了完整的训练流程、推理接口以及模型评估功能。

## 技术栈

- **PyTorch** - 深度学习框架，用于构建和训练神经网络模型
- **NumPy** - 数值计算库
- **Matplotlib** - 数据可视化，用于绘制训练损失曲线
- **NLTK** - 自然语言处理工具包，用于BLEU评分计算
- **tqdm** - 进度条显示工具
- **Python** - 3.9+

## 项目结构

```
Seq2seq-attention/
├── data/
│   └── spa.txt              # 西班牙语-英语平行语料库
├── logs/
│   ├── best_seq2seq_attention.pth  # 最佳模型权重
│   ├── src_tokenizer.pkl          # 源语言分词器
│   ├── trg_tokenizer.pkl          # 目标语言分词器
│   └── loss_curve.png             # 训练损失曲线
└── src/
    ├── model.py            # 模型定义（Encoder、Attention、Decoder、Seq2Seq）
    ├── train.py            # 训练脚本
    ├── dataset.py          # 数据集类
    ├── tokenizer.py        # 自定义分词器
    ├── utils.py            # 文本预处理工具
    └── interface.py        # 翻译推理接口
```

## 模型架构

### Encoder
- 使用GRU作为编码器
- 将输入序列编码为隐藏状态表示

### Attention
- 实现了加性注意力机制
- 计算查询和键之间的注意力权重
- 生成上下文向量

### Decoder
- 带有注意力机制的GRU解码器
- 结合上下文向量和嵌入向量生成预测

### Seq2Seq
- 整合编码器和解码器
- 支持teacher forcing训练策略


## 使用方法

### 1. 准备数据

确保 `data/spa.txt` 文件存在，格式为每行包含一个西班牙语句子和对应的英语句子，用制表符分隔：

```
Go.	Vete.
Hi.	Hola.
Run!	¡Corre!
```

### 2. 训练模型

运行训练脚本：

```bash
cd src
python train.py
```

训练参数：
- `embedding_dim`: 128
- `hidden_dim`: 256
- `epochs`: 10
- `learning_rate`: 0.0005
- `batch_size`: 64
- `teacher_force_ratio`: 0.5

训练过程中会：
- 保存每个epoch的模型权重到 `logs/` 目录
- 保存验证集上表现最好的模型
- 生成训练和验证损失曲线图
- 计算BLEU分数评估模型性能

### 3. 使用模型进行翻译

训练完成后，可以使用推理接口进行翻译：

```bash
cd src
python interface.py
```

输入西班牙语句子，模型会输出对应的英语翻译。输入 `quit` 退出程序。

示例：
```
Input Spanish: Hola mundo.
Translation: hello world .
```

## 模型评估

模型使用以下指标进行评估：
- **交叉熵损失**：训练和验证集上的损失值
- **BLEU分数**：使用NLTK的corpus_bleu计算翻译质量

## 注意事项

1. 训练脚本中的路径使用的是绝对路径，需要根据实际环境修改
2. 确保有足够的GPU内存，或使用CPU训练（速度较慢）
3. 可以通过调整超参数（embedding_dim、hidden_dim、learning_rate等）来优化模型性能
4. 数据集使用的是西班牙语-英语平行语料，可以替换为其他语言对

## 许可证

本项目仅用于学习和研究目的。
