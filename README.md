# nano_gpt
NanoGPT（微型 GPT）示例

CPU 友好，极小参数量

功能概述：
- 使用极小规模的 Transformer（约 2 万参数）进行字符级建模
- 学习从短输入到短输出的映射（如 "hello" -> "world"）
- 单文件包含：分词器、模型结构、数据构造、训练与生成

使用说明：
- 直接运行此脚本即可在 CPU 上训练并生成示例
- 如未安装 PyTorch，可用 CPU 版安装：pip install torch -f https://download.pytorch.org/whl/cpu

设计取舍：
- 词表仅包含 a-z 与空格，以及 BOS/SEP/EOS 特殊符号
- 模型配置：n_embd=32, n_head=2, n_layer=2, block_size=32
- 训练目标：下一字符预测（自回归），序列格式为 BOS + 输入 + SEP + 输出 + EOS

这份代码是一个单文件、CPU 可跑通的最小 GPT 教学框架，包含完整链条：
Tokenizer(字符级) → build_dataset(数据构造+噪声增强) → TinyGPT(Transformer)
→ 训练循环(AdamW+余弦退火+梯度裁剪) → generate(温度/top-k/top-p 采样)
→ checkpoint 保存/加载 → quick_generate 推理接口
