"""
NanoGPT（微型 GPT）示例（CPU 友好，极小参数量）
----------------------------------------------------------------
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

"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
torch.set_num_threads(1)

class Tokenizer:
    """
    极简字符级分词器。
    - 词表：特殊符号 <BOS>/<SEP>/<EOS>、小写字母 a-z、空格
    - encode：将字符串转为词表 id 列表
    - decode：将 id 列表还原为字符串（跳过前三个特殊符号）
    """
    def __init__(self, vocab=None):
        """
        可选自定义词表：
        - 若未提供 vocab，则使用默认（特殊符号+字母+空格）
        - 若提供 vocab（包含 <BOS>/<SEP>/<EOS>），可与检查点一致重建分词器
        """
        if vocab is None:
            specials = ["<BOS>", "<SEP>", "<EOS>"]
            letters = [chr(i) for i in range(ord("a"), ord("z") + 1)]
            space = [" "]
            self.vocab = specials + letters + space
        else:
            self.vocab = list(vocab)
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}
        # 特殊符号索引（要求词表中存在）
        self.bos = self.stoi.get("<BOS>", 0)
        self.sep = self.stoi.get("<SEP>", 1)
        self.eos = self.stoi.get("<EOS>", 2)
    def encode(self, s):
        s = s.lower()
        return [self.stoi[c] for c in s if c in self.stoi]
    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i in self.itos and i >= 3)

class TinyGPT(nn.Module):
    """
    超小型 GPT 模型：
    - Token 嵌入 + 位置嵌入
    - 多层 TransformerBlock（包含因果自注意力与 MLP）
    - 最终线性层生成各词的 logits
    """
    def __init__(self, vocab_size, n_embd=32, n_head=2, n_layer=2, block_size=32):
        super().__init__()
        self.block_size = block_size
        self.n_layer = n_layer  # 保存层数用于初始化缩放
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        # 权重绑定：共享 embedding 和输出层权重（不增加参数量，且能减少参数）
        self.tok_emb.weight = self.head.weight
        # 初始化权重
        self.apply(self._init_weights_scaled)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_weights_scaled(self, module):
        self._init_weights(module)
        if hasattr(module, "NANOGPT_SCALE_INIT"):
             # 按层数的平方根缩放（GPT-2 论文建议）
             module.weight.data *= (2 * self.n_layer) ** -0.5

    def forward(self, idx):
        # idx: (B, T) 的 token id 序列
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0).expand(B, T)
        # token 嵌入与位置嵌入相加
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, eos_id, temperature=1.0, top_k=None, top_p=None, eos_suppress_steps=3):
        """
        自回归生成（支持温度与 top-k 采样）：
        - temperature：温度缩放（>1 更随机，<1 更保守；=0 退化为贪心）
        - top_k：仅在概率最高的前 k 个词中采样（限制长尾，提升连贯性）
        - 若生成 EOS 则提前停止
        说明：
        - 先对 logits 应用温度缩放（logits / temperature）
        - 若设置 top_k，则将非前 k 的 logits 设为 -inf，使其概率为 0
        - 最终对 softmax 后的分布进行多项式采样得到下一个 token
        """
        for t in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :]  # 只取最后一个位置的预测
            if eos_suppress_steps and t < eos_suppress_steps:
                logits[:, eos_id] = float("-inf")
            if temperature is not None and temperature > 0:
                logits = logits / temperature
            else:
                # 温度为 0 时等价于贪心
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat([idx, next_id], dim=1)
                if (next_id == eos_id).all():
                    break
                continue
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits, k, dim=-1)
                cutoff = topk_vals[:, -1].unsqueeze(-1)
                logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
            if top_p is not None and 0 < top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                mask = cumprobs > top_p
                mask[..., 0] = False
                sorted_logits[mask] = float("-inf")
                logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if (next_id == eos_id).all():
                break
        return idx

class TransformerBlock(nn.Module):
    """
    标准 Transformer 块（预归一化风格）：
    - LayerNorm -> 自注意力 -> 残差
    - LayerNorm -> 两层 MLP -> 残余连接
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd * 2),
            nn.GELU(),
            nn.Linear(n_embd * 2, n_embd),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    """
    因果自注意力层：
    - Q,K,V 线性映射并分头
    - 通过下三角掩码确保仅关注历史位置（防泄露未来信息）
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        # 对残差投影层应用特殊初始化（缩放权重以保持深层网络的方差）
        self.proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size()
        # 将通道分为多头，形状变换为 (B, n_head, T, head_dim)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # 缩放点积注意力
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        # 因果掩码：仅允许关注到当前位置及其之前
        mask = torch.ones(T, T, device=x.device).tril()
        att = att.masked_fill(mask == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

def build_dataset(tok, block_size):
    """
    构造训练集（含数据增强与额外样例）：
    - 每条数据为（输入词 -> 输出词）映射
    - 打包为单一序列：BOS 输入 SEP 输出 EOS，并填充到固定长度
    - 通过噪声增强提升对输入拼写扰动的鲁棒性
    - 加入更多短词映射，测试未见样例的泛化行为
    """
    pairs = [
        # 基础映射（原始示例）
        ("hello", "world"),
        ("hi", "world"),
        ("cat", "dog"),
        ("abc", "xyz"),
        ("foo", "bar"),
        ("good", "bad"),
        ("red", "blue"),
        ("sun", "moon"),
        ("zero", "one"),
        ("begin", "end"),
        # 更多常见短词映射（覆盖更广的字母组合）
        ("hot", "cold"),
        ("left", "right"),
        ("on", "off"),
        ("up", "down"),
        ("open", "close"),
        ("day", "night"),
        ("man", "woman"),
        ("big", "small"),
        ("fast", "slow"),
        ("hard", "soft"),
        ("win", "lose"),
        ("in", "out"),
        ("black", "white"),
        ("east", "west"),
        ("north", "south"),
    ]

    def random_word(min_len=2, max_len=5):
        """生成随机短词（a-z），用于扩充样本与测试未见词。"""
        L = random.randint(min_len, max_len)
        return "".join(random.choice([chr(i) for i in range(ord("a"), ord("z") + 1)]) for _ in range(L))

    def noisy_string(s, p_subst=0.10, p_drop=0.05, p_swap=0.05):
        """
        对输入字符串施加轻量噪声：
        - p_subst：以该概率用随机字母替换某字符
        - p_drop：以该概率删除某字符
        - p_swap：以该概率与邻居交换（打乱相邻次序）
        目的：提升模型在输入拼写轻微扰动下的稳健性（输出不变）。
        """
        chars = list(s)
        # 替换/删除
        out = []
        for c in chars:
            if random.random() < p_drop:
                continue
            if random.random() < p_subst and c.isalpha():
                c = random.choice([chr(i) for i in range(ord("a"), ord("z") + 1)])
            out.append(c)
        # 相邻交换
        i = 0
        while i < len(out) - 1:
            if random.random() < p_swap:
                out[i], out[i + 1] = out[i + 1], out[i]
                i += 2
            else:
                i += 1
        return "".join(out)

    # 额外随机样例：输入为随机短词，输出从既有目标集中采样
    extra_targets = ["world", "dog", "xyz", "bar", "bad", "blue", "moon", "one", "end", "cold", "right", "down", "close"]
    for _ in range(40):
        a = random_word()
        b = random.choice(extra_targets)
        pairs.append((a, b))

    seqs = []
    for a, b in pairs:
        # 过滤非法字符并统一为小写
        a = "".join([c for c in a.lower() if c in tok.stoi or c == " "])
        b = "".join([c for c in b.lower() if c in tok.stoi or c == " "])
        # 以 SEP 分隔输入与输出，并在末尾加 EOS
        seq = [tok.bos] + tok.encode(a) + [tok.sep] + tok.encode(b) + [tok.eos]
        # 截断并填充至 block_size
        seq = seq[:block_size]
        if len(seq) < block_size:
            seq = seq + [tok.eos] * (block_size - len(seq))
        seqs.append(seq)
        # 噪声增强：对输入 a 注入轻微噪声，但输出 b 保持不变
        for _ in range(2):
            a2 = noisy_string(a)
            seq2 = [tok.bos] + tok.encode(a2) + [tok.sep] + tok.encode(b) + [tok.eos]
            seq2 = seq2[:block_size]
            if len(seq2) < block_size:
                seq2 = seq2 + [tok.eos] * (block_size - len(seq2))
            seqs.append(seq2)
    data = torch.tensor(seqs, dtype=torch.long)
    return data

def get_batch(data, batch_size):
    """
    从数据集中随机采样一个批次：
    - x 为输入序列（去掉最后一个 token）
    - y 为目标序列（去掉第一个 token），实现下一字符预测
    """
    ix = torch.randint(0, data.size(0), (batch_size,))
    x = data[ix, :-1]
    y = data[ix, 1:]
    return x, y

def count_params(model):
    """统计模型参数总量（用于验证规模约束）。"""
    return sum(p.numel() for p in model.parameters())

def save_checkpoint(path, model, tok, config):
    """
    保存检查点（权重+配置+词表）：
    - path：保存路径（如 'nano_gpt.pt'）
    - model.state_dict：用于重建权重
    - config：记录模型结构参数（n_embd/n_head/n_layer/block_size）
    - vocab：保存分词器词表以确保一致的编码/解码
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "vocab": tok.vocab,
        },
        path,
    )

def load_checkpoint(path, device="cpu"):
    """
    加载检查点并重建模型与分词器：
    - 返回 (model, tok, config)
    - 将模型移动到指定 device，并设为 eval 状态
    """
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    config = ckpt["config"]
    vocab = ckpt["vocab"]
    tok = Tokenizer(vocab=vocab)
    model = TinyGPT(
        vocab_size=len(tok.vocab),
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        block_size=config["block_size"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, tok, config

def quick_generate(checkpoint_path, prompt, temperature=0.9, top_k=8, top_p=0.95, samples=1, device="cpu"):
    """
    快速推理接口（便于在其他脚本中直接调用）：
    - checkpoint_path：检查点路径
    - prompt：输入提示词（字符级）
    - temperature/top_k：采样控制，见 TinyGPT.generate
    - samples：同一提示词生成次数，返回列表
    """
    #内部先调用 load_checkpoint 从磁盘加载模型
    model, tok, cfg = load_checkpoint(checkpoint_path, device=device)
    block_size = cfg["block_size"]
    outputs = []
    for _ in range(samples):
        prompt_clean = "".join([c for c in prompt.lower() if c in tok.stoi])
        ids = [tok.bos] + tok.encode(prompt_clean) + [tok.sep]
        # 截断过长的输入，保留 BOS
        ids = ids[:block_size]
        
        # 【修复】移除填充逻辑
        # 原填充逻辑导致 max_new_tokens 计算为 0，从而无法生成新内容
        # if len(ids) < block_size:
        #     ids = ids + [tok.eos] * (block_size - len(ids))

        inp = torch.tensor([ids], dtype=torch.long, device=device)
        
        # 计算剩余可生成的 Token 数量
        max_new_tokens = block_size - len(ids)
        #进行推理
        out = model.generate(
            inp,
            max_new_tokens=max_new_tokens, # 使用计算出的剩余长度
            eos_id=tok.eos,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_suppress_steps=3,
        )
        gen_ids = out[0].tolist()
        try:
            sep_index = gen_ids.index(tok.sep)
        except ValueError:
            sep_index = 0
        tail = gen_ids[sep_index + 1 :]
        if tok.eos in tail:
            tail = tail[: tail.index(tok.eos)]
        text = tok.decode(tail)
        if text == "":
            # 重试逻辑中也需要使用正确的 max_new_tokens
            out = model.generate(
                inp,
                max_new_tokens=max_new_tokens,
                eos_id=tok.eos,
                temperature=0.7,
                top_k=max(20, top_k or 0),
                top_p=0.98,
                eos_suppress_steps=5,
            )
            gen_ids = out[0].tolist()
            sep_index = gen_ids.index(tok.sep) if tok.sep in gen_ids else 0
            tail = gen_ids[sep_index + 1 :]
            if tok.eos in tail:
                tail = tail[: tail.index(tok.eos)]
            text = tok.decode(tail)
        outputs.append(text)
    return outputs


def main():
    """
    训练入口：
    - 构造数据与模型
    - 使用 AdamW 在 CPU 上训练若干迭代
    - 训练后调用生成函数对若干提示词进行输出
    - 演示保存/加载检查点与快速推理接口
    """
    #随机种子
    torch.manual_seed(42)
    device = torch.device("cpu")
    block_size = 32
    #实例化分词器
    tok = Tokenizer()
    #生成训练数据集和验证集
    data = build_dataset(tok, block_size)
    n_embd = 32
    n_head = 2
    n_layer = 2
    #实例化模型
    model = TinyGPT(vocab_size=len(tok.vocab), n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size).to(device)
    #计算参数总量
    params = count_params(model)
    print("params:", params)
    # 划分训练集与验证集 (90% / 10%)
    n_train = int(len(data) * 0.9)
    train_data = data[:n_train]
    val_data = data[n_train:]
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    batch_size = 16
    max_lr = 3e-3
    min_lr = max_lr * 0.1
    warmup_iters = 50
    iters = 600
    
    # 学习率调度器：带有预热的余弦退火
    def get_lr(it):
        if it < warmup_iters:
            return max_lr * (it + 1) / warmup_iters
        if it > iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    #初始化优化器
    opt = optim.AdamW(model.parameters(), lr=max_lr)
    #损失函数
    loss_fn = nn.CrossEntropyLoss()
    
    # 评估验证集 Loss 的函数
    @torch.no_grad()
    def estimate_loss():
        model.eval()
        eval_iters = 20
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(val_data, batch_size)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            losses[k] = loss.item()
        model.train()
        return losses.mean()

    for i in range(iters):
        # 动态更新学习率
        lr = get_lr(i)
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        #获取一个批次的输入 x 和目标 y。
        x, y = get_batch(train_data, batch_size)
        x = x.to(device)
        y = y.to(device)
        #前向传播
        logits = model(x)
        #计算损失
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad()
        #反向传播
        loss.backward()
        # 梯度裁剪（提升稳定性）
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #更新参数
        opt.step()
        #每 100 次迭代调用 estimate_loss 评估验证集损失并打印
        if (i + 1) % 100 == 0:
            val_loss = estimate_loss()
            print(f"iter {i+1} | lr {lr:.4e} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f}")

    def gen(prompt, temperature=0.9, top_k=8, top_p=0.95, samples=1):
        """
        推理函数（支持多样性采样）：
        - 将提示词打包为 BOS + prompt + SEP 序列并填充
        - 通过 generate 使用温度与 top-k 策略进行采样
        - 返回 SEP 之后到 EOS 之前的内容
        参数：
        - temperature：采样温度（0 为贪心；默认 0.9 提升随机性）
        - top_k：限制采样候选的数量（默认 8，兼顾连贯与多样）
        - samples：对同一提示词生成次数（便于观察多样性）
        """
        outputs = []
        for _ in range(samples):
            prompt_clean = "".join([c for c in prompt.lower() if c in tok.stoi])
            ids = [tok.bos] + tok.encode(prompt_clean) + [tok.sep]
            ids = ids[:block_size]
            # if len(ids) < block_size:
            #     ids = ids + [tok.eos] * (block_size - len(ids))
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(
                inp,
                max_new_tokens=block_size - len(ids),
                eos_id=tok.eos,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_suppress_steps=3,
            )
            gen_ids = out[0].tolist()
            try:
                sep_index = gen_ids.index(tok.sep)
            except ValueError:
                sep_index = 0
            tail = gen_ids[sep_index + 1 :]
            if tok.eos in tail:
                tail = tail[: tail.index(tok.eos)]
            text = tok.decode(tail)
            if text == "":
                out = model.generate(
                    inp,
                    max_new_tokens=block_size - len(ids),
                    eos_id=tok.eos,
                    temperature=0.7,
                    top_k=max(20, top_k or 0),
                    top_p=0.98,
                    eos_suppress_steps=5,
                )
                gen_ids = out[0].tolist()
                sep_index = gen_ids.index(tok.sep) if tok.sep in gen_ids else 0
                tail = gen_ids[sep_index + 1 :]
                if tok.eos in tail:
                    tail = tail[: tail.index(tok.eos)]
                text = tok.decode(tail)
            outputs.append(text)
        return outputs
    # 评估集包含：训练词、近似扰动词、以及未见随机短词
    prompts = [
        "hello", "hi", "cat", "abc", "foo", "good", "red", "sun", "zero", "begin",
        # 扰动版本（模拟输入噪声）
        "helo", "godd", "reed", "sunn",
        # 未见短词（随机）
        "tree", "milk", "bird", "glass", "lamp", "train", "note"
    ]
    for p in prompts:
        gs = gen(p, temperature=0.9, top_k=8, top_p=0.95, samples=3)
        print(p, "->", "; ".join(gs))
    # 保存检查点并演示快速加载与推理
    config = {"n_embd": n_embd, "n_head": n_head, "n_layer": n_layer, "block_size": block_size}
    ckpt_path = "nano_gpt.pt"
    #将训练好的模型保存
    save_checkpoint(ckpt_path, model, tok, config)
    print("checkpoint saved ->", ckpt_path)
    demo_outs = quick_generate(ckpt_path, "hello", temperature=0.9, top_k=8, samples=3, device=device)
    print("loaded quick_generate('hello') ->", "; ".join(demo_outs))

if __name__ == "__main__":
    main()

