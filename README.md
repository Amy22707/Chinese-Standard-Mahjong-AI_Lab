# Chinese Standard Mahjong Lab

本项目为北京大学人工智能基础大作业。
本项目实现了一个面向 Botzone国标麻将平台 / IJCAI麻将人工智能比赛的国标麻将 AI Bot，核心目标是在四人、不完全信息、8 番起胡的复杂规则下，训练一个兼顾胡牌能力、大牌能力和防守稳定性的智能体。最终提交 Bot 在 IJCAI 积分赛中取得第 12 名。

## 项目背景

国标麻将具有动作空间大、隐藏信息多、奖励稀疏、四人博弈非平稳等特点。相比围棋、五子棋等完全信息游戏，麻将 Bot 不仅要判断自己能否更快胡牌，还要根据弃牌、副露、牌墙进度和分数形势估计点炮风险。本项目主要探索两条路线：

- 使用监督学习（SL）从牌谱中学习强 Bot 的决策，并通过特征工程和推理后处理提升实战表现。
- 使用强化学习（RL）在 SL 初始化基础上进行 PPO 自博弈微调，探索长期收益优化。

实践结果表明，在有限算力和训练时间下，增强版 SL Bot 明显比未充分训练的 RL Bot 更稳定。

## 核心算法

### 监督学习 Bot

SL 部分位于 `src/SL`，当前版本包含以下设计：

- `feature.py`：将麻将局面编码为 70 通道 `4 x 9` 张量，只使用比赛时可见的信息。特征包括手牌、副露、弃牌、剩余牌估计、牌墙进度、现物危险度、弃牌位置、向听数、有效牌、七对子/清一色路线，以及可选的公开分数和名次差上下文。
- `model.py`：使用 ResNet 卷积主干提取牌面特征，并加入 GRU 编码最近弃牌序列。动作空间被分解为动作类型头和具体动作子头，同时包含胜率、番数、向听、弃牌排序、点炮风险和番型路线等辅助头。
- `preprocess.py`：把原始牌谱转换为 `.npz` 训练样本，支持赢家过滤、番数加权、风险标签、全局弃牌序列和番型路线标签。
- `dataset.py`：实现懒加载 LRU 缓存和花色置换数据增强，降低数据读取开销。
- `supervised.py`：使用 AdamW、余弦学习率退火、加权交叉熵、动作类型损失、条件子动作损失和多任务辅助损失训练模型。
- `__main__.py`：Botzone 交互入口。推理阶段在模型输出上加入轻量后处理，包括胡牌优先、听牌进攻、现物奖励、危险牌惩罚、吃碰杠惩罚、七对子保对子和后期防守门控。

### 强化学习 Bot

RL 部分位于 `src/RL`，实现 PPO 自博弈框架：

- `actor.py`：并行采样对局，支持与历史模型池中的对手对战。
- `learner.py`：使用 PPO clipped objective、GAE、价值函数、熵正则和 SL teacher KL 约束更新策略。
- `model_pool.py`：维护历史模型，增加自博弈对手多样性。
- `env.py`：封装国标麻将环境，并加入胡牌得分、流局听牌奖励和未听牌惩罚。
- `train.py`：训练入口，支持 SL checkpoint warm start、断点续训、NPU/CUDA/CPU 自动选择和 TensorBoard 日志。

由于麻将奖励极其稀疏，且四人自博弈样本效率低，当前 RL 在有限训练预算下没有超过 SL。后续更合理的路线是用强 SL checkpoint 初始化，在更长训练周期内使用 KL 约束进行小步微调。

## 项目结构

```text
.
├── src/
│   ├── SL/                 # 监督学习 Bot、特征、模型、训练和 Botzone 入口
│   └── RL/                 # PPO 自博弈强化学习框架
├── docs/
│   ├── mahjong_sl_report.tex
│   ├── mahjong_sl_report.pdf
│   └── deep-research-report.md
├── models and logs/        # 不同版本 checkpoint 与实验代码备份
├── requirements.txt
├── LICENSE
└── README.md
```

## 环境依赖

推荐使用 Python 3.10 或更高版本。核心依赖包括：

- `torch`
- `numpy`
- `MahjongGB`
- `tensorboard`（可选，用于 RL 日志）
- `torch_npu`（可选，仅 Ascend NPU 环境需要）

安装依赖：

```bash
pip install -r requirements.txt
```

如果在 Ascend NPU 环境训练，需要额外安装与机器 CANN 版本匹配的 `torch_npu`。

## 运行指南

### 1. 预处理 SL 数据

将原始牌谱放在 `src/SL/data/data.txt`，然后运行：

```bash
cd src/SL
python preprocess.py --workers 4
```

如果希望保留四家全部样本，而不是只保留赢家样本：

```bash
python preprocess.py --workers 4 --all-players
```

### 2. 训练 SL 模型

```bash
cd src/SL
python -u supervised.py --num-workers 0 --epochs 16 --batch-size 1024
```

在部分 Ascend/NPU 环境中，`num-workers > 0` 可能因为多进程与算子编译交互而卡住。若无输出或长时间停在编译 warning 后，建议先使用 `--num-workers 0` 保证稳定，再逐步尝试 1 或 2。

实时查看日志可以使用：

```bash
python -u supervised.py --num-workers 0 --epochs 16 --batch-size 1024 > training.log 2>&1
tail -f training.log
```

### 3. 运行 Botzone SL Bot

`src/SL/__main__.py` 会默认读取 `/data/best9.pkl`，也可以通过环境变量指定模型路径：

```bash
cd src/SL
MODEL_PATH=./model/checkpoint/best.pkl python __main__.py
```

Windows PowerShell 中可写为：

```powershell
$env:MODEL_PATH="./model/checkpoint/best.pkl"
python __main__.py
```

可选推理参数：

- `POSTPROCESS_MODE=none`：关闭规则后处理，直接使用模型输出。
- `USE_AUX_RANK=1`：使用弃牌排序辅助头参与打牌评分。
- `USE_RISK_HEAD=1`：使用风险辅助头参与点炮风险估计。

### 4. 训练 RL 模型

```bash
cd src/RL
python train.py --sl_checkpoint ../SL/model/checkpoint/best.pkl --num_actors 8 --episodes_per_actor 200
```

RL 训练耗时远高于 SL。若只是复现实战提交，优先使用 SL checkpoint；RL 更适合作为后续长时间微调实验。

## 主要优化总结

- 将基础特征扩展到 70 通道，并避免训练时引入不可见的上帝视角信息。
- 使用 ResNet + GRU 融合结构，同时建模静态牌面和弃牌时间序列。
- 将 235 维动作空间拆成动作类型与子动作，降低训练难度。
- 使用赢家过滤、番数加权和花色置换，提升高质量样本与大牌样本的影响。
- 引入风险、番型路线、向听、胜率和弃牌排序辅助任务。
- 推理阶段加入攻守平衡后处理，降低远手危险弃牌和不必要吃碰杠。
- 实现 PPO 自博弈框架，为 SL 到 RL 的长期迁移预留接口。

## Bot实操展示
![](https://ik.imagekit.io/Amyxue/Chinese_Standard_Mahjong/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20260620140036_972_514.png)
## 参考资料

- Suphx: Mastering Mahjong with Deep Reinforcement Learning.
- IJCAI Mahjong AI Competition / Botzone Chinese Standard Mahjong.
- `docs/deep-research-report.md` 中整理的公开麻将 Bot 文献与实现思路。
- MahjongGB 国标麻将规则与计番库。

## AI 工具声明

本项目开发过程中使用了 ChatGPT 5.5 Codex 辅助进行代码审查、特征设计讨论、训练问题排查、参数调优建议和文档润色。核心代码结构、训练实验、比赛提交和最终策略取舍均由作者结合实验结果人工确认与修改。
