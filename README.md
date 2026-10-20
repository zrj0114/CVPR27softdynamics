# CVPR27softdynamics

面向 Deform360 交互的物理参数敏感度学习与参数优化。此仓库整理了当前实验实际使用的核心算法，去除了旧实验目录依赖、结果网页和作业管理脚本。

四个输出依次是 **杨氏模量 E、泊松比 ν、地面摩擦、速度衰减率**。每个输出独立，比较同一个物体的不同交互对同一参数的敏感程度；不使用触觉，不做跨参数 softmax。

## 代码结构

| 路径 | 功能 |
|---|---|
| `softdynamics/data.py` | 固定点轨迹、初始几何和 EEF 动作的归一化；跟踪点材料绑定 |
| `softdynamics/motion.py`, `pretrain.py` | 历史运动编码器、未来动作条件预测、按物体划分预训练 |
| `softdynamics/teacher.py`, `labels.py` | 完整状态恢复、短窗中央差分、步长稳定性检查、缺失标签掩码 |
| `softdynamics/features.py`, `training.py` | 冻结编码器、四个独立预测头、整段交互划分、训练集尺度校准 |
| `softdynamics/inference.py` | 无需教师或模拟器的模型推理 |
| `softdynamics/weights.py`, `search.py` | 固定参数级交互权重、等权/常数/教师/打乱/模型对照、循环坐标搜索 |
| `softdynamics/rollouts.py`, `objective.py` | 完整物理模拟、缓存、多卡分配和表面/跟踪误差评分 |
| `runtime/` | 共享 canonical asset、每段初态配准、接触与确定性 EMPM 前向模拟 |
| `configs/croissant_four_parameters.json` | 当前牛角包局部实验的物理尺度、窗口和搜索配置 |

此版本没有新增模型结构或更换物理公式。主要变化是显式传入路径、移除旧缓存依赖、合并可复用接口。完整实验集群的调度器不在仓库内；搜索入口按需运行所访问的候选，共用实际轨迹缓存。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
# 生成物理标签、构建资产或执行真实仿真时额外安装：
pip install -e '.[physics]'
pytest -q
```

建议 Python 3.10 以上。物理环境需安装匹配显卡驱动的 PyTorch/CUDA，Warp 固定为 1.8.1。`runtime/` 从源码 checkout 使用；普通模型训练和推理只依赖安装后的 `softdynamics` 包。

## 从数据到模型

数据集、资产、模型权重、教师标签和运行结果均不随代码上传。真实数据和合成运动都需保留固定点对应关系。字段与准备步骤见 [数据接口](docs/data_contract.md)。

### 1. 准备编码器输入

```bash
softdynamics-prepare --input data/raw_episode.npz \
  --body data/workspace/asset/body.npz \
  --initial-state data/workspace/initial_states/episode_0.npz \
  --output data/normalized/episode_0.npz
```

`raw_episode.npz` 包含米制 `tracks`、可见性 `visible`、官方 EEF `poses`、开口 `openings` 和秒制 `times`。物理接触需要的双指位姿是另外一个输入，不能用 EEF 位姿直接替代。

### 2. 预训练运动编码器

预训练清单是一组 `{"id":"object/episode", "object":"object", "path":"relative/episode.npz"}`。相对路径以清单文件目录为起点。

```bash
softdynamics-pretrain --manifest data/pretrain.json \
  --heldout-object 121-croissant-plush --output runs/pretrain --device cuda:0
```

固定训练预算后才读取留出物体并评价；未来物体观测只作为监督目标，未来控制动作可以输入网络。当前实验的编码器用 sponge/bear 预训练，croissant 没有参与编码器预训练。

### 3. 离线生成敏感度教师

```bash
softdynamics-teacher --config configs/croissant_four_parameters.json \
  --workspace data/workspace --episode 0 --truth center \
  --observation data/observations/episode_0.npz \
  --metadata data/observations/episode_0.json \
  --output runs/teachers/center/episode_0 --device cuda:0
```

先用已知参数连续模拟真实初态和控制动作，再保存完整中间状态。在交互前/中/释放后各选两个窗口，每次只扰动一个参数、推进一个观测间隔；合成 phase-weighted RMS 运动导数。粗细步长不一致时细化，仍不可靠的标签保存为 NaN 和掩码。输出含每个工作点的 `S`、`valid`、`q` 和 Jacobian。

对于当前牛角包数据，episode 4/6/7/8/9 使用 `--padding 64`；其他物体的域大小需要自行配置。参数基准和尺度不能直接通用于全部物体。生成程序支持的快照契约是 elastic FCR，其他材料不能直接套用。

### 4. 提取特征并训练独立预测头

特征清单每行包含 `episode_id`、`input` 和 `teacher`，例如 `{"episode_id":"croissant/0", "input":"normalized/episode_0.npz", "teacher":"teachers/episode_0/workpoint_0.npz"}`。同一交互的所有真值、窗口和参数工作点必须使用相同 `episode_id`，避免划分泄漏。

```bash
softdynamics-features --manifest data/features.json --encoder runs/pretrain/encoder.pt \
  --config configs/croissant_four_parameters.json --output data/features.npz
softdynamics-train --dataset data/features.npz --encoder runs/pretrain/encoder.pt \
  --config configs/croissant_four_parameters.json \
  --validation croissant/8 --test croissant/9 --output runs/confidence
```

训练目标为 `log1p(S / tau)`，其中 `tau` 仅由训练交互确定。每个参数至少需要三个有可靠标签的训练交互。验证集选择停止点，测试集只做评价。循环更换验证/测试的整段交互即可做 leave-one-episode-out；不能把重复工作点当成新的独立交互。

### 5. 推理与物理参数搜索

```bash
softdynamics-predict --checkpoint runs/confidence/model.pt \
  --input data/normalized/episode_0.npz \
  --parameters 3547.0699 0.278514 0.853391 0.359543

softdynamics-weights --manifest data/search.json --checkpoint runs/confidence/model.pt \
  --start 0.025 0.025 0.025 0.025 --output data/weights.npz
softdynamics-search --config configs/croissant_four_parameters.json \
  --weights data/weights.npz --manifest data/search.json --workspace data/workspace \
  --gpus 0 1 --cache runs/rollouts --output runs/search.json
```

`search.json` 的有序 episode 列表与权重一致，每行包含 `episode`、`input`、`observation`、`metadata`、`padding`。`input` 给网络，`observation` 给物理评分；两者字段不同。需要教师对照时提供该初始工作点的 `teacher` 文件并显式添加 `--teacher-mask`。此时所有对照共享教师可靠性掩码，不能将它表述成完全无需教师的部署验证。

也可给搜索传入 `--loss-bank data/full_losses.npz`，直接复用已算好的完整候选误差。它必须包含 `q[C,4]`、`losses[C,E]`、`episodes[E]`，缺失候选会报错，不插值或偷偷跳过。

一次搜索固定初始预测的四维权重。每个参数在三档取值之间比较完整轨迹损失，循环三轮；没有通过模拟器进行梯度回传。改变权重和改变参数方向是两件事：方向来自候选损失。新增候选需要重新跑完整物理轨迹，因此可能很耗时。

搜索输出是候选参数，包含物理有效性和未加权目标；不会覆盖原资产。真实数据实验应在参数冻结后单独运行留出交互，并按预先规定的验收规则决定是否采用候选。

## 方法边界

- 分数为 `S/(S+tau)`，是相对局部运动敏感度，尚未标定成参数正确概率。
- 教师固定中间状态，不包含该状态对参数的历史导数；不等于完整轨迹总导数。
- 当前目标不扣除参数耦合；保存相关矩阵供诊断，没有把 Schur complement 接入优化。
- 完整物理拟合从每段初态连续运行，没有用后续真实几何重置模拟器。
- 拟合误差变小不能单独证明恢复了真实材料参数；需要已知真值恢复和留出交互验证。
- 此仓库是实验代码整理，不宣称当前敏感度模型已稳定超过等权或常数基线。

[运行物理资产代码](runtime/README.md) · [数据接口](docs/data_contract.md) · [来源与许可证](THIRD_PARTY.md)
