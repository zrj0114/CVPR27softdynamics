# 历史牛角包敏感度与三档搜索入口

这些命令保留早期 28 维 EEF 动作实验的复现接口。当前清洗版 52 维指垫训练和 CMA-ES 入口见仓库首页。

## 从数据到模型

数据集、资产、模型权重、教师标签和运行结果均不随代码上传。真实数据和合成运动都需保留固定点对应关系。字段与准备步骤见 [数据接口](data_contract.md)。

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

固定训练预算后才读取留出物体并评价；未来物体观测只作为监督目标，未来控制动作可以输入网络。历史牛角包实验的编码器用 sponge/bear 预训练，croissant 没有参与该编码器预训练；清洗版全量模型使用单独的按物体划分流程。

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
