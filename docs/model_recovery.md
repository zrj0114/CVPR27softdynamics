# 冻结预训练模型的参数优化对照

`softdynamics.recovery_data` 和 `softdynamics.recovery_benchmark` 将清洗版 Deform360 模型接回完整 FCR/MPM 反演。比较对象为现有均匀交互损失的坐标搜索流程；本实验没有物理反向传播，不代表原论文所有梯度优化配置。

## 协议

- 固定 `training/full/model.pt`，不重训，不用下游误差选 checkpoint。
- 使用模型测试物体 048、095、125、187。要求该 episode 的 center/mixed 离线真值轨迹都完成；只使用完成状态，不读取敏感度值或标签有效性掩码。
- 每个物体最大 episode 序号为查询集，其余为拟合集。实际为 048：拟合 0/1/2/3/4/6/7/8，查询 9；095：拟合 0/2/3，查询 4；125：拟合 0/2，查询 8；187：拟合 0/1/2/3，查询 5。
- 同物体共用已有 canonical asset；每段使用已有初态和完整真实动作。每段的网格余量取先前两条成功教师轨迹使用余量的较大值，在所有候选中固定，时间步也固定。所有候选都从原初态完整前向模拟，逐帧检查有限性、正行列式和网格边界，不用教师中间状态重置优化 rollout。
- 表面点从 canonical 边界确定性采样 1500 点，跟踪点沿用已准备的视觉点。真实评分表面从每帧至少双视角支持的点云确定性采样最多 1500 点。没有更换初始几何或读取触觉。
- 保留 `objective.score`：对称表面距离和 5 mm Huber 跟踪误差各占一半，前／中／后阶段权重 0.1/0.45/0.45。缺测帧不评分，但模拟不跳帧。

## 方法

三种方法共享所有候选物理损失：

1. `uniform`：每段交互等权。
2. `learned`：当前参数 j 更新时，以模型输出 c[e,j] 加权每段交互的完整拟合损失。
3. `shuffled`：分别打乱每个参数的模型分数，保留数值分布，检查收益是否依赖正确的交互对应关系。

分数由拟合集运动、初始几何、动作及初始参数猜测预测，在三轮搜索中冻结。合成加噪实验的模型输入与拟合观察使用相同噪声；首次几何不加噪。查询集不参与分数生成、搜索或候选接受。

所有方法沿用 E、ν、地面摩擦、速度衰减的三轮循环，每次评价 −0.025/0/+0.025 三个候选。每次试验为 36 个逻辑候选参数向量，每个向量模拟全部拟合交互。物理结果可缓存共享，但不能把缓存命中宣称为某方法的运行加速。常数正权重与均匀权重在这种搜索中等价，已有测试验证。

合成实验使用原先的 center 和 mixed 两组真值、两个相反符号的初值。观测噪声为 0、0.5、2 mm；非零噪声使用 19/23/29 三个种子，组合独立点噪声和时间平滑共模漂移。真值位于有限网格上，只验证局部离散恢复。每个物体 84 个合成试验，另有三种方法各一个真实观测试验。

真实拟合从 q=0 开始；仅在候选有效且无权重的拟合集损失严格改善时接受，否则回退初始参数。记录原候选、是否接受和最终查询误差；不通过查询误差决定是否接受。

## 输出与判据

- `plan.json`：模型哈希、协议、数据和划分。
- `trajectories/`：每个候选的完整时间轴评分点、逐帧物理检查、来源哈希、各条件损失。
- `weights/`：固定模型输出和打乱对照；不读取教师标签。
- `selections/`：查询评分前冻结的参数选择及每一步完整候选代价。
- `results/`：合成参数误差（每项以及平均网格步）、完全恢复比例、查询表面／跟踪／组合误差；真实数据只报告观察误差，没有真实物理参数标注。
- `report.json`：每个物体等权汇总与同条件配对差值。重复初值／种子不作为独立物体。
- `analysis/recovery_report.md`、`recovery_report.json`、`recovery_comparison.png/pdf`：三种方法都完成的同条件配对汇总、独立模拟失败计数、真实初始参数的查询基线和执行核查。查询均值使用三者共同有效的上下文；同时保留逐方法有效数，不将失败样本静默删除。
- `analysis/efficiency_report.md/json`、`efficiency_budget_curves.png/pdf`：每次参数更新后的参数恢复曲线、首次／稳定达到真值的预算、达到同一最终候选的配对预算，以及完整一轮无变化的早停规则重放。实际实验仍跑满三轮；不会把真值驱动的事后到达时刻当作可部署早停条件。
- `jobs.json`、`status.json`：可恢复的调度和进度。

先运行准备，再从冻结代码快照后台运行调度器：

```bash
python -m softdynamics.recovery_data --mode prepare \
  --root /path/to/new/recovery --pretrain /path/to/completed/pretrain
python -u -m softdynamics.recovery_benchmark --mode supervise \
  --root /path/to/new/recovery --gpus 0 1 2 3 4 5 6 7
```

必须保证 `plan.json` 的 runtime 指向同一冻结快照。调度器用文件锁防止重复，恢复时续接自己的现存 worker；GPU 不空闲或磁盘不足时等待，不停止其他用户任务。数值失败保留为失败候选；基础设施错误最多重试一次并显式报告。

用只读分析入口生成结果，或随新结果自动更新：

```bash
python -m softdynamics.recovery_report --root /path/to/recovery --verify-payload
python -u -m softdynamics.recovery_report --root /path/to/recovery --watch
python -m softdynamics.recovery_efficiency --root /path/to/recovery
python -u -m softdynamics.recovery_efficiency --root /path/to/recovery --watch
```

分析程序用原始 `search.cycle` 重放每次已完成搜索，检查相同逻辑预算、完整候选代价、查询评分前冻结的参数、源文件哈希、固定网格／时间步以及模拟时间轴。`--verify-payload` 额外核对全部已访问轨迹文件哈希。分析过程不生成新候选，不改变实验协议或运行中的模型。

效率分析同时列出逻辑候选访问数和每种方法假设拥有独立空缓存时的唯一候选数。重复一轮可能只读取已有轨迹，不能将少一轮直接表述成少一轮物理模拟。严格运行时间对比需计入模型推理，并使用相同硬件、独立缓存和相同停止规则；共享多卡任务完成先后不作为方法加速证据。
