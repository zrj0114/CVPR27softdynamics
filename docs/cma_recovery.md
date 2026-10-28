# Ours + CMA-ES 连续参数恢复

本实验保留冻结预训练模型、同物体共享首帧 canonical asset、每段独立初态、完整真实夹爪动作、FCR/MPM 求解器和原表面/跟踪评分。改变的是参数搜索，候选不再限制在三档。安装额外依赖：`pip install -e '.[search]'`。

## 如何保留四个独立置信度

当前采用**按参数轮换的 CMA-ES**。优化参数 j 时固定其余三个参数，每段交互完整前向模拟得到 L[e,q]，目标为：

`F_j(q) = sum_e(c[e,j] * L[e,q]) / sum_e(c[e,j])`。

等权对照使用 c=1。四个置信度独立，不做 softmax，不把四列取平均。权重仅由拟合交互和初始参数生成，整个搜索中冻结；不读取教师置信度，不用查询运动。

每个参数块使用 pycma 的 CMAEvolutionStrategy，单维度，4 个候选/代、2 代/块。依次优化 E、ν、地面摩擦、速度衰减，循环 3 次，在 32/64/96 个候选处冻结参数并评价查询交互。这是组合的参数块算法，**不是同时搜索四个参数、学习完整 4×4 协方差的联合 CMA-ES**。

为绕过 pycma 4.5.0 单维度自动 maxstd 初始化错误，关闭可选 maxstd 上限，保留 BoundTransform 的实际参数边界；CMA_active=False。状态和独立随机数生成器一同保存，断点恢复不会重新采样待评候选。

## 对照、预算和边界

Ours 和等权使用相同种子 9101、初值、块顺序、初始 sigma=0.2、边界和候选预算。原 q 范围 [-0.025,+0.025] 线性映射到 CMA 的 [0,1]。每个块保留进入块时的参数作为候选，只有有效且改善该块目标的候选能替换它。

该范围覆盖当前模型的训练工作点；本实验验证局部连续搜索，不能证明宽范围材料识别。合成真值使用原 center/mixed，报告连续参数误差和已知真值恢复阈值：每项误差不超过原 0.025 网格间距的 0.1 倍。真实数据没有物理参数真值，仅在未加权拟合误差严格优于 q=0 时接受候选，并独立报告查询误差。

固定上下文：四个原测试物体，每物体包含真实观察，以及 center/mixed 无噪声和 mixed 2 mm、seed19 的正/负两个初值。没有根据上一轮哪个上下文出现改善来挑选。总计 56 个搜索、每个 3 个预算评估。

预算记录候选向量数、逻辑 episode 评价数、包含起始评价的数量，以及每种方法假设独立缓存时的唯一候选数。共享物理缓存和后台并行完成时间不能作为方法运行加速的证据。无真值驱动早停；预算间的恢复效率需对相同条件配对比较。

## 运行和输出

```bash
python -m softdynamics.cma_benchmark --mode prepare \
  --root /path/to/cma --previous /path/to/completed_or_stopped/recovery
python -u -m softdynamics.cma_benchmark --mode supervise \
  --root /path/to/cma --gpus 0 1 2 3 4 5 6 7
```

准备阶段验证旧轨迹 payload、初态、asset、动作、时间步和网格余量，再硬链接已完成轨迹；条件评分在新目录重新计算。连续候选以完整浮点坐标的 SHA256 为缓存键，不与附近三档候选合并。各候选始终从初态完整模拟，无中间状态重置或物理反传。

首次使用时，可先运行 `python -m softdynamics.recovery_data --mode prepare --root /path/to/recovery --pretrain /path/to/pretrain`，再以这个准备好的目录作为 `--previous`；无需先完成三档搜索。`prepare` 会将源码与 runtime 复制到新运行目录的 `code/`，检查之前使用的求解器及适配器逐文件一致。`physics_parity.json` 记录的是代码一致性，不代表又运行了一轮轨迹重放。缓存硬链接要求源、目标位于同一文件系统。

- `plan.json`、`code_manifest.json`、`physics_parity.json`：冻结模型、协议和源码核验。
- `search/<object>/<context>/<method>.pkl/json`：可恢复 CMA 状态、逐代候选、损失和参数轨迹。
- `selections/`：查询评价前冻结的参数与接受判断。
- `results/`、`report.json`：32/64/96 预算的查询误差和配对差值；数值失败单独保留。
- `status.json`、`jobs.json`、`logs/`：多卡调度和进度；持续检查其他用户 GPU 进程归属，使用空闲卡，不停止其他用户任务。
