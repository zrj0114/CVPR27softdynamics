# MPM 教师与真实 GT 残差校正

该版本保留原来的动力学头、敏感度头和八项 MPM 教师损失。新增的小分支修正逐步运动量，真实 GT 只监督校正后的轨迹。没有将敏感度头替换为材料推断头。

## 模型与单位

`CoupledWorld(..., residual=None)` 是原架构，原检查点仍由 `load_coupled` 加载。启用残差时使用：

```python
from softdynamics.real_residual_train import initialize

model, base = initialize(
    '/path/to/base.pt',
    {'hidden': 128, 'max_velocity_m_s': 0.02},
    device='cuda:0',
)
```

转换复制所有基础权重、GRU、敏感度头和统计量，仅新增 `world.velocity_residual`。输入是共享编码器与因果记忆的特征、基础预测运动量、当前运动量；输出是每个节点的三维速度修正。默认分支含 34,575 个参数，末层权重和偏置初始化为零。

代码中的原速度通道是 `v_m_s * DT_REF / scale_m`，不是 m/s。残差先在 m/s 下逐分量限幅，再乘 `DT_REF / scale_m` 加入该通道，位置按实际时间间隔积分。校正后的状态同时反馈至下一步基础编码器和 GRU。物理尺度变化或帧率变化不改变限幅的物理含义。

该实现使用原 512 节点神经代理，没有改动物理教师或接触核。F/C 仍保留为未使用的接口列，残差限幅不保证局部体积、无翻折或无穿透；这与在完整 MPM 网格内注入残差的实现不同。

## 保留敏感度的语义

`confidence(contexts, q)` 仍输出两轴敏感度、相关性和半正定 Gram 矩阵。`local_response` 和 `CoupledWorld.forward` 的八项教师损失显式使用 `base_step`，包括成对轨迹损失。`forecast(..., corrected=False)` 可以检查未校正轨迹。

真实残差训练冻结所有基础参数和敏感度参数，但保留基础计算对输入状态的梯度，使误差能够跨校正后的状态递推回传。优化器只持有 `velocity_residual` 参数。

原敏感度描述 MPM／基础响应，不是校正后真实预测的置信概率。因此第一版真实 CMA 搜索对支持交互等权，不用未经校准的敏感度重新加权 GT。敏感度输出及原 MPM 训练入口继续可用。

## 真实观测包

每段交互保存为一个 NPZ。长度 T 包含完整夹爪动作时间线，可包含末尾没有 GT 的撤离帧。N 是节点数，P 是轨迹点数，S 是用于表面重建的采样点数；K 是固定绑定的邻居数。

| 数组 | 形状／含义 |
|---|---|
| `state0` | N×24，原基础模型的归一化初始状态；不包含未来 GT 状态 |
| `x0`, `origin`, `scale` | N×3 米制初始节点、3 维原点、正标量尺度 |
| `canonical`, `neighbors`, `surface_mask` | 原基础模型的参考坐标、节点邻居及表面标记 |
| `pads`, `times` | T×4×13：平移3、旋转9、存在标记1；T 个秒制时间戳 |
| `P`, `track0` | P×N 固定凸绑定、P×3 米制初始轨迹点 |
| `tracks`, `visible` | T×P×3 真实米制轨迹及 T×P 布尔可见性 |
| `surface0`, `surface_binding`, `surface_weights` | S×3 米制初始表面、S×K 节点绑定及权重 |
| `reference`, `reference_valid` | T×Q×3 真实表面观测及 T×Q 布尔掩码 |
| `phase` | T，0 接触前、1 交互中、2 释放 |

几何和动作沿用当前教师坐标约定，地面世界高度为 -0.01 m。节点运动通过固定初始绑定重建轨迹与表面，不能逐帧重找对应。遮挡 GT 可以为 NaN，但可见点、初始绑定、动作和状态必须有效。输入转换不会把未来 GT 传给模型。

观测索引格式：

```json
{
  "episodes": [
    {"object": "OBJECT_ID", "episode": 0, "data": "packets/episode_0.npz", "data_sha256": "OPTIONAL_SHA256"}
  ]
}
```

省略 `data` 时默认读取索引旁 `episodes/OBJECT_ID/episode_0/data.npz`，兼容已有完整真实交互比较导出。`data_sha256` 可以省略，准备时总会计算并固定哈希。对象训练／验证划分来自基础检查点，排除测试物体。索引需包含配置要求的所有支持交互。

```bash
python -m softdynamics.real_residual_data \
  --observations /path/to/observations.json \
  --base-checkpoint /path/to/old_run/training/best.pt \
  --output /path/to/new_real_data \
  --fit-episodes 0 1 2 --query-episodes 3 4 5 6 7 8 9

python -m softdynamics.real_residual_train prepare \
  --data-root /path/to/new_real_data --output /path/to/new_residual_run
```

准备步骤复制并固定基础检查点和观测数组、验证绑定和分割，不启动训练。旧实验目录和旧检查点不应作为输出目录。

## 残差训练与材料搜索

```bash
CUDA_VISIBLE_DEVICES=0 python -m softdynamics.real_residual_train train \
  --output /path/to/new_residual_run --device cuda:0
```

训练先用零残差模型，在每个对象的支持交互上联合搜索 log-E/ν；然后只在训练对象的支持交互上训练残差。默认每 500 次残差更新重新拟合训练材料。验证前在验证对象的支持交互上校准材料，以该对象的留出交互评估；验证／留出 GT 不参与残差反向传播或材料搜索。基础模型曾使用 MPM 数据中的交互，不能将这个协议描述为完全未见过交互的端到端训练。

每段从真实初始状态开始，之后始终使用预测状态。训练按阶段抽取前缀终点，部分前缀延伸到完整动作末尾；权重梯度尾段逐渐从 4、8、16 增长到 32 个区间，之前的预测前缀停止梯度，不用 GT 重置。验证与 CMA 始终完整递推。

真实损失单位为 mm：

- 固定轨迹对应的可见点 Huber，阈值 5 mm。
- 单向 GT→预测材料表面距离，以免把未观测到的背面拉向可见表面。
- 两者均可用时各占 0.5；只有一种观测时使用可用项。按阶段均值加权 0.1／0.45／0.45，缺失阶段重新归一化。
- 残差幅度和空间邻域平滑正则分别乘 0.1、0.05；修正速度按最大限幅归一化。

初始帧不计损失，未观测撤离帧不制造 GT。隐藏 NaN 在距离计算前排除。该目标不同于历史比较中的双向表面距离；跨版本指标比较必须重新统一评估口径。

CMA 默认 4 代、每代 6 个候选，外加初值评估；搜索 E=1000–12000 Pa、ν=0.08–0.42，两个轴联合优化。网络在搜索中固定。任何支持交互预测失败都会使整个候选获得有限大惩罚，失败记录保留。该预算是可调整的初始设置，不代表收敛。

搜索已训练残差检查点并输出留出轨迹：

```bash
python -m softdynamics.real_residual_train fit \
  --checkpoint /path/to/new_residual_run/training/best.pt \
  --data-root /path/to/new_real_data --object OBJECT_ID \
  --output /path/to/new_material_fit --device cuda:0
```

`selection.json` 先提交支持集选择结果，之后才评估和保存留出轨迹。材料估计是当前模型下的标定结果，不是材料实测标签。

## 检查点与恢复

残差检查点 schema 为 `softdynamics.coupled-residual-checkpoint.v1`，保存完整模型、基础来源与步数、残差配置、材料搜索结果、优化器和随机数状态。`base_step` 与残差更新 `step` 分开记录。

`train --resume` 恢复相同协议，包括材料重拟合边界。创建运行目录中的 `pause.request.json` 可在迭代边界保存并暂停。配置或源文件变化需要新运行目录。准备和测试通过不代表已经开始或完成残差训练。

CPU 检查：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
```

测试覆盖旧检查点零残差等价、物理单位、校正状态反馈、未来 GT 隔离、隐藏 NaN、教师损失与敏感度保留、仅残差更新、CMA 支持／留出隔离，以及重拟合边界的精确恢复。

2026-11-11 的验证记录见 [验证摘要](real_residual_validation_20261111.json)：302 项 CPU 测试通过；实际 step 16500 权重在牛角包和蛇玩偶上各做 32 步零残差等价检查，差异为零。一次性副本的 3 次反向更新与完整支持交互 CMA 检查通过，原基础参数和敏感度输出不变。这些检查用于验证实现，尚未启动正式残差训练，也不代表留出集效果改善。
