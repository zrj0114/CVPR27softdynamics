# 数据接口

仓库从已有的初帧重建和轨迹准备结果开始；不包含原始 RGB 的 Gaussian Splatting 重建器、跟踪器或 Deform360 下载镜像。先使用既有重建/跟踪流程，或官方处理数据，准备以下输入。所有 NPZ 默认 `allow_pickle=False`。

## 编码器输入

| 字段 | 形状 | 约定 |
|---|---|---|
| positions | T,N,3 | `(world_position-origin)/scale`，固定点编号 |
| valid | T,N | 布尔可见性；遮挡处仍保留有限的重建坐标 |
| geometry | N,15 | 初帧归一化位置 3 + 插值 canonical rest position/scale 3 + 初始 F 展开 9 |
| actions | T,28 | 两个夹爪，每个为 EEF 3×4 位姿 12 + opening/scale 1 + 有效标志 1 |
| times | T | 秒，严格递增 |
| scale_m | 标量 | 物体尺度，预训练误差换算为毫米时使用 |
| point_ids | N | 对应上游固定跟踪点编号 |

预处理选择首帧可见的最多 512 点；不根据未来运动或标签选点。origin 为这组点的首帧均值，scale 为首帧最大轴跨度。后续筛选点集时保持原有 origin/scale；CLI 支持通过输入的 `point_ids`、`origin_m`、`scale_m` 显式复用。推理至少需 16 帧和 128 点；预训练至少 32 帧。

每个 EEF 位姿按行展开，平移先减 origin 再除 scale；旋转不缩放。缺失 EEF 观测及不存在的第二夹爪填零，有效标志为零。网络将每个窗口内的控制位姿转换为相对窗口末帧。物体遮挡位置不应任意填零：网络的窗口参考位置仍来自跟踪结果。

## 教师和物理评分输入

`observations/episode_N.npz`：

| 字段 | 形状 | 含义 |
|---|---|---|
| saved_ids | P | 完整物理资产中的材料点编号，与 runtime trajectory 完全一致 |
| initial | P,3 | 上述材料点的该段初始世界坐标，米 |
| surface_ids | K | 边界点在 saved_ids 数组中的索引 |
| tracks | T,N,3 | 固定对应的可见/重建跟踪点，米 |
| track_visible | T,N | 布尔可见性 |
| binding | N,B | 每个跟踪点绑定到 initial 的 B 个近邻索引 |
| weights | N,B | 固定插值权重，各行和为 1 |
| frames | T | 原始模拟帧编号，按严格递增排列 |
| point_ids | N | 上游点编号 |
| observations | T,M,3 | 每帧表面观测，padding 后存储；教师无需此项 |
| counts | T | 每帧有效表面点数；物理评分使用 |

`episode_N.json` 至少有 `clip_start`、`clip_end`，均为原始帧编号。评分要求三个阶段都有观测。

可以直接复用旧实验的这个 observation cache；不需要重建资产。新数据可用 `softdynamics.data.bind_observations` 建立固定材料绑定，再追加表面观测和有效数量。`saved_ids` 从基准 runtime 的 `trajectory.npz` 读取；候选模拟必须与基准一致。完整资产的 `body.npz`、初态和双指动作格式见 runtime 文档。

## 合成恢复实验

训练敏感度预测头时，`input` 应对应教师的合成真值运动。保持真实初始几何、控制动作及点可见性，使用已知参数模拟得到 tracks。不要给多个不同合成运动复用同一份真实运动输入。训练/验证/测试的划分单位仍是源交互，同一源交互的全部合成变体必须同组。

检查参数恢复时，`observations` 和 `tracks` 应来自已知真值模拟，加噪后作为固定观测；候选模拟使用相同初态和动作。真实数据确认则使用实际观测。这两类实验不混称“恢复了真实参数”。
