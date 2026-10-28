# CVPR27softdynamics

整理已实现的 **初始 real2sim 流程 + 第一创新点：交互条件下的物理参数敏感度学习与加权参数辨识**。

同一物体共用 canonical asset，各段交互仅用各自首帧配准初态。模型对每段交互输出四个独立分数：**杨氏模量 E、泊松比 ν、地面摩擦、速度衰减率**。优化某个参数时，在同一物体的不同交互之间使用对应的那一列权重；不使用触觉，不做跨参数 softmax。

```text
初帧 Gaussian + 多视角首帧轮廓 + 已测夹爪动作
  → 共享材料 asset + 每段初态 → EMPM 完整前向模拟
  → 离线有限差分敏感度标签 → 运动编码器预训练 + 四个独立预测头
  → 冻结置信度模型 → 按参数加权 CMA-ES → 留出交互评测
```

数据、资产、教师标签、训练权重和实验输出不包含在代码仓库中。首帧 Gaussian、掩码、相机和动作来自已有的 Deform360 数据处理结果；本仓库不重新实现上游 Gaussian 训练或夹爪标定。

## 安装

建议从源码 checkout 以 editable 模式安装，Python >=3.10。物理模拟需要与驱动匹配的 PyTorch/CUDA，Warp 固定为 1.8.1。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[physics,deform360,search,report,test]'
pytest -q
```

六项指标中的 Gaussian 渲染另需安装外部 EMPM checkout 的 rasterizer、SSIM 和 LPIPS 依赖，见[评测说明](docs/six_metric_evaluation.md)。`runtime/` 从源码 checkout 使用；普通神经网络推理只依赖 `softdynamics` 包。

## 两个已有物理运行版本

| 使用场景 | 入口 | 输入与接触 |
|---|---|---|
| 原始牛角包共享资产实验 | `runtime/prepare_legacy.py`, `build_asset.py`, `build_queries.py`, `run_episode.py`; `adapters/empm.py` | 旧版测量的 marker 位姿、固定材料接触标签、50×36 mm 接触面；参考 episode 5 |
| 清洗版数据的全量第一创新点 | `shared_data.py`, `shared_teacher.py`; `adapters/pad_empm.py` | 修正相机、独立指垫中心位姿、64×25 mm 接触面；各物体最早的候选段作参考 |

两者共用 FCR/MPM 求解核心，但接触坐标、尺寸、材料接触资格及网格设定有区别。历史约 3.19 mm 是旧牛角包实验特定协议的结果，不能直接作为清洗版多物体或其他指标的结果。复现时保留同一套 asset、初态、动作、参数、接触版本和评分定义。

原始初始化与运行命令见 [runtime/README.md](runtime/README.md)。历史 28 维 EEF 编码器、教师与三档坐标搜索接口见[历史敏感度流程](docs/legacy_sensitivity.md)。

## 当前第一创新点：从清洗数据到加权 CMA-ES

### 1. 数据准备、离线标签与训练

`SOURCE` 是包含 `物体/episode_N` 的 processed 数据目录，`CORRECTED` 是已洗好的相机和独立指垫交付目录。输出使用新目录，源数据只读。

```bash
SOURCE=/path/to/processed
CORRECTED=/path/to/corrected_release
PRETRAIN=/path/to/pretrain_run

python -m softdynamics.shared_data --select-only \
  --source "$SOURCE" --corrected "$CORRECTED" \
  --config configs/corrected_release_pretrain.json --output "$PRETRAIN"
python -u -m softdynamics.shared_pipeline --root "$PRETRAIN" \
  --gpus 0 1 --prep-workers 4
```

这条流程完成首帧多视角体积构建、共享资产与各段初态、教师标签、运动编码器预训练和四个独立预测头训练。模型按物体划分；同一物体的所有交互、真值和工作点同组。输出模型为 `training/full/model.pt`。已有完整预训练目录可以直接用于下一步。

教师固定完整中间状态，逐一扰动物理参数，通过中央差分计算短窗运动导数，并用两组步长检查可靠性。网络学习 `log1p(S/tau)`，推理映射为 `S/(S+tau)`；`tau` 仅由训练数据确定。详细字段、筛选和失败处理见[清洗版预训练说明](docs/corrected_release_pretrain.md)。

### 2. 冻结模型，准备同物体多交互参数辨识

```bash
RECOVERY=/path/to/recovery_inputs
python -m softdynamics.recovery_data --mode prepare \
  --root "$RECOVERY" --pretrain "$PRETRAIN"
```

读取第一步实际生成的 `training/full/model.pt`，保存模型与资产哈希。各物体的拟合交互和查询交互在搜索前冻结。这里准备的是完整轨迹拟合及评测输入，不需要先跑完旧三档搜索。

### 3. Ours + CMA-ES，与等权严格配对

```bash
CMA=/path/to/cma_run
python -m softdynamics.cma_benchmark --mode prepare \
  --root "$CMA" --previous "$RECOVERY"
python -u -m softdynamics.cma_benchmark --mode supervise \
  --root "$CMA" --gpus 0 1
```

准备时创建实际的代码和 runtime 快照，检查与前一阶段的物理源文件一致，再导入可复用轨迹。若已有三档搜索目录，可用它代替 `RECOVERY` 复用经过哈希验证的缓存；硬链接缓存导入要求两个运行目录在同一文件系统。

优化第 j 个参数时使用 `sum_e(c[e,j] * L[e,q]) / sum_e(c[e,j])`，等权使用 `c=1`。`c` 由冻结模型在初始参数处计算，搜索中不读取教师置信度、不使用查询交互。每个候选都从各段初态完整模拟。当前是按参数轮换的一维 CMA-ES，四个分数保持独立；不是联合四维协方差搜索。

两种方法共享初值、随机种子、搜索边界与前向预算，报告 32/64/96 个候选时的结果。局部范围继承已有模型工作点。详见 [CMA-ES 协议](docs/cma_recovery.md)。

### 4. 留出交互与六项指标

```bash
EVALUATION=/path/to/evaluation
python -m softdynamics.render_evaluation --mode prepare \
  --root "$EVALUATION" --run "$CMA" --empm /path/to/external/empm
python -m softdynamics.render_evaluation --mode all \
  --root "$EVALUATION" --device cuda:0
```

参数先冻结，再计算 Dist、Track、IoU、PSNR、SSIM、LPIPS。评测不重新拟合参数或外观。几何/图像指标与用于搜索的阶段加权损失分别报告，定义及外部依赖见[六指标评测](docs/six_metric_evaluation.md)。

## 代码索引

| 模块 | 功能 |
|---|---|
| `runtime/prepare_legacy.py`, `build_asset.py`, `registration.py`, `physical_asset.py` | 原始首帧体积、共享材料资产及初态配准 |
| `runtime/build_queries.py`, `run_episode.py`, `gs-layer/`, `empm/` | 动作打包、接触与确定性 MPM 前向模拟 |
| `shared_data.py`, `shared_pipeline.py` | 清洗版输入、共享资产、可恢复多卡调度 |
| `teacher.py`, `shared_teacher.py`, `labels.py` | 完整状态恢复、有限差分、步长检查及标签掩码 |
| `motion.py`, `pretrain.py`, `shared_training.py`, `features.py`, `training.py` | 编码器预训练与独立参数头 |
| `inference.py`, `weights.py` | 冻结模型推理及参数级交互权重 |
| `recovery_data.py`, `recovery_benchmark.py`, `cma_search.py`, `cma_benchmark.py` | 完整轨迹拟合、三档对照及连续 CMA-ES |
| `recovery_report.py`, `recovery_efficiency.py` | 配对收益、候选预算和恢复效率 |
| `evaluation_metrics.py`, `render_evaluation.py` | 六项指标、物理重放与外观传输 |

表中的资产和物理运行文件位于 `runtime/`，训练、搜索和评测模块位于 `softdynamics/`。实验输入契约见[数据接口](docs/data_contract.md)，整理验证记录见[发布核对说明](docs/code_release.md)。

## 方法边界与来源

- 分数尚未标定成“参数正确概率”；每个参数独立，不限制一段交互只能对一个参数敏感。
- 教师是条件于中间状态的局部偏导，不含完整历史对参数的总导数；参数相关矩阵用于诊断，未接入 Schur complement。
- 参数更新使用完整前向候选搜索，没有通过 MPM 做物理反传；不使用后续真实几何重置模拟。
- 实现完整不等于已证明加权稳定优于等权；需依据固定预算、留出交互和已知真值对照判断。

保留原有许可证及第三方版权。EMPM 派生运行时受非商业研究许可限制，根目录 MIT 不覆盖其限制。见 [THIRD_PARTY.md](THIRD_PARTY.md) 和[来源清单](docs/source_provenance.json)。
