# 共享资产与 EMPM 运行时

这里保留实验使用的资产构建、初态配准、单向接触和固定顺序 P2G 前向计算。路径适配后，物理源代码目录与数据工作目录分离：

```bash
export SOFTDYNAMICS_WORKSPACE="$PWD/data/workspace"
export SOFTDYNAMICS_CAPTURE_ROOT="$PWD/data/captures"
python runtime/build_asset.py --reference-episode 5 --episodes 0 1 2 3 4 5 6 7 8 9
python runtime/build_queries.py
python runtime/run_episode.py --episode 0 --transfer-mode deterministic \
  --compact-output --output-root "$PWD/runs/baseline"
```

如已有共享资产和准备好的输入，可直接放入 workspace 后从最后一步开始。此配置的接触尺寸、固定密度和地面高度沿用牛角包实验，不是通用夹爪校准器。

## 构建资产前需要的文件

```text
workspace/
  prepared/episode_N/
    first-frame-volume.npz   # positions: 3 mm 网格体素中心，世界坐标米
    source/splat_0.ply       # 初帧 Gaussian 几何和外观
    actions.npz             # 下面定义的独立双指动作
captures/episodes/episode_N/
  cameras.npz
  repair_evidence/manifest.json  # clip_start, clip_end
```

资产构建把参考 episode 的初帧体素细分到 1.5 mm，保存同一套 canonical material IDs、体积、质量、外观绑定和 rest surface；各 episode 仅用自己的首帧表面配准出初始位置与形变梯度。它不是从 RGB 自动重建首帧的入口。

`actions.npz` 字段：

- `poses[T,G,2,4,4]`：每个夹爪两侧的世界坐标变换。
- `valid[T,G,2]`：原始观测有效性；缺失位姿保留 NaN。
- `source_frames[T]`：从 0 连续计数。
- `source_timestamps_us[T]`：严格递增的微秒时间戳。

`build_queries.py` 复制输入并生成来源哈希与接触参考帧。接触仅在相邻两帧都有效时启用，不跨缺失区间外推。EEF+opening 的神经网络动作格式不能直接替代此处的双指位姿。

## 工作目录与复用

构建后产生 `asset/asset.json`、`asset/body.npz`、`asset/appearance.npz`、固定外观绑定、`initial_states/episode_N.npz` 及对应 JSON，另有 `queries/episode_N/actions.npz` 与 `episode.json`。资产或初态已存在时，构建程序会拒绝覆盖。

`PhysicalAsset` 提供现有资产加载、刚体放置及仅初帧表面配准 API。`run_episode.py --help` 列出六个物理设置和时间步选项；主模型仅优化其中四个，夹爪静/滑动摩擦固定。

确定性路径只实现前向传输，没有接入伴随梯度。拟合参数通过候选搜索更新。完整运行不使用后续物体几何重置状态。`--last-frame` 仅供显式诊断，不能当作完整物理评价。

这里是 EMPM 派生运行代码，包含项目接触和传输适配，受 [RAI Institute Research License](empm/LICENSE.md) 的非商业研究限制；仓库根目录 MIT 许可不覆盖第三方许可。
