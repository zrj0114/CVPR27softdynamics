# 来源与许可证

- 仓库原有根目录 `LICENSE` 保持不变。
- `runtime/empm/` 为实验所使用的 EMPM solver 源码快照，来源运行清单记录 upstream commit `8e241057ca9ec6ad19c3fb7d75dc7407baba8de0`。保留 [RAI Institute Research License](runtime/empm/LICENSE.md)：仅限非商业研究使用，保留原版权、条件与免责声明。
- `runtime/` 的接触、确定性传输、资产与路径接口为项目适配；它们与上游快照共同构成修改后的研究运行时，不能视为未修改的官方 EMPM 发布。根目录 MIT 许可不解除上游研究限制。`softdynamics/adapters/empm.py` 也依赖这一运行时。
- `softdynamics/motion.py`、教师数值方法、预测头、轨迹评分及坐标搜索从当前 Deform360 实验中提取；新增了显式参数、命令行和缓存接口。核心提取文件的来源及提取前 SHA-256 见 [source_provenance.json](docs/source_provenance.json)。
- Deform360 数据、Gaussian 资产、动作、教师标签、训练权重和第三方下载文件不随本仓库分发。

方法参考：

- [ESFIM: Optimal Experimental Design for Reliable Learning of History-Dependent Constitutive Laws](https://arxiv.org/abs/2603.12365)：离线物理导数与信息矩阵代理。
- [Derivative-Informed Neural Operator](https://arxiv.org/abs/2206.10745)：导数监督学习。
- [Informationally Decoupled Trajectory Design](https://arxiv.org/abs/2610.10905)：有限差分与交互信息诊断。

当前代码没有声称完整复现上述方法；短窗 RMS 标签及按参数加权搜索是本项目的实验设计。
