# 代码导读

项目默认采用纯单元格准确率，通过配置选择任务、模型和仓库。模型必须完成真实抽取、加载和 dbt 转换，评分器读取实际仓库状态。历史奖励仅作为显式选择的可选基线保留。

| 模块 | 文件 | 职责 |
|---|---|---|
| 统一入口 | `src/elt_rl/cli.py` | 本地集成、任务准备、训练和独立评测 |
| 实验配置 | `framework_config.py`、`train.py` | 校验任务、模型、批次、采样和执行限制 |
| 任务契约 | `task_spec.py`、`task_adapters/` | 任务资产、源、目标类型、主键、官方版本与评分器 |
| 数据源 | `source_adapters.py`、`native_sources.py` | 用户权限服务、连接能力、真实读取和资源清理 |
| 仓库 | `warehouse_backends.py`、`warehouse.py` | 加载、类型、命名空间、查询、转换和权限隔离 |
| 环境和工具 | `environment.py`、`native_runtime.py` | 多轮执行、配置修复、提交冻结和状态一致性 |
| 模型 | `model_registry.py`、`model_runtime.py` | Cookbook tokenizer、renderer、工具调用和采样 |
| 官方评测 | `evaluation.py`、`official_grader.py` | 调用固定版本评分器并保存独立官方指标 |
| 执行奖励 | `rewards.py`、`execution_scoring.py` | 统计正确与全部目标单元格；保存可信预测、参考表和二元判断证据 |
| 训练调度 | `batch_protocol.py`、`cookbook_env.py` | 分组轨迹、Advantage、真实优化、评测和资源回收 |
| 断点恢复 | `batch_recovery.py`、`resume.py`、`lineage.py` | 防重复更新、部分评测恢复和 Checkpoint 证据链 |
| 结果输出 | `reporting.py`、`metrics.py` | 每条轨迹、训练曲线、独立评测和实际失败记录 |

默认奖励为 `正确目标单元格数 / 全部官方目标单元格数`。跨表采用 Micro Average；缺失行仍保留完整分母，数值按任务容忍规则二元判断。官方 SRDEL、SRDT 和完整任务成功率另行报告，不参与训练 Reward。历史奖励需要显式选择。评测轨迹不进入梯度，等分组不执行无效更新。

详细接口、输入输出及错误处理见 [接口契约](INTERFACES.md)，新增任务步骤见 [任务接入](ADDING_TASKS.md)。新增任务应提供真实资产、受支持的数据源、目标类型、可靠行对齐和对应评分器；每个任务的实际验证状态独立记录。

正式 cars 实验使用 0.5.2，human_resources 跨模型验证使用 0.5.8；最终交付代码为 0.6.1。各次运行保持原始实现指纹，私有归档保留对应 wheel 和原始记录，不用新版源码冒充旧实验版本。最终版本移除了配额预检、废弃 Terraform 执行分支和未调用的旧函数，保留有效测试、CI、全部已验证的历史 Reward 及必要复现工具。

公开 `evidence/` 只包含脱敏统计、图表、回执摘要及原文件哈希。真实预测表、原始 SQL、完整轨迹、云端 Checkpoint 地址和历史实验放在独立私有归档中。代码和文档审核通过后再发布 GitHub。

0.6.1 只修正官方 SRDT 的统计层：按完全正确的目标表数除以全部目标表数，保留分数并与完整 Pipeline 成功区分。Reward 与训练算法不变；历史结果离线重聚合，不改写原记录。正式交付只保留 cars Pure Cell 训练与 human_resources 跨模型验证，英文结果见 RESULTS.md。
