# AI Agent Instructions: Multimodal Feature Screening (MDFS) Framework

## 1. 项目核心目标 (Project Core)
本项目旨在复现并验证一种**多模态深度特征筛选方法 (MDFS)**。
该方法需支持 **有监督 (Supervised)** 和 **无监督 (Unsupervised)** 场景，核心能力是从多模态数据中筛选关键特征。

## 2. 任务模块划分 (Task Modules)
系统必须严格包含以下四个独立部分：
1.  **数值模拟 - 回归 (Regression Simulation)**：重点评估在连续变量预测中的**特征筛选能力**。
2.  **数值模拟 - 分类 (Classification Simulation)**：评估分类任务中的筛选与预测能力。
3.  **数值模拟 - 聚类 (Clustering Simulation)**：无监督环境下的特征筛选与聚类效果。
4.  **实例分析 (Real-world Instance)**：
    * 当前计划：一个多模态分类任务。
    * **预留接口**：必须预留清晰的数据加载接口 (`load_instance_data`)，目前可用模拟数据占位，但需方便后续替换为真实 CSV/Image 数据。

## 3. 数值模拟实验规范 (Simulation Protocol) - ⚠️ 重点执行
为了保证统计显著性，所有数值模拟必须遵循以下流程：

* **重复次数**：每个实验配置必须运行 **500次 (Repeats)**。
* **并行计算 (Parallel Execution)**：
    * **必须实现并行化**。禁止使用简单的单线程 `for` 循环串行跑500次。
    * 推荐使用 `joblib.Parallel` 或 `concurrent.futures.ProcessPoolExecutor`。
    * 设计模式：`run_simulation_suite` 函数应分发任务，每个任务独立完成 "生成数据 -> 训练所有方法 -> 评估" 的全过程。
* **对比实验 (Benchmarking)**：
    * 代码结构必须支持 **"Proposed Method" vs "Baseline Methods"**。
    * 每次模拟需在**同一份生成数据**上运行所有对比方法，确保公平性。
* **结果汇总**：最终输出应包含指标的 **均值 (Mean)** 和 **标准差 (Std)**。

## 4. 技术栈与实现约束 (Technical Constraints)
* **框架**：PyTorch (用于深度模型), scikit-learn (用于对比方法和聚类指标).
* **数据生成**：必须模块化，确保每次 repeat 生成的数据分布一致但样本随机。
* **代码结构建议**：
    * `src/simulations/`: 存放并行控制代码。
    * `src/methods/`: 存放 Proposed Method 和 Baselines 类。
    * `src/utils/`: 存放评估指标计算代码。

## 5. 交互与输出 (Output)
* 生成的代码必须包含详细的 **中文注释**。
* 在 `main.py` 或入口脚本中，提供一个开关变量 `DEBUG_MODE`。如果为 True，则只跑 2 次循环以测试代码通畅性；如果为 False，则跑全量 500 次并行。
