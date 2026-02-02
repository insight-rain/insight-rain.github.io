# SpikySpace: A Spiking State Space Model for Energy-Efficient Time Series Forecasting

**相关性评分**: 6.0/10

**排名**: #83


---


## 基本信息

- **arXiv ID**: [2601.02411v1](https://arxiv.org/abs/2601.02411v1)
- **发布时间**: 2026-01-02T13:10:53Z
- **相关性评分**: 6.0/10
- **是否相关**: 是

## 作者

Kaiwen Tang, Jiaqi Zheng, Yuze Jin, Yupeng Qiu, Guangda Sun, Zhanglu Yan, Weng-Fai Wong

## 关键词

Inference Efficiency, Lightweight Architecture, Edge Deployment

## 一句话总结

SpikySpace是一种基于脉冲状态空间模型的能量高效时间序列预测方法，通过选择性扫描和稀疏脉冲训练优化推理效率，适用于边缘设备部署。

## 摘要

Time-series forecasting often operates under tight power and latency budgets in fields like traffic management, industrial condition monitoring, and on-device sensing. These applications frequently require near real-time responses and low energy consumption on edge devices. Spiking neural networks (SNNs) offer event-driven computation and ultra-low power by exploiting temporal sparsity and multiplication-free computation. Yet existing SNN-based time-series forecasters often inherit complex transformer blocks, thereby losing much of the efficiency benefit. To solve the problem, we propose SpikySpace, a spiking state-space model (SSM) that reduces the quadratic cost in the attention block to linear time via selective scanning. Further, we replace dense SSM updates with sparse spike trains and execute selective scans only on spike events, thereby avoiding dense multiplications while preserving the SSM's structured memory. Because complex operations such as exponentials and divisions are costly on neuromorphic chips, we introduce simplified approximations of SiLU and Softplus to enable a neuromorphic-friendly model architecture. In matched settings, SpikySpace reduces estimated energy consumption by 98.73% and 96.24% compared to two state-of-the-art transformer based approaches, namely iTransformer and iSpikformer, respectively. In standard time series forecasting datasets, SpikySpace delivers competitive accuracy while substantially reducing energy cost and memory traffic. As the first full spiking state-space model, SpikySpace bridges neuromorphic efficiency with modern sequence modeling, marking a practical and scalable path toward efficient time series forecasting systems.

## 详细分析

## 论文摘要：SpikySpace: 一种用于高效能时间序列预测的脉冲状态空间模型

### 1. 研究背景和动机
在交通管理、工业状态监测和边缘设备传感等领域，时间序列预测通常需要在严格的功耗和延迟预算下进行。现有基于深度学习的预测模型（如Transformer）虽然精度高，但计算能耗巨大，难以在边缘设备上部署。脉冲神经网络（SNN）因其事件驱动、无乘法运算的特性，被视为实现超低功耗边缘智能的潜在方案。然而，现有的SNN预测器往往继承了Transformer的复杂结构，未能充分发挥SNN的能效优势。因此，亟需一种既能保持竞争力预测精度，又能实现极致能效的新型模型。

### 2. 核心方法和技术创新
本文提出了 **SpikySpace**，这是首个**完全脉冲化的状态空间模型**。其核心创新在于：
- **脉冲驱动的状态空间架构**：将状态空间模型（SSM）的线性时间复杂度和长程依赖建模能力，与SNN的稀疏、事件驱动计算相结合。通过**选择性扫描**机制，仅在脉冲事件发生时更新状态，避免了密集的矩阵乘法。
- **硬件友好的激活函数近似**：针对SSM中复杂且耗能的SiLU和Softplus函数，提出了**PTSoftplus**和**PTSiLU**两种近似函数。它们使用基于2的幂次运算和线性变换，可在神经形态硬件上通过高效的加法与位移操作实现，移除了昂贵的指数和除法运算。
- **高效的训练与转换流程**：采用**ANN-to-SNN转换**策略，首先在连续域训练一个使用上述近似函数的量化ANN，然后确定性地将其转换为完全由脉冲事件驱动的SNN模型。

### 3. 主要实验结果
在四个标准多元时间序列预测数据集（Metr-la, Pems-bay, Solar, Electricity）上的实验表明：
- **预测精度**：SpikySpace达到了与最先进Transformer模型（iTransformer）和SNN模型（iSpikformer, SpikeSTAG）相当或更优的精度（例如，在Electricity数据集上R²达0.992）。
- **能效提升**：与iTransformer和iSpikformer相比，SpikySpace的**估计能耗分别降低了98.73%和96.24%**，同时模型参数量减少了约45-47%。
- **消融研究**：验证了PTSoftplus和PTSiLU在极低比特量化下（如1-bit）仍能保持接近原函数的性能，且模型对脉冲时间步长选择具有鲁棒性。

### 4. 研究意义和价值
SpikySpace首次将现代序列建模的状态空间模型与神经形态计算的能效优势成功桥接，为在资源受限的边缘设备上部署高效、实时的时序预测系统提供了一条切实可行的技术路径。这项工作不仅推动了脉冲神经网络在复杂序列任务中的应用边界，也为设计下一代超低功耗AI芯片的算法-硬件协同优化提供了重要参考。

## 问答对

### 问题 1

**Q**: 这篇论文的核心创新点是什么，他想解决什么问题，怎么解决的？

**A**: ## 论文核心分析：SpikySpace

### **一、 论文旨在解决的核心问题**
论文瞄准**边缘设备上时间序列预测任务面临的严峻能效挑战**。具体问题包括：
1.  **高能耗与低延迟需求的矛盾**：在交通管理、工业监测等场景，模型需在资源受限的边缘设备上实现**近实时预测**，但现有高性能模型（如Transformer、RNN）因密集的乘法和指数运算而能耗过高。
2.  **现有脉冲神经网络模型的效率瓶颈**：尽管SNN具有事件驱动、超低功耗的潜力，但当前基于SNN的时间序列预测器（如SpikeTCN, iSpikformer）大多**沿用了Transformer的复杂结构**，其注意力机制仍存在二次复杂度，未能充分发挥SNN的能效优势。
3.  **状态空间模型与脉冲计算范式不兼容**：SSM（如Mamba）虽能以线性复杂度建模长序列，但其**连续状态更新和复杂激活函数**与SNN的离散、稀疏脉冲事件驱动特性不匹配，难以直接用于构建全脉冲SSM。

### **二、 核心创新点**
SpikySpace 提出了**首个全脉冲状态空间模型**，通过三大创新实现能效与精度的平衡：

#### **1. 架构创新：脉冲驱动的选择性扫描**
- **核心机制**：将SSM的连续状态更新改造为**事件驱动的脉冲更新**。仅在输入脉冲事件发生时执行“选择性扫描”进行状态更新，避免了传统SSM每一步的密集矩阵乘法。
- **关键改进**：
    - **线性复杂度**：利用SSM的选择性扫描机制，将计算复杂度从Transformer的 `O(L²)` 降至 `O(L)`。
    - **稀疏计算**：状态 `h_t` 和输出 `y_t` 均通过脉冲神经元生成，使计算量和内存访问与脉冲发放率成正比，而非序列长度。

#### **2. 算法创新：神经形态硬件友好的激活函数近似**
为解决SSM中关键但计算昂贵的 **Softplus** 和 **SiLU** 函数在神经形态芯片上实现的难题，提出了两种高效的近似函数：
- **PTSoftplus** 与 **PTSiLU**：
    - **设计原则**：使用**分段函数**，结合**2的幂次方**和线性变换来近似原函数。
    - **硬件优势**：2的幂次方运算可通过高效的**位移操作**实现，彻底消除了原函数中对能耗高的指数、对数和除法运算。
    - **理论保证**：论文严格证明了这两个近似函数是**连续可微的**，且与原函数的偏差有界（函数值偏差≤0.914/0.316，导数偏差≤0.371/0.263），确保了训练稳定性。

#### **3. 训练与部署流程创新**
- **两步法训练**：
    1.  **训练量化ANN**：使用提出的 PTSoftplus 和 PTSiLU 作为激活函数，在低比特约束下训练一个ANN。
    2.  **确定性ANN-to-SNN转换**：将训练好的量化ANN中的激活值、门控和状态更新**映射为脉冲事件**，得到最终的SpikySpace模型。
- **能效优化技巧**：通过调整脉冲发放阈值和权重缩放，将脉冲发放率从~93%显著降低至~26.8%，进一步减少计算操作。

### **三、 解决方案的总体思路**
论文的解决方案可以概括为：**用线性复杂度的SSM替代二次复杂度的注意力机制，并将其彻底“脉冲化”**，同时辅以硬件友好的算法近似，最终实现一个**既保持强大序列建模能力，又具备极低能耗**的预测模型。

1.  **模型框架**：采用类Mamba的残差结构，核心是 **SpikingMamba模块**，内部包含局部卷积、脉冲SSM块和选择性扫描。
2.  **工作流程**：输入通过脉冲神经元编码为稀疏脉冲序列 → 在SSM块中，参数（Δ, B, C）也由脉冲生成 → 执行**脉冲驱动的选择性扫描**进行状态更新 → 输出与门控残差结合产生最终预测。
3.  **硬件映射**：所有核心运算（近似激活、2的幂次方状态转移）均可通过加法、位移和查找表实现，与Intel Loihi等神经形态硬件高度兼容。

### **四、 实际价值与效果**
- **卓越的能效**：在相同设置下，相比iTransformer和iSpikformer，估计能耗分别降低 **98.73%** 和 **96.24%**。模型参数量也减少约45%。
- **竞争力的精度**：在Metr-la、Electricity等多个标准时间序列数据集上，取得了与最先进SNN和ANN模型**相当或更优**的预测精度（`R²` 分数）。
- **部署可行性**：为在严格功耗和延迟预算下的边缘实时预测（如车载传感器、智能电网）提供了一条切实可行的技术路径，**弥合了现代序列建模与神经形态计算效率之间的鸿沟**。

**总结**：SpikySpace 的核心贡献在于**创造性地融合了SSM的长程建模能力和SNN的极致能效特性**，并通过硬件感知的算法设计解决了关键兼容性问题，为边缘时间序列分析提供了一个**高精度、超低功耗**的实用化解决方案。


### 问题 2

**Q**: 请用一段话总结这篇论文，明确说明：论文试图解决的核心问题；提出的主要方法或框架；最终取得的主要效果或结论。要求语言简洁、信息密度高，不要复述摘要原文。

**A**: 这篇论文旨在解决边缘设备上时间序列预测任务中**高能耗与低延迟需求**与现有高性能模型（尤其是基于Transformer的SNN模型）**计算复杂度高、能耗大**之间的矛盾。为此，论文提出了 **SpikySpace**，这是首个**完全脉冲驱动的状态空间模型**。其核心创新在于：1）将线性复杂度的选择性扫描（Selective Scan）机制与脉冲神经网络的**事件驱动、稀疏计算**特性相结合，用脉冲序列替代密集的SSM状态更新；2）设计了硬件友好的**PTSoftplus和PTSiLU激活函数近似**，用幂次运算和线性变换替代昂贵的指数、除法运算，使其可通过高效的移位操作实现。最终，模型在多个时间序列预测基准测试中达到了与先进Transformer和SNN模型**相竞争的精度**，同时**大幅降低了能耗**（相比iTransformer和iSpikformer分别节能约98.73%和96.24%），为在神经形态硬件和边缘设备上部署高效的时间序列预测系统提供了一条可行路径。


### 问题 3

**Q**: 这篇论文相对于已有工作有哪些明确的创新点？请逐条列出，并对每一条说明：相比以往方法改进或不同之处在哪里，以及该创新解决了什么具体问题或带来了什么优势。

**A**: ## SpikySpace 论文创新点分析

这篇论文提出了一种名为 **SpikySpace** 的完全脉冲状态空间模型，旨在解决边缘设备上时间序列预测任务对高能效和低延迟的需求。其核心创新点在于将现代状态空间模型的高效序列建模能力与脉冲神经网络的超低功耗特性相结合。以下是其相对于已有工作的明确创新点：

### 1. **首次提出完全脉冲化的状态空间模型**
   - **改进/不同之处**： 以往将SNN与SSM结合的工作（如SpikingSSMs, P-SpikeSSM）通常只在部分模块使用脉冲，或者将SNN作为SSM的包装器/特征提取器，其状态更新仍依赖密集的连续值乘法和复杂激活函数（如Softplus, SiLU）。SpikySpace是**第一个**实现**完全脉冲化**的SSM，即模型的**潜在状态更新和非线性变换均通过离散的脉冲驱动计算**完成。
   - **解决的问题/优势**： 这从根本上解决了现有“混合”方法无法充分利用SNN事件驱动稀疏计算优势的问题。通过将SSM的连续状态演化转化为基于脉冲的选择性扫描，SpikySpace实现了计算与信息内容（即脉冲事件）而非序列长度直接挂钩，从而在理论上获得了SNN的极致能效，同时保留了SSM处理长程依赖的能力。

### 2. **提出针对神经形态硬件的轻量级激活函数近似：PTSoftplus 和 PTSiLU**
   - **改进/不同之处**： 标准的Softplus (`ln(1+e^x)`)和SiLU (`x * sigmoid(x)`)函数包含指数、对数和除法运算，在神经形态芯片上实现成本极高。论文创新性地提出了**基于2的幂次方的分段近似函数PTSoftplus和PTSiLU**。这些近似函数由幂次方运算和线性变换构成，例如PTSoftplus在`x < x_c`时输出`2^x`，否则输出`x + C`。
   - **解决的问题/优势**：
     - **硬件友好性**： 幂次方运算在硬件上可通过高效的**位移操作**实现，彻底消除了昂贵的指数和对数运算。
     - **保持性能**： 论文通过理论证明（连续可微）和实验验证，表明这些近似函数与原函数非常接近（函数值最大偏差≤0.914，导数最大偏差≤0.371），在量化至低比特（如1-bit）时仍能保持模型精度，解决了复杂激活函数阻碍SSM类SNN在能效约束系统上部署的关键障碍。

### 3. **设计脉冲驱动的选择性扫描模块**
   - **改进/不同之处**： 借鉴了Mamba模型中的选择性扫描机制（根据输入动态调整参数Δ），但对其进行了彻底的脉冲化改造。关键改进包括：
     1.  **脉冲化参数生成**： 将决定状态更新频率的步长参数Δ也编码为脉冲序列。
     2.  **幂次方近似状态转移**： 将状态转移矩阵 `Ā = exp(Δt A)` 近似为 `Ā = 2^[round(Δt A)]`，用位移操作替代指数运算。
     3.  **脉冲化状态更新**： 在选择性扫描的每一步，状态`h_t`和输出`y_t`都通过脉冲神经元(`SN`)生成脉冲，确保整个数据流是稀疏的二进制事件。
   - **解决的问题/优势**：
     - **线性复杂度与稀疏性结合**： 继承了SSM的`O(L)`线性序列建模复杂度，同时通过脉冲实现了计算稀疏性。计算和内存访问仅发生在有脉冲的时刻，且与脉冲率成正比，而非固定的时间步长。
     - **结构化记忆与事件驱动执行**： SSM的潜在状态`h_t`提供了显式的、紧凑的长期记忆结构，而脉冲驱动确保了只有信息量大的输入（脉冲）才会触发状态更新和计算，实现了“按需计算”。

### 4. **采用结合量化的ANN-to-SNN转换训练流程**
   - **改进/不同之处**： 训练策略上，并非直接训练SNN，而是先训练一个**低比特量化的人工神经网络**，其中使用了提出的PTSoftplus和PTSiLU作为激活函数。然后，将这个量化ANN确定性地转换为SNN。
   - **解决的问题/优势**：
     - **训练稳定性与性能**： 避免了直接使用替代梯度法训练深层SNN可能遇到的梯度不稳定和性能损失问题。利用成熟的ANN训练技术获得高性能的初始模型。
     - **硬件效率对齐**： 由于ANN在训练时已进行了低比特量化并使用了硬件友好的激活函数，转换得到的SNN天然与目标硬件（支持位移和累加操作）的计算特性高度匹配，确保了仿真评估的能量优势能够在实际部署中实现。

### 总结的创新价值
这些创新点共同作用，使得SpikySpace在**能效**上实现了**突破性提升**（论文报告相比iTransformer和iSpikformer分别节能98.73%和96.24%），同时在**预测精度**上达到或超越了现有的先进SNN方法（如SpikeSTAG）和强大的ANN基线（如iTransformer）。它**首次**成功地将SSM的**强大序列建模能力**与SNN的**极致硬件能效潜力**在一个**端到端的全脉冲框架**内统一起来，为在严格功耗和延迟约束下的边缘时间序列分析提供了一条切实可行的技术路径。


### 问题 4

**Q**: 论文在实验或评估中最终实现了怎样的效果？请说明使用了哪些数据集和评价指标，与哪些基线方法进行了对比，以及在关键指标上的主要性能提升或结论。如果论文未给出明确的定量结果，也请说明原因。

**A**: ## 论文实验评估效果总结

### 一、 使用的数据集
论文在四个广泛使用的多元时间序列预测基准数据集上进行了评估，涵盖了交通、太阳能和电力消耗等不同领域。具体信息如下：

| 数据集 | 样本数 | 变量数 | 采样间隔 | 序列长度 | 训练/验证/测试划分比例 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **METR-LA** | 34,272 | 207 | 5分钟 | 12 | (0.7, 0.2, 0.1) |
| **PEMS-BAY** | 52,116 | 325 | 5分钟 | 12 | (0.7, 0.2, 0.1) |
| **Solar-Energy** | 52,560 | 137 | 10分钟 | 168 | (0.6, 0.2, 0.2) |
| **Electricity** | 26,304 | 321 | 1小时 | 168 | (0.6, 0.2, 0.2) |

### 二、 评价指标
论文采用以下两个**尺度无关**的指标，便于在不同量级的数据集间进行比较：
1.  **决定系数 (R²)**：衡量模型对数据方差的解释能力。**值越高越好**，最大为1。
    ```math
    R^{2}=1-\frac{\sum_{t}(y_{t}-\hat{y}_{t})^{2}}{\sum_{t}(y_{t}-\bar{y})^{2}}
    ```
2.  **根相对平方误差 (RRSE)**：衡量预测误差相对于数据均方差的平方根。**值越低越好**。
    ```math
    \mathrm{RRSE}=\sqrt{\frac{\sum_{t}(y_{t}-\hat{y}_{t})^{2}}{\sum_{t}(y_{t}-\bar{y})^{2}}}
    ```

### 三、 对比的基线方法
论文与以下方法进行了对比，包括传统ANN和先进的SNN模型：
- **ANN基线**:
    - **GRU**: 经典的循环神经网络，作为基础序列模型。
    - **iTransformer**: 最先进的Transformer变体，代表全精度ANN的高性能。
- **SNN基线**:
    - **SpikeTCN**: 基于时间卷积的SNN。
    - **SpikeRNN**: 基于循环单元的SNN。
    - **iSpikformer**: 基于Transformer架构的SNN。
    - **SpikeSTAG**: 结合图神经网络的时空SNN模型。

### 四、 关键性能与结论

#### 1. 预测准确性
- **总体表现**：SpikySpace在四个数据集上取得了**具有竞争力或更优的预测精度**。
- **具体结果**（基于平均R²）：
    - **Metr-la**: SpikySpace达到 **0.778**，优于所有SNN基线（如SpikeSTAG的0.755），并超越了ANN基线GRU（0.715）和iTransformer（0.754）。
    - **Electricity**: SpikySpace达到 **0.992**，是所有模型中最好的，显著优于iSpikformer（0.976）和iTransformer（0.978）。
    - **Solar和Pems-bay**: SpikySpace表现与最佳基线模型相当或接近。
- **结论**：SpikySpace证明了其**在保持高精度的同时，实现了完全脉冲化计算**，弥合了SNN与密集ANN之间的精度差距。

#### 2. 能源效率（核心创新价值）
论文进行了详细的能量分析，结论极为突出：
- **对比基准**：在Electricity数据集（预测步长=3）上，与最先进的ANN和SNN模型比较。
- **能量消耗**：
    - SpikySpace: **0.12 mJ**
    - iTransformer (ANN): 9.47 mJ
    - iSpikformer (SNN): 3.19 mJ
    - SpikeSTAG (SNN): 4.39 mJ
- **能量节省**：
    - 相比 **iTransformer**，能耗降低 **98.73%**。
    - 相比 **iSpikformer**，能耗降低 **96.24%**。
- **模型大小**：SpikySpace参数量仅为0.868M，约为对比SNN基线（1.566M-1.634M）的**53%-55%**，模型更加轻量。

#### 3. 消融研究结论
- **激活函数近似（PTSoftplus/PTSiLU）**：在4比特甚至1比特量化下，使用近似函数与原始Softplus/SiLU相比，精度损失极小（ΔR² < 0.01），证明了其有效性且对硬件友好。
- **时间步长（Timestep）**：实验表明，在`T=1, 3, 7, 15`不同设置下，模型性能总体稳定。最终选择`T=3`在精度和效率间取得了良好平衡。

### 五、 核心结论
SpikySpace通过将状态空间模型（SSM）完全脉冲化，实现了：
1.  **精度与效率的卓越平衡**：在多个时间序列预测任务上达到与顶尖ANN/SNN模型相当的精度。
2.  **革命性的能效提升**：相比现有Transformer类SNN（iSpikformer）和ANN（iTransformer），能耗降低了**1-2个数量级**。
3.  **硬件部署潜力**：提出的PTSoftplus和PTSiLU激活函数近似，以及事件驱动的选择性扫描机制，使其非常适合在**神经形态芯片或边缘设备**上部署。

**最终效果**：SpikySpace成功地将现代序列建模的强大能力与神经形态计算的超高效能相结合，为资源严格受限的边缘场景下的实时时间序列预测，提供了一条切实可行且可扩展的技术路径。


## 相关链接

- [arXiv 页面](https://arxiv.org/abs/2601.02411v1)
- [HTML 版本](https://arxiv.org/html/2601.02411v1)
