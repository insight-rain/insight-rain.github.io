# MiMo-V2-Flash Technical Report

**相关性评分**: 6.0/10

**排名**: #71


---


## 基本信息

- **arXiv ID**: [2601.02780v2](https://arxiv.org/abs/2601.02780v2)
- **发布时间**: 2026-01-06T07:31:47Z
- **相关性评分**: 6.0/10
- **是否相关**: 是

## 作者

Xiaomi LLM-Core Team, :, Bangjun Xiao, Bingquan Xia, Bo Yang, Bofei Gao, Bowen Shen, Chen Zhang, Chenhong He, Chiheng Lou, Fuli Luo, Gang Wang, Gang Xie, Hailin Zhang, Hanglong Lv, Hanyu Li, Heyu Chen, Hongshen Xu, Houbin Zhang, Huaqiu Liu, Jiangshan Duo, Jianyu Wei, Jiebao Xiao, Jinhao Dong, Jun Shi, Junhao Hu, Kainan Bao, Kang Zhou, Lei Li, Liang Zhao, Linghao Zhang, Peidian Li, Qianli Chen, Shaohui Liu, Shihua Yu, Shijie Cao, Shimao Chen, Shouqiu Yu, Shuo Liu, Tianling Zhou, Weijiang Su, Weikun Wang, Wenhan Ma, Xiangwei Deng, Bohan Mao, Bowen Ye, Can Cai, Chenghua Wang, Chengxuan Zhu, Chong Ma, Chun Chen, Chunan Li, Dawei Zhu, Deshan Xiao, Dong Zhang, Duo Zhang, Fangyue Liu, Feiyu Yang, Fengyuan Shi, Guoan Wang, Hao Tian, Hao Wu, Heng Qu, Hongfei Yi, Hongxu An, Hongyi Guan, Xing Zhang, Yifan Song, Yihan Yan, Yihao Zhao, Yingchun Lai, Yizhao Gao, Yu Cheng, Yuanyuan Tian, Yudong Wang, Zhen Tang, Zhengju Tang, Zhengtao Wen, Zhichao Song, Zhixian Zheng, Zihan Jiang, Jian Wen, Jiarui Sun, Jiawei Li, Jinlong Xue, Jun Xia, Kai Fang, Menghang Zhu, Nuo Chen, Qian Tu, Qihao Zhang, Qiying Wang, Rang Li, Rui Ma, Shaolei Zhang, Shengfan Wang, Shicheng Li, Shuhao Gu, Shuhuai Ren, Sirui Deng, Tao Guo, Tianyang Lu, Weiji Zhuang, Weikang Zhang, Weimin Xiong, Wenshan Huang, Wenyu Yang, Xin Zhang, Xing Yong, Xu Wang, Xueyang Xie, Yilin Jiang, Yixin Yang, Yongzhe He, Yu Tu, Yuanliang Dong, Yuchen Liu, Yue Ma, Yue Yu, Yuxing Xiang, Zhaojun Huang, Zhenru Lin, Zhipeng Xu, Zhiyang Chen, Zhonghua Deng, Zihan Zhang, Zihao Yue

## 关键词

Inference Efficiency, Lightweight Architecture, Inference Acceleration

## 一句话总结

MiMo-V2-Flash 是一个混合专家模型，通过混合注意力架构和推测解码技术，在推理效率和轻量化架构方面表现出色，但与视觉-语言-动作模型和机器人应用无直接关联。

## 摘要

We present MiMo-V2-Flash, a Mixture-of-Experts (MoE) model with 309B total parameters and 15B active parameters, designed for fast, strong reasoning and agentic capabilities. MiMo-V2-Flash adopts a hybrid attention architecture that interleaves Sliding Window Attention (SWA) with global attention, with a 128-token sliding window under a 5:1 hybrid ratio. The model is pre-trained on 27 trillion tokens with Multi-Token Prediction (MTP), employing a native 32k context length and subsequently extended to 256k. To efficiently scale post-training compute, MiMo-V2-Flash introduces a novel Multi-Teacher On-Policy Distillation (MOPD) paradigm. In this framework, domain-specialized teachers (e.g., trained via large-scale reinforcement learning) provide dense and token-level reward, enabling the student model to perfectly master teacher expertise. MiMo-V2-Flash rivals top-tier open-weight models such as DeepSeek-V3.2 and Kimi-K2, despite using only 1/2 and 1/3 of their total parameters, respectively. During inference, by repurposing MTP as a draft model for speculative decoding, MiMo-V2-Flash achieves up to 3.6 acceptance length and 2.6x decoding speedup with three MTP layers. We open-source both the model weights and the three-layer MTP weights to foster open research and community collaboration.

## 详细分析

## 论文摘要：MiMo-V2-Flash 技术报告

**1. 研究背景和动机**
当前，推动人工智能向通用智能（AGI）发展的两大前沿是高级推理链和自主智能体工作流，其基础是大规模强化学习。然而，构建可扩展的推理器和智能体面临一个共同的关键瓶颈：长上下文建模必须同时具备**快速**和**强大**的能力。为此，小米LLM-Core团队提出了MiMo-V2-Flash，旨在设计一个高效、经济的大语言模型，以提供强大的推理和智能体性能。

**2. 核心方法和技术创新**
MiMo-V2-Flash是一个总参数量为309B、激活参数量为15B的**混合专家模型**，其核心技术创新体现在三个方面：
- **混合注意力架构**：采用滑动窗口注意力与全局注意力交错的结构（5:1混合比，128词元窗口），结合可学习的注意力汇聚偏置，在保证长上下文建模能力的同时，显著降低了KV缓存和计算开销。
- **轻量级多词元预测**：将MTP同时作为训练目标和**推测解码的草稿模型**，在推理时通过生成多个候选词元进行并行验证，实现了高达2.6倍的解码加速。
- **新颖的后训练范式**：提出了**多教师策略蒸馏**，通过让学生模型从多个领域专家教师处获取密集的词元级奖励，高效地整合了不同领域的峰值能力，解决了传统后训练中能力不平衡和学习效率低下的问题。

**3. 主要实验结果**
- **性能对标**：在多数推理基准测试上，性能媲美参数量为其2-3倍的顶级开源模型（如DeepSeek-V3.2、Kimi-K2）。
- **长上下文能力**：在256K长度内，检索成功率接近100%；在极端长上下文推理基准GSM-Infinite上表现稳健。
- **智能体能力**：在SWE-Bench Verified上达到73.4%，成为开源模型中软件工程任务的领先者。
- **推理加速**：使用3层MTP进行推测解码，在批处理场景下可实现最高**2.7倍**的解码速度提升。

**4. 研究意义和价值**
该工作展示了一种通过**高效的模型架构设计**与**创新的训练范式**，以更少参数量实现顶尖性能的可行路径。其开源的模型权重与MTP权重将促进开源研究和社区协作。所提出的MOPD范式为高效整合多领域专家知识、持续扩展模型能力提供了新的框架，对推动高效、强推理的智能体模型发展具有重要价值。

## 问答对

### 问题 1

**Q**: 这篇论文的核心创新点是什么，他想解决什么问题，怎么解决的？

**A**: ## 论文核心分析：MiMo-V2-Flash

### **一、 论文旨在解决的核心问题**
论文瞄准当前大语言模型（LLM）迈向通用人工智能（AGI）的两个关键瓶颈：
1.  **效率与性能的权衡**：如何构建一个既能处理**长上下文**（支持复杂推理和智能体工作流），又能在**推理时保持高速**的模型。传统全局注意力（GA）计算和内存开销巨大。
2.  **后训练（Post-Training）的扩展难题**：如何高效地通过强化学习（RL）等后训练方法规模化地提升模型的**推理和智能体能力**，同时避免常见的“能力失衡”（提升某一能力导致其他能力衰退）和“学习低效”问题。

### **二、 核心技术创新点**
论文通过三大核心技术模块系统性解决上述问题：

#### **1. 混合滑动窗口注意力架构**
*   **问题**：全注意力（Full Attention）的二次方复杂度是长上下文建模的主要瓶颈。
*   **创新解决方案**：
    *   **混合设计**：模型主体由 **5个滑动窗口注意力块 + 1个全局注意力块** 交替堆叠（共8个这样的混合块）。滑动窗口大小仅为 **128个token**，混合比例为 **5:1**。
    *   **关键技术**：引入了 **可学习的注意力汇聚偏置**。该机制允许模型在需要时“忽略”某些token的注意力，显著增强了基于SWA架构的建模能力。
    *   **效果**：在长上下文场景下，实现了近 **6倍的KV缓存存储和注意力计算量减少**，同时性能**超越或媲美全注意力基线模型**。实验表明，小窗口（128）比大窗口（512）效果更好，形成了更清晰的“局部建模-全局依赖”分工。

#### **2. 轻量级多令牌预测**
*   **问题**：LLM解码本质是内存瓶颈，且RL训练中的推演阶段是主要耗时瓶颈。
*   **创新解决方案**：
    *   **双重角色**：MTP不仅作为**训练目标**提升模型质量，更被设计为**原生草稿模型**，用于推测解码。
    *   **轻量设计**：MTP块使用**密集前馈网络**（而非MoE）和**滑动窗口注意力**（而非全局注意力），参数量仅0.33B，避免成为新瓶颈。
    *   **效果**：
        *   **训练**：加速RL训练，使小批量、同策略训练更可行，缓解长尾任务造成的GPU闲置。
        *   **推理**：作为草稿模型，在3层MTP下，实现了最高 **3.6的接受长度** 和 **最高2.6倍的解码加速**。

#### **3. 多教师同策略蒸馏后训练范式**
*   **问题**：传统后训练存在“能力失衡”（跷跷板效应）和“学习低效”问题，难以整合多个领域专家的能力。
*   **创新解决方案**：
    *   **三阶段范式**：
        1.  **监督微调**：建立基础的指令跟随能力。
        2.  **领域专家训练**：通过独立的RL训练，得到多个在特定领域（如代码、数学、搜索）表现卓越的**教师模型**。
        3.  **MOPD**：**学生模型**通过**同策略RL**，从多个教师模型处获得**密集的、令牌级别的奖励信号**（基于KL散度），同时也可结合传统的结果奖励模型。
    *   **核心优势**：
        *   **高效且有效**：通过令牌级奖励实现稳定的信用分配和快速收敛，避免了离策略方法中的分布不匹配问题。
        *   **保留峰值能力**：学生模型能够**掌握最强教师在其领域的峰值性能**，而非性能折中。
        *   **模块化与可扩展**：教师模型可以灵活替换或增加，支持师生协同进化的迭代循环。

### **三、 实际价值与成果**
1.  **高性能与高效率模型**：MiMo-V2-Flash以 **309B总参数、15B激活参数** 的较小规模，在多项基准测试中达到了与总参数量为其2-3倍的顶级开源模型（如DeepSeek-V3.2, Kimi-K2）相媲美的性能。
2.  **卓越的长上下文与代码能力**：
    *   在长上下文检索（NIAH-Multi）上达到近100%成功率（32K-256K）。
    *   在**SWE-Bench Verified**上达到 **73.4%**，在**SWE-Bench Multilingual**上达到 **71.7%**，成为开源模型中软件工程任务的领先者。
3.  **开源与社区贡献**：论文开源了**模型权重**和**三层MTP权重**，促进了高效模型架构和训练范式的开放研究。
4.  **为AGI发展提供路径**：通过高效的架构设计（混合注意力）和创新的训练范式（MOPD），为解决构建**快速、强推理、自主的智能体**这一AGI核心挑战提供了可行的技术方案。

### **总结**
MiMo-V2-Flash的核心创新在于**系统性地整合了模型架构、训练目标和后训练范式的改进**，旨在打破大模型在**长上下文、强推理、高速解码**三者不可兼得的传统困局。其**混合滑动窗口注意力**解决了计算效率问题，**轻量级MTP**解决了训练和推理速度问题，而**MOPD范式**则解决了后训练阶段能力规模化与整合的难题。这三者共同作用，产出了一个在参数效率、推理速度和综合能力上都具有竞争力的先进模型。


### 问题 2

**Q**: 请用一段话总结这篇论文，明确说明：论文试图解决的核心问题；提出的主要方法或框架；最终取得的主要效果或结论。要求语言简洁、信息密度高，不要复述摘要原文。

**A**: ## 论文总结

这篇技术报告旨在解决构建兼具**强大推理/智能体能力**与**高效长上下文处理**的大语言模型所面临的瓶颈问题。为此，论文提出了**MiMo-V2-Flash**模型，其核心创新在于一个**混合注意力架构**（交错使用128词元的滑动窗口注意力与全局注意力，比例为5:1）以降低计算开销，并引入了一种新颖的**多教师策略蒸馏**后训练范式。MOPD框架通过让学生模型从多个领域专家（如通过大规模强化学习训练）获取密集的、词元级的奖励信号，高效地整合了各领域的峰值能力，同时避免了传统方法中的能力失衡与学习低效问题。最终，该模型仅以309B总参数（15B激活参数）的较小规模，在多项推理、代码和长上下文基准测试中达到了与DeepSeek-V3.2、Kimi-K2等更大规模顶尖开源模型相媲美甚至超越的性能，并在SWE-Bench等软件工程任务上确立了开源模型的领先地位，同时通过复用多词元预测模块进行推测解码，实现了最高**2.6倍的解码加速**。


### 问题 3

**Q**: 这篇论文相对于已有工作有哪些明确的创新点？请逐条列出，并对每一条说明：相比以往方法改进或不同之处在哪里，以及该创新解决了什么具体问题或带来了什么优势。

**A**: ## 论文核心创新点分析

这篇技术报告介绍了小米推出的MiMo-V2-Flash模型，它是一个在架构设计、训练范式和推理优化上均有显著创新的混合专家模型。其创新点明确且具有实际价值，主要体现在以下几个方面：

### 1. **创新的混合注意力架构：结合滑动窗口与全局注意力**
- **改进/不同之处**：
    - 采用了一种**5:1的混合比例**（5层滑动窗口注意力 + 1层全局注意力），并使用了**极小的滑动窗口（128个token）**。这与以往许多混合注意力模型（如Gemma 2）不同，后者通常使用更大的窗口（如4096）或更保守的混合比例。
    - 在设计中**集成了可学习的注意力汇聚偏置**，允许模型在需要时“忽略”某些token，从而缓解了因使用激进的小窗口而可能导致的性能下降。
- **解决的问题/带来的优势**：
    - **核心问题**：解决长上下文建模中**计算和内存的二次方复杂度瓶颈**。
    - **具体优势**：
        1.  **高效性**：相比全注意力，KV缓存存储和注意力计算减少了近**6倍**，实现了**快速且强大的长上下文建模**。
        2.  **性能提升**：实验表明，这种激进的设计（小窗口+汇聚偏置）不仅没有降低性能，反而在多项基准测试（如复杂推理、长上下文理解）上**超越了全注意力基线**。论文假设这得益于更清晰的“分工”（局部信息由SWA处理，长程依赖由GA处理）和更好的正则化效果。

### 2. **新颖的后训练范式：多教师策略蒸馏**
- **改进/不同之处**：
    - 提出了 **“多教师策略蒸馏”** 框架。传统方法（如参数合并、顺序训练）常导致能力失衡（跷跷板效应）。MOPD则将多教师知识整合**建模为一个策略强化学习过程**。
    - **关键机制**：学生模型从自身策略分布中采样，接收来自**领域专业化教师模型**的**密集的、token级别的奖励**（通过KL散度计算），同时也可结合基于结果的奖励。
- **解决的问题/带来的优势**：
    - **核心问题**：解决后训练中**能力失衡**和**学习效率低下**的问题。
    - **具体优势**：
        1.  **高效且有效**：通过策略学习，学生模型能够**同时掌握多个领域教师（如数学RL模型、代码RL模型）的峰值能力**，避免了能力倒退。表7显示，学生在多数领域达到甚至超过了最佳教师的水平。
        2.  **模块化与可扩展**：教师模型可以灵活选择（RL模型、SFT模型或学生自身），便于集成新能力而无需重构整个流程。
        3.  **支持迭代协同进化**：蒸馏后的学生可以重新进入专业化RL阶段，产出更强的教师，形成自我强化的能力提升循环。

### 3. **轻量级多令牌预测的双重作用**
- **改进/不同之处**：
    - 将MTP不仅作为**训练目标**来提升模型质量，更**原生地作为推测解码的草稿模型**用于加速推理。
    - **轻量化设计**：MTP模块使用**密集FFN（而非MoE）** 和**滑动窗口注意力（而非全局注意力）**，参数量仅0.33B，旨在避免自身成为推理瓶颈。
- **解决的问题/带来的优势**：
    - **核心问题**：
        1.  **LLM解码内存瓶颈**：传统自回归解码算术强度低。
        2.  **RL训练瓶颈**：策略展开阶段因推理成本高而成为主要瓶颈。
    - **具体优势**：
        1.  **推理加速**：通过并行生成和验证多个草稿token，实现了**token级并行**，提升了FFN和注意力计算的算术强度。实验显示，使用3层MTP可实现最高**2.6倍的解码加速**（平均接受长度3.6）。
        2.  **RL训练加速**：MTP使得**小批量、策略RL训练**变得可行（通过提升token级并行度而非依赖大批量），并缓解了处理长序列任务时的GPU闲置问题。

### 4. **针对MoE模型的大规模RL训练基础设施优化**
- **改进/不同之处**：
    - 提出了 **“Rollout Routing Replay”** 技术，确保训练时使用与策略展开阶段**完全相同的专家路由**，解决了因数值精度问题导致的专家路由不一致性。
    - 设计了**数据调度器**和**工具箱/工具管理器**，以细粒度调度序列、管理异构工具调用环境，并解决资源争用和局部效率低下问题。
- **解决的问题/带来的优势**：
    - **核心问题**：确保**MoE模型在RL训练中的稳定性和效率**，并支持**大规模、多环境（代码、终端、搜索）的智能体训练**。
    - **具体优势**：
        1.  **训练稳定性**：R3消除了路由不一致性，是MoE模型成功进行RL训练的关键。
        2.  **高资源利用率**：数据调度器动态分配任务，结合部分展开等技术，最大化GPU利用率，支撑了在超过12万个代码环境上的大规模策略训练。
        3.  **可扩展的智能体训练**：模块化的工具管理框架使得能够并行训练多种类型的智能体（代码、搜索、通用工具使用），并将代码智能体训练中获得的能力有效泛化到其他领域。

### 5. **整体模型设计的效率与性能平衡**
- **改进/不同之处**：
    - 模型总参数量309B，但**激活参数量仅15B**。在性能上对标DeepSeek-V3.2（671B总参/37B激活）和Kimi-K2（1043B总参/32B激活）等顶级模型，但**总参数量仅为它们的1/2至1/3**。
    - 全面采用**FP8混合精度训练和推理**，在保证数值稳定性的同时大幅提升效率。
- **解决的问题/带来的优势**：
    - **核心问题**：在追求强大推理和智能体能力的同时，**控制模型的计算和存储成本**。
    - **具体优势**：
        1.  **高性价比**：以更少的参数和计算成本，达到了与更大模型相媲美的性能（尤其在SWE-Bench等代码智能体任务上表现领先），体现了**卓越的“性能-参数”效率比**。
        2.  **实用性**：高效的架构和推理优化（MTP）使其更易于实际部署和应用。

---

**总结**：MiMo-V2-Flash的核心创新在于**系统性地追求效率与性能的极致统一**。它通过**混合滑动窗口注意力**解决长上下文计算瓶颈，通过**MOPD**高效整合多领域专家能力，通过**轻量级MTP**同时加速训练和推理，并通过**定制化RL基础设施**确保大规模训练的稳定与高效。这些创新共同使其成为一个在保持竞争力的同时，在参数效率、推理速度和训练可扩展性方面具有显著优势的模型。


### 问题 4

**Q**: 论文在实验或评估中最终实现了怎样的效果？请说明使用了哪些数据集和评价指标，与哪些基线方法进行了对比，以及在关键指标上的主要性能提升或结论。如果论文未给出明确的定量结果，也请说明原因。

**A**: ## 论文实验与评估效果总结

该技术报告通过一系列广泛的实验，全面评估了 **MiMo-V2-Flash** 模型在基础能力、长上下文理解、复杂推理和智能体任务上的性能。评估结果表明，该模型在参数效率、推理速度和综合能力方面达到了顶尖水平。

### 一、 主要评估数据集与指标

论文使用了覆盖多个维度的基准测试集，具体如下：

| 能力维度 | 主要数据集/基准测试 | 关键评价指标 |
| :--- | :--- | :--- |
| **通用语言理解与推理** | MMLU、MMLU-Pro、MMLU-Redux、BBH、HellaSwag、ARC-Challenge、DROP、TriviaQA、GPQA-Diamond、SuperGPQA、SimpleQA | 准确率 (Accuracy) |
| **数学推理** | GSM8K、MATH、AIME (2024 & 2025) | 准确率 (Accuracy) |
| **代码能力** | HumanEval+、MBPP+、CRUXEval、MultiPL-E、BigCodeBench、LiveCodeBench、**SWE-Bench** (Agentless Repair) | 通过率 (Pass Rate) |
| **中文理解** | C-Eval、CMMLU、C-SimpleQA | 准确率 (Accuracy) |
| **多语言理解** | GlobalMMLU、INCLUDE | 准确率 (Accuracy) |
| **长上下文能力** | **NIAH-Multi** (检索)、**GSM-Infinite** (推理)、NoLiMa、RULER-32k、MRCR、LongBench V2 | 检索成功率、准确率 |
| **智能体与工具使用** | **SWE-Bench Verified/Multilingual** (代码修复)、Terminal-Bench、BrowseComp (搜索)、**τ²-Bench** (通用工具)、Arena-Hard (创意写作) | 问题解决率、得分 |

### 二、 对比的基线方法

论文将 MiMo-V2-Flash 与当前主流的开源和闭源大模型进行了全面对比：

1.  **开源模型 (主要对比对象)**:
    *   **Kimi-K2** (1043B总参，32B激活参): 代表超大规模MoE模型。
    *   **DeepSeek-V3.2** (671B总参，37B激活参): 代表另一顶尖开源MoE模型。
    *   **Gemma 2** 等 (用于架构消融实验)。

2.  **闭源模型 (作为性能上限参考)**:
    *   Google **Gemini 3.0 Pro**
    *   Anthropic **Claude 3.5 Sonnet**
    *   OpenAI **GPT-5 High**

### 三、 关键性能结果与结论

#### 1. **参数效率与综合性能**
*   **结论**: MiMo-V2-Flash (309B总参/15B激活参) 在多数基准测试上达到了与参数量大得多的 **Kimi-K2 (1043B)** 和 **DeepSeek-V3.2 (671B)** 相当甚至更优的性能。
*   **关键数据**:
    *   在 **MMLU-Pro** (复杂推理) 上得分为 **73.2**，显著优于 DeepSeek-V3.2-Exp-Base (62.1) 和 Kimi-K2-Base (69.2)。
    *   在 **AIME 2024&2025** (高难度数学) 上得分为 **35.3**，优于对比的所有开源基线模型。
    *   在 **SWE-Bench** (代码修复) 上，其Base模型即达到 **30.8%**，优于参数量3倍于它的Kimi-K2-Base (28.2%)，证明了其数据与架构在代码任务上的高效性。

#### 2. **长上下文能力**
*   **结论**: 采用 **混合滑动窗口注意力 (Hybrid SWA)** 的架构在长上下文任务上表现强劲且稳定，验证了其设计的有效性。
*   **关键数据**:
    *   在 **NIAH-Multi** (检索任务) 上，从32K到256K长度均保持接近 **100%** 的检索成功率。
    *   在 **GSM-Infinite** (极端长上下文推理) 上，从16K扩展到128K时性能下降最小，表现优于使用稀疏注意力的DeepSeek-V3.2，后者在更长上下文下性能衰减严重。

#### 3. **后训练范式 (MOPD) 效果**
*   **结论**: 提出的 **多教师策略蒸馏 (MOPD)** 框架能有效整合不同领域专家教师的能力，使学生模型达到或超越最佳教师的表现，避免了能力失衡问题。
*   **关键数据** (见表7):
    *   在 **AIME 2025** 上，学生模型经过MOPD后得分为 **94.1**，超过了最佳RL教师 (93.9)。
    *   在 **LiveCodeBench** 上，学生模型得分为 **83.2**，接近最佳RL教师 (82.6)。
    *   在 **SWE-Bench Verified** 上，学生模型达到 **73.4%**，几乎追平最佳RL教师 (74.2%)，同时在其他领域（如数学、通用写作）也保持了高水平。

#### 4. **推理速度优化 (MTP)**
*   **结论**: 轻量级 **多令牌预测 (MTP)** 模块能有效加速解码，尤其适用于低不确定性任务。
*   **关键数据**:
    *   MTP的**平均接受长度**与下一个令牌的交叉熵高度负相关 (`R²=0.995`)。在WebDev等任务中接受长度可达 **3.6**。
    *   使用3层MTP进行推测解码，在16K输入/1K输出的设定下，可实现最高 **~2.7倍** 的解码加速（见表10）。

#### 5. **最终模型综合排名**
*   **结论**: 经过完整后训练的 MiMo-V2-Flash 在多项关键任务上位列开源模型前列，甚至逼近顶尖闭源模型。
*   **关键数据** (见表9):
    *   **代码智能体**: 在 **SWE-Bench Verified** 上达到 **73.4%**，为开源模型最佳，接近GPT-5 High (74.9%)。
    *   **数学推理**: 在 **AIME 2025** 上达到 **94.1**，与顶级模型（如Gemini 3.0 Pro: 95.0）差距很小。
    *   **长上下文**: 在 **LongBench V2** 上得分 **60.6**，优于采用全注意力的大模型Kimi-K2-Thinking (48.1)。
    *   **通用智能体**: 在 **τ²-Bench** 上得分 **80.3**，与DeepSeek-V3.2-Thinking持平。

### 总结
MiMo-V2-Flash 通过**创新的混合滑动窗口注意力架构**实现了高效的长上下文建模，通过**轻量级MTP**显著提升了训练与推理效率，并通过**新颖的MOPD后训练范式**成功整合了多领域专家能力。实验评估全面且具有说服力，表明该模型以**显著更少的激活参数**，在**推理、代码、长上下文和智能体能力**上达到了与当今最大规模开源模型媲美甚至领先的水平，同时在**解码速度**上具有实际优势。论文给出了明确的定量结果，所有关键结论均有详实的数据支撑。


## 相关链接

- [arXiv 页面](https://arxiv.org/abs/2601.02780v2)
- [HTML 版本](https://arxiv.org/html/2601.02780v2)
