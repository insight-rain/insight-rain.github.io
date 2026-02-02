# The RoboSense Challenge: Sense Anything, Navigate Anywhere, Adapt Across Platforms

**相关性评分**: 6.0/10

**排名**: #87


---


## 基本信息

- **arXiv ID**: [2601.05014v1](https://arxiv.org/abs/2601.05014v1)
- **发布时间**: 2026-01-08T15:16:18Z
- **相关性评分**: 6.0/10
- **是否相关**: 是

## 作者

Lingdong Kong, Shaoyuan Xie, Zeying Gong, Ye Li, Meng Chu, Ao Liang, Yuhao Dong, Tianshuai Hu, Ronghe Qiu, Rong Li, Hanjiang Hu, Dongyue Lu, Wei Yin, Wenhao Ding, Linfeng Li, Hang Song, Wenwei Zhang, Yuexin Ma, Junwei Liang, Zhedong Zheng, Lai Xing Ng, Benoit R. Cottereau, Wei Tsang Ooi, Ziwei Liu, Zhanpeng Zhang, Weichao Qiu, Wei Zhang, Ji Ao, Jiangpeng Zheng, Siyu Wang, Guang Yang, Zihao Zhang, Yu Zhong, Enzhu Gao, Xinhan Zheng, Xueting Wang, Shouming Li, Yunkai Gao, Siming Lan, Mingfei Han, Xing Hu, Dusan Malic, Christian Fruhwirth-Reisinger, Alexander Prutsch, Wei Lin, Samuel Schulter, Horst Possegger, Linfeng Li, Jian Zhao, Zepeng Yang, Yuhang Song, Bojun Lin, Tianle Zhang, Yuchen Yuan, Chi Zhang, Xuelong Li, Youngseok Kim, Sihwan Hwang, Hyeonjun Jeong, Aodi Wu, Xubo Luo, Erjia Xiao, Lingfeng Zhang, Yingbo Tang, Hao Cheng, Renjing Xu, Wenbo Ding, Lei Zhou, Long Chen, Hangjun Ye, Xiaoshuai Hao, Shuangzhi Li, Junlong Shen, Xingyu Li, Hao Ruan, Jinliang Lin, Zhiming Luo, Yu Zang, Cheng Wang, Hanshi Wang, Xijie Gong, Yixiang Yang, Qianli Ma, Zhipeng Zhang, Wenxiang Shi, Jingmeng Zhou, Weijun Zeng, Kexin Xu, Yuchen Zhang, Haoxiang Fu, Ruibin Hu, Yanbiao Ma, Xiyan Feng, Wenbo Zhang, Lu Zhang, Yunzhi Zhuge, Huchuan Lu, You He, Seungjun Yu, Junsung Park, Youngsun Lim, Hyunjung Shim, Faduo Liang, Zihang Wang, Yiming Peng, Guanyu Zong, Xu Li, Binghao Wang, Hao Wei, Yongxin Ma, Yunke Shi, Shuaipeng Liu, Dong Kong, Yongchun Lin, Huitong Yang, Liang Lei, Haoang Li, Xinliang Zhang, Zhiyong Wang, Xiaofeng Wang, Yuxia Fu, Yadan Luo, Djamahl Etchegaray, Yang Li, Congfei Li, Yuxiang Sun, Wenkai Zhu, Wang Xu, Linru Li, Longjie Liao, Jun Yan, Benwu Wang, Xueliang Ren, Xiaoyu Yue, Jixian Zheng, Jinfeng Wu, Shurui Qin, Wei Cong, Yao He

## 关键词

Vision-Language-Action Model, VLA for Robotics, Adapt Across Platforms

## 一句话总结

RoboSense 2025挑战赛旨在通过多任务基准测试提升机器人感知的鲁棒性和跨平台适应性，涉及语言引导决策和跨模态感知，但未明确强调推理效率或轻量化架构。

## 摘要

Autonomous systems are increasingly deployed in open and dynamic environments -- from city streets to aerial and indoor spaces -- where perception models must remain reliable under sensor noise, environmental variation, and platform shifts. However, even state-of-the-art methods often degrade under unseen conditions, highlighting the need for robust and generalizable robot sensing. The RoboSense 2025 Challenge is designed to advance robustness and adaptability in robot perception across diverse sensing scenarios. It unifies five complementary research tracks spanning language-grounded decision making, socially compliant navigation, sensor configuration generalization, cross-view and cross-modal correspondence, and cross-platform 3D perception. Together, these tasks form a comprehensive benchmark for evaluating real-world sensing reliability under domain shifts, sensor failures, and platform discrepancies. RoboSense 2025 provides standardized datasets, baseline models, and unified evaluation protocols, enabling large-scale and reproducible comparison of robust perception methods. The challenge attracted 143 teams from 85 institutions across 16 countries, reflecting broad community engagement. By consolidating insights from 23 winning solutions, this report highlights emerging methodological trends, shared design principles, and open challenges across all tracks, marking a step toward building robots that can sense reliably, act robustly, and adapt across platforms in real-world environments.

## 详细分析

## 论文摘要：RoboSense 2025挑战赛

### 1. 研究背景和动机
现代自主系统（如自动驾驶汽车、无人机、移动机器人）越来越多地部署在开放、动态的真实环境中。然而，即使是最先进的感知模型，在面对传感器噪声、环境变化、平台差异等**分布偏移**时，性能也常常会严重下降。这揭示了当前机器人感知在**鲁棒性**和**泛化性**方面的巨大差距。为了系统性地评估和推动这一领域的发展，RoboSense 2025挑战赛应运而生。它旨在建立一个统一的基准，测试感知系统在多种现实世界扰动下的可靠性和适应性。

### 2. 核心方法和技术创新
挑战赛设立了五个互补的赛道，全面覆盖了机器人感知的关键前沿：
- **赛道1：语言驱动驾驶**：评估视觉-语言模型在视觉退化（如模糊、遮挡）下，基于多视角图像进行感知、预测和规划问答的鲁棒性。
- **赛道2：社会合规导航**：在逼真的室内动态人机环境中，评估RGB-D导航系统生成既高效又符合社会规范（如保持人际距离）的轨迹的能力。
- **赛道3：传感器布局泛化**：测试LiDAR 3D目标检测模型能否在不重新训练的情况下，泛化到不同的、未见过的传感器安装位置。
- **赛道4：跨模态无人机导航**：要求模型根据自然语言描述，在极端视角变化下，实现地面场景与航拍/卫星图像之间的检索与对齐。
- **赛道5：跨平台3D目标检测**：评估3D检测器在异构平台（车辆、无人机、四足机器人）之间进行知识迁移和适应的能力。

### 3. 主要实验结果
挑战赛吸引了来自16个国家、85个机构的143支队伍参与，反映了社区的广泛关注。通过对23个获胜方案的分析，发现了以下关键趋势：
- **数据与训练策略至关重要**：许多领先方案通过**数据增强**（如模拟遮挡、视角扰动）、**多源预训练**和**改进的自训练策略**获得了比单纯架构修改更大的性能提升。
- **几何一致性是关键瓶颈**：跨平台、跨视图的鲁棒性高度依赖于学习**视角不变的几何表示**，例如通过地面平面归一化或局部几何编码（如GBlobs）。
- **结构化推理提升可靠性**：在语言驱动任务中，采用**链式思维提示**、**任务特定路由**和**元数据 grounding** 能有效减少幻觉，提升推理的稳定性和可解释性。
- **成果显著**：各赛道的顶级方案均大幅超越了官方基线模型，例如在跨传感器布局检测中mAP相对提升超过18%，在语言驱动驾驶的腐蚀条件下稳定性显著提高。

### 4. 研究意义和价值
RoboSense 2025挑战赛通过提供标准化的数据集、基线模型和评估协议，为**鲁棒机器人感知**研究建立了一个大规模、可复现的基准。它强调了从追求“干净数据下的高精度”到“复杂真实世界下的高可靠性”的范式转变。挑战赛的结果和洞察为构建能够**感知万物、导航四方、跨平台自适应**的下一代自主系统指明了方向，对推动计算机视觉、机器人学和多模态学习在安全关键领域的实际部署具有重要价值。

## 问答对

### 问题 1

**Q**: 这篇论文的核心创新点是什么，他想解决什么问题，怎么解决的？

**A**: ## 论文核心分析

### **核心问题**
论文旨在解决一个关键瓶颈：**当前最先进的机器人感知模型在现实开放动态环境（如城市街道、室内、空中）中部署时，其可靠性会因传感器噪声、环境变化、平台差异等分布偏移而严重下降**。这揭示了“基准测试性能”与“实际部署可靠性”之间的巨大鸿沟。

### **核心创新点**
论文并非提出单一算法，而是构建了一个**系统性、大规模、多任务的基准挑战赛（Benchmark Challenge）**——RoboSense 2025。其创新性体现在：

1.  **统一的鲁棒性评估框架**：将机器人感知的“鲁棒性”和“适应性”从单一任务扩展到五个互补且关键的维度，形成一个全面的评估体系。
2.  **强调“分布偏移下的性能”而非“静态精度”**：与传统基准关注固定场景下的准确率不同，RoboSense专注于评估模型在**未见过的条件**下的表现，如传感器损坏、视角剧变、平台切换等。
3.  **从“车辆感知”到“跨平台通用感知”的范式扩展**：在上一届RoboDrive（专注于自动驾驶）的基础上，将范围扩展到无人机、足式机器人等多种平台，并融入语言交互、社会导航等更高层次的智能任务。

### **解决方案：五大挑战赛道**
论文通过组织涵盖五个赛道的挑战赛，为上述问题提供了系统性的解决方案和评估平台：

| 赛道 | 核心任务 | 评估的鲁棒性维度 | 关键技术挑战 |
| :--- | :--- | :--- | :--- |
| **1. 语言驾驶** | 基于多视角视觉输入，回答感知、预测、规划的文本问题。 | **视觉退化下的可靠推理**（如模糊、遮挡、恶劣天气）。 | 减少幻觉，保持视觉 grounding；多任务一致性。 |
| **2. 社会导航** | 在动态人机共存室内环境中，生成高效且符合社会规范的轨迹。 | **动态交互与部分可观测性**下的安全与舒适性。 | 理解人类意图，遵守社交距离，处理遮挡。 |
| **3. 传感器布局** | 训练于一种LiDAR布局的3D检测模型，需泛化到**未见过的**传感器布局。 | **传感器配置变化**引起的几何与数据分布偏移。 | 学习布局无关的几何表示，避免过拟合绝对坐标。 |
| **4. 跨模态无人机导航** | 根据自然语言描述，从航拍/卫星图像库中检索匹配的地面场景。 | **极端视角与外观变化**下的跨模态对齐。 | 解决地面与空中视图间的语义鸿沟，进行细粒度空间推理。 |
| **5. 跨平台3D检测** | 将3D物体检测模型从**车辆**平台迁移到**无人机**和**足式机器人**平台。 | **平台差异**导致的视角、运动模式、点云稀疏性等分布偏移。 | 实现无监督域适应，处理几何与动态特性的巨大差异。 |

### **方法论与关键发现**
通过对23支获胜队伍方案的分析，论文总结了实现鲁棒性的**共性技术趋势**：

- **数据与训练策略优先**：鲁棒性提升更多源于**数据增强**（如腐蚀模拟、布局混合、时间序列聚合）、**预训练**（大规模多数据集）、**训练策略**（课程学习、自训练）和**提示工程**，而非单纯的架构改动。
- **几何一致性是关键**：在跨视角、跨平台任务中，**显式建模几何不变性**（如地面平面归一化、局部几何编码GBlobs）的方法表现更优。
- **结构化推理增强可靠性**：在语言任务中，**思维链（CoT）、任务特定提示、元数据注入**能有效提升推理的可靠性和可解释性。
- **模块化与高效适应**：**参数高效微调（如LoRA）、专家混合（MoE）模型**在保持大模型能力的同时，实现了对特定领域或任务的快速适配。

### **实际价值**
1.  **为工业界与学术界提供“试金石”**：提供了标准化数据集、基线模型和评估协议，使大规模、可复现的鲁棒性研究成为可能。
2.  **指明研究方向**：明确了当前感知系统的脆弱环节，并指出未来方向应关注**复合分布偏移、不确定性校准、传感器无关表示**等。
3.  **推动实用化进程**：通过强调“在野”性能，直接推动了构建能在真实、多变世界中可靠感知和行动的机器人系统的研究。

**总结**：RoboSense 2025的核心创新在于**构建了一个以“鲁棒性”和“跨平台适应性”为核心评价标准的下一代机器人感知基准**。它通过五个精心设计的赛道，系统性地暴露并评估了现有模型在真实世界部署中的弱点，并通过社区竞赛汇集了前沿解决方案，为迈向“感知万物、无处不达、跨平台适应”的可靠机器人系统奠定了重要基石。


### 问题 2

**Q**: 请用一段话总结这篇论文，明确说明：论文试图解决的核心问题；提出的主要方法或框架；最终取得的主要效果或结论。要求语言简洁、信息密度高，不要复述摘要原文。

**A**: 这篇论文旨在解决机器人感知系统在真实开放环境中因传感器噪声、环境变化和平台差异导致的性能退化问题。为此，论文提出了**RoboSense 2025挑战赛**，这是一个统一的基准测试框架，通过五个互补的赛道（语言驱动的驾驶决策、社会合规导航、传感器配置泛化、跨视图/跨模态匹配、跨平台3D感知）来系统性地评估感知模型在分布偏移下的鲁棒性和适应性。该挑战赛提供了标准化数据集、基线模型和评估协议，吸引了全球广泛参与。通过对获胜方案的分析，论文总结了提升鲁棒性的关键方法趋势（如数据中心的增强、几何感知表示、参数高效微调），并得出结论：**实现可靠感知的关键往往在于训练策略和领域对齐，而非单纯的模型架构改进**，这为构建能跨平台可靠感知的机器人系统指明了方向。


### 问题 3

**Q**: 这篇论文相对于已有工作有哪些明确的创新点？请逐条列出，并对每一条说明：相比以往方法改进或不同之处在哪里，以及该创新解决了什么具体问题或带来了什么优势。

**A**: ## 论文创新点分析

这篇论文（RoboSense 2025挑战赛报告）的核心创新点在于其**系统性、多维度和平台化的评估框架**，旨在解决机器人感知系统在真实、开放世界中面临的**鲁棒性、适应性和泛化性**不足的根本问题。它并非提出单一的新算法，而是构建了一个全面的基准测试和竞赛平台，其创新点体现在以下几个方面：

### 1. **从单一领域到多领域、多任务的统一鲁棒性评估框架**
   - **改进/不同之处**：以往的基准测试（如KITTI、nuScenes）或挑战赛（如RoboDrive 2024）通常专注于单一领域（如自动驾驶）或单一任务（如3D检测），在固定、理想的传感器配置和环境下评估性能。RoboSense 2025将评估范围**显著扩展**到五个互补的赛道，涵盖了**语言驱动的决策、社会性导航、传感器配置、跨模态匹配和跨平台3D感知**。
   - **解决的问题/优势**：这解决了“模型在特定基准上表现优异，但在面对真实世界的复合性分布偏移时性能急剧下降”的痛点。通过统一框架评估模型在多种“意外情况”（如传感器噪声、视点变化、平台差异、社会交互）下的表现，更能反映实际部署的需求，推动了**面向真实世界的鲁棒性研究范式转变**。

### 2. **明确提出并系统评估“跨平台感知”的泛化能力**
   - **改进/不同之处**：传统3D感知研究大多假设传感器平台固定（如车载激光雷达）。RoboSense在**Track 3（传感器布局）和Track 5（跨平台3D检测）** 中，首次在大型基准中系统性地评估模型对**传感器布局变化**和**异构机器人平台**（车辆、无人机、四足机器人）的泛化能力。
   - **解决的问题/优势**：解决了“一个模型难以适配不同车辆型号或不同机器人形态”的实际工程难题。这鼓励研究**传感器无关**和**平台无关**的几何表征学习，促进了可移植、可复用的感知模块发展，降低了为每个新平台重新训练模型的成本。

### 3. **将视觉-语言模型（VLM）的评估置于“鲁棒性”和“安全关键”的驾驶场景下**
   - **改进/不同之处**：虽然已有工作（如DriveLM、DriveVLM）探索VLM在驾驶中的应用，但评估多集中在清洁数据上的推理能力。**Track 1** 创新性地在**视觉退化**（模糊、遮挡、恶劣天气）条件下，评估VLM的感知、预测和规划问答能力。
   - **解决的问题/优势**：直接针对VLM在感知证据模糊或损坏时容易产生**“幻觉”和“ grounding 错误”** 的安全隐患。这推动了**可靠、可解释的驾驶决策系统**的发展，要求模型在输出答案时不仅准确，还要在视觉输入不可靠时保持稳健和诚实。

### 4. **强调“社会合规性”作为导航系统的核心评估维度**
   - **改进/不同之处**：传统导航评估主要关注成功率（SR）和路径长度（SPL）。**Track 2** 在逼真的室内动态人机环境中，引入了**个人空间合规性**和**人机碰撞率**等量化指标，将社会行为规范纳入核心评估体系。
   - **解决的问题/优势**：解决了机器人导航中“高效但令人不适或危险”的行为问题。这推动了导航策略从**几何避障**向**社会意识交互**的演进，对于服务机器人在人群中的安全、可接受部署至关重要。

### 5. **构建大规模、标准化的数据集与基线系统，促进可复现研究与公平比较**
   - **改进/不同之处**：论文为每个赛道提供了精心构建的数据集（如扩展的GeoText-190、Social-HM3D、Place3D、Pi3DET）、强基线模型（如Qwen2.5-VL, Falcon, BEVFusion-L）和统一的在线评估服务器（CodaBench, EvalAI）。
   - **解决的问题/优势**：解决了该领域研究因数据集不统一、评估协议各异导致的**难以复现和公平比较**的问题。这极大地降低了研究门槛，使社区能够在大规模、可比较的平台上集中攻关鲁棒性难题，加速了技术进步。

### 6. **通过大规模社区竞赛，提炼出共性技术趋势与设计原则**
   - **改进/不同之处**：论文不仅报告结果，还深度分析了23支获胜队伍的解决方案，**归纳出跨赛道的共性方法论**。例如：鲁棒性更多源于**训练策略**（数据增强、预训练）而非纯架构改进；**几何一致性**是多模态对齐的瓶颈；**伪标签质量**对无监督域适应至关重要。
   - **解决的问题/优势**：为未来研究提供了清晰的**技术路线图**和**设计启示**。研究者可以避免重复探索，直接在这些被验证有效的方向上（如结构化的提示工程、正样本回放、局部几何编码、分层对比学习）进行深化和创新。

### 总结
RoboSense 2025的核心创新在于其**前瞻性的问题定义**和**系统性的工程实现**。它将研究焦点从“在理想条件下追求更高精度”转向“在复杂变化中保证基本可靠性”，通过一个**统一、多维、可操作的基准测试生态系统**，切实推动了鲁棒机器人感知从学术研究走向实际应用。其带来的最大优势是**为社区指明了下一代感知系统必须攻克的关键挑战，并提供了评估这些挑战的标准工具**。


### 问题 4

**Q**: 论文在实验或评估中最终实现了怎样的效果？请说明使用了哪些数据集和评价指标，与哪些基线方法进行了对比，以及在关键指标上的主要性能提升或结论。如果论文未给出明确的定量结果，也请说明原因。

**A**: ## 论文实验与评估效果分析

该论文（RoboSense 2025挑战赛报告）的核心并非提出单一新方法，而是**组织并总结了一个大规模、多赛道的机器人感知鲁棒性基准测试**。其实验与评估效果体现在**整个挑战赛的参与规模、各赛道基线模型的性能提升以及从获胜方案中提炼出的方法论趋势**。

### 1. 整体规模与参与度
- **数据集**：挑战赛为五个赛道分别提供了标准化的数据集，均基于现有知名基准扩展，以强调鲁棒性和跨域泛化：
    - **Track 1**: 基于DriveBench和RoboBEV，包含视觉退化（模糊、遮挡、恶劣天气等）的多视角驾驶场景语言问答数据。
    - **Track 2**: 基于Social-HM3D数据集，包含844个逼真室内场景和动态行人。
    - **Track 3**: 基于Place3D基准，包含不同LiDAR传感器放置配置（如线形、梯形、金字塔形）的合成数据。
    - **Track 4**: 基于扩展的GeoText-190数据集（源自GeoText-1652），包含33,516个文本查询和11,172张航空图像，涵盖190个未见类别。
    - **Track 5**: 基于扩展的Pi3DET数据集，包含来自车辆、无人机和四足机器人平台的LiDAR点云数据。
- **评价指标**：各赛道采用针对性的指标评估鲁棒性和性能：
    - **Track 1**: 综合**准确率（MCQ）**和**LLMScore（VQA）**，加权平均评估感知、预测、规划任务在干净和损坏数据上的表现。
    - **Track 2**: 综合**成功率（SR）**、**路径长度加权成功率（SPL）**、**个人空间合规率（PSC）**和**人机碰撞率（H-Coll）**。
    - **Track 3**: 主要使用**平均精度（mAP）**，辅以**NDS**及一系列误差指标（mATE, mASE等）。
    - **Track 4**: 使用**Recall@1**和**Recall@10**的平均值作为最终得分。
    - **Track 5**: 使用**3D平均精度（AP）**评估跨平台（车辆→无人机、车辆→四足机器人）的检测性能。
- **参与情况**：吸引了**143支团队**（来自85个机构，16个国家），反映了社区对鲁棒性感知研究的广泛关注。

### 2. 各赛道基线对比与性能提升
论文为每个赛道设立了官方基线模型，并与获胜团队方案进行了对比。**所有赛道的顶级方案均显著超越了基线性能**，证明了所提出方法（数据增强、模型调整、训练策略等）的有效性。

#### **Track 1: 语言驱动驾驶**
- **基线**: Qwen2.5-VL-7B
- **关键提升**: 冠军团队（TQL）的最终得分（61.00）远超基线（43.50）。**核心结论**：通过**伪标签预训练、多视图融合、长尾数据重平衡**等策略，大幅提升了VLM在视觉退化条件下的推理鲁棒性和答案一致性。

#### **Track 2: 社会导航**
- **基线**: Falcon
- **关键提升**: 冠军团队（Are Ivan）在Phase 2的**总得分（70.2）**显著高于基线（62.5）。**核心结论**：引入**正向经验回放（PER）**等强化学习优化技巧，有效提升了在动态人机环境中的导航成功率和社交合规性。

#### **Track 3: 传感器放置泛化**
- **基线**: BEVFusion-L
- **关键提升**: 冠军团队（LRP）的**mAP达到0.784**，比基线（0.605）提升约**29.6%**。**核心结论**：采用**局部几何编码（GBlobs）**和**双模型混合检测**等策略，使模型对未知的传感器放置配置具有更强的泛化能力。

#### **Track 4: 跨模态无人机导航**
- **基线**: X-VLM
- **关键提升**: 冠军团队（TeleAI）的**最终得分（49.82%）**比基线（37.27%）提升显著。**核心结论**：**参数高效的专家混合框架**和**领域对齐的数据预处理**能有效桥接极端视角变化带来的语义鸿沟。

#### **Track 5: 跨平台3D检测**
- **基线**: PV-RCNN + ST3D++
- **关键提升**: 冠军团队（Visionary）在Quadruped平台上的**mAP达到58.54%**，相比基线有大幅提升。**核心结论**：**大规模统一预训练**和**视点归一化（地面平面规范化）**是实现跨异构机器人平台稳健检测的关键。

### 3. 核心结论与趋势
论文通过对23支获胜团队方案的分析，提炼出以下超越单一赛道的重要结论：

1.  **鲁棒性源于训练策略与数据工程**：性能提升更多来自**数据中心的鲁棒性策略**（如损坏感知增强、更强的预训练、更好的采样调度、伪标签过滤），而非单纯的架构改动。
2.  **几何理解是跨模态鲁棒性的共同瓶颈**：无论是跨视图检索（Track 4）、跨平台检测（Track 5），还是传感器放置泛化（Track 3），**几何感知的表征和空间先验**对于实现稳定转移至关重要。
3.  **分布偏移是复合且多因素的**：现实中的挑战往往是多种偏移（如传感器损坏+平台切换）同时发生。当前方法虽在单一偏移上表现良好，但应对**复合偏移**仍是未来挑战。
4.  **可靠性需要校准与验证**：仅提高平均指标不足以保证安全。需要关注**不确定性校准、拒绝机制和可验证的推理轨迹**，以规避关键性故障。

### 总结
RoboSense 2025挑战赛通过系统性的实验评估表明，尽管现有最先进的感知模型在特定分布下表现优异，但其在**真实世界变化下的鲁棒性依然脆弱**。然而，通过采用针对性的增强、适应和推理技术，可以显著提升模型对**传感器损坏、视角变化、配置差异和平台迁移**的适应能力。该工作为社区提供了一个全面的基准和清晰的改进方向，推动了构建“**在任何地方感知，跨平台适应**”的可靠机器人系统的研究进程。


## 相关链接

- [arXiv 页面](https://arxiv.org/abs/2601.05014v1)
- [HTML 版本](https://arxiv.org/html/2601.05014v1)
