# Literature Review: Language-Conditioned Neural Operators for PDE Solving

This literature review surveys the foundational and recent work supporting the thesis that **language can serve as a general-purpose latent conditioning interface for neural PDE solvers**. We organize the discussion around five areas: (1) neural operators for learning PDE solution maps, (2) feature-wise conditioning mechanisms, (3) grounding language models in physical systems, (4) foundation models and multi-task learning for PDEs, and (5) parameter-efficient adaptation for scientific models.

---

## 1. Neural Operators for Parametric PDEs

The core computational backbone of this project is the Fourier Neural Operator (FNO), introduced by Li et al. [1] as a method for learning resolution-invariant mappings between infinite-dimensional function spaces. Unlike classical solvers that discretize and solve a single PDE instance, the FNO learns the entire solution operator: given any input function (e.g., an initial condition or coefficient field), it produces the corresponding solution in a single forward pass. The key architectural insight is to parameterize the integral kernel directly in Fourier space, restricting it to a finite number of low-frequency modes and leveraging the Fast Fourier Transform for quasi-linear complexity. Li et al. demonstrated that this approach achieves state-of-the-art accuracy on Burgers' equation, Darcy flow, and the Navier-Stokes equation in the turbulent regime, while being up to three orders of magnitude faster than traditional pseudo-spectral solvers. Critically for our work, FNO supports zero-shot super-resolution --- it can be trained at one spatial resolution and evaluated at another without retraining --- and handles non-periodic boundary conditions despite operating in Fourier space, thanks to the local linear transform (bias term) W that tracks boundary information [1].

The neural operator paradigm was independently established by Lu et al. [2] with DeepONet, which decomposes the operator into a branch network (encoding the input function) and a trunk network (encoding the query location), based on the universal approximation theorem for operators. While DeepONet operates in physical space and naturally handles irregular geometries, FNO's spectral parameterization offers superior computational efficiency on uniform grids. Subsequent work has extended the FNO framework substantially. The Tensorized FNO (TFNO) by Kossaifi et al. [3] applies Tucker factorization to the spectral weight tensors, achieving 150x parameter compression with negligible accuracy loss. The Geometry-Informed Neural Operator (GINO) by Li et al. [4] extends FNO to arbitrary non-grid geometries by coupling graph neural operators with latent spectral processing, demonstrating 26,000x speedup over CFD on 3D aerodynamics problems. U-NO by Rahman et al. [5] introduces a U-Net-style multi-scale architecture for neural operators, enabling deeper models with lower memory and achieving 26-44% improvement over standard FNO on benchmark problems.

Physics-Informed Neural Operators (PINO) by Li et al. [6] combine data-driven training with PDE physics constraints, showing that FNO architectures can be enhanced through physics-based regularization without architectural changes. More recently, Latent Neural Operators (LNO) by Wang et al. [7] introduce Physics-Cross-Attention to map between geometric space and a learned latent space where operator learning occurs, achieving state-of-the-art results on multiple benchmarks with 50% GPU memory reduction. The latent space concept in LNO parallels our use of LLM latent vectors as conditioning signals, though the source and nature of the latent representations differ fundamentally. A comprehensive survey by Azizzadenesheli et al. [8] reviews the theoretical foundations, architectures, and applications of neural operators, establishing the broader context within which language-conditioned neural operators would represent a novel contribution.

For benchmarking, PDEBench by Takamoto et al. [9] provides large-scale datasets across multiple PDE families with standardized evaluation protocols, while APEBench by Koehler et al. [10] --- the framework used in this project --- focuses specifically on autoregressive neural emulators with JAX-based differentiable simulation across 46 PDEs in 1D, 2D, and 3D. APEBench's emphasis on rollout stability and temporal generalization makes it particularly suitable for evaluating whether language conditioning improves the long-horizon prediction capabilities of neural operators.

## 2. Feature-wise Conditioning: FiLM and Beyond

The primary conditioning mechanism in this project is Feature-wise Linear Modulation (FiLM), introduced by Perez et al. [11] for visual reasoning. FiLM learns to adaptively influence a neural network's computation by applying a feature-wise affine transformation --- producing per-channel scale (gamma) and shift (beta) parameters conditioned on an arbitrary input. Formally, given conditioning input x, FiLM computes gamma = f(x) and beta = h(x), then modulates feature maps F as: FiLM(F | gamma, beta) = gamma * F + beta. Perez et al. showed that this simple mechanism is remarkably powerful: on the CLEVR visual reasoning benchmark, FiLM halved the state-of-the-art error, outperforming even methods with explicit compositional reasoning priors such as neural module networks [11].

Several properties of FiLM make it particularly well-suited for conditioning neural operators with language. First, FiLM is computationally efficient --- it requires only two parameters per modulated feature map and its cost does not scale with spatial resolution, a critical property for resolution-invariant neural operators. Second, Perez et al.'s ablation study revealed that FiLM is robust to architectural modifications: it functions effectively without normalization layers, at various positions within residual blocks, and with as few as a single FiLM layer in the network. Third, t-SNE analysis of learned FiLM parameters showed that they cluster by reasoning function in a hierarchical manner --- low-level groupings in early layers and high-level semantic groupings in later layers --- suggesting that FiLM learns structured, function-based modularity purely from end-to-end training [11]. This property is directly relevant to our hypothesis that FiLM layers applied after FNO spectral blocks can learn physics-aware modulation patterns from language-derived latent vectors.

FiLM's applicability extends well beyond visual reasoning. In AudioLDM, Liu et al. [12] use FiLM layers to inject text and audio embeddings (from a CLAP encoder) into a U-Net backbone for text-to-audio generation. This represents the closest architectural analogue to our project in a non-vision domain: text embeddings conditioned via FiLM into a generative model operating in a spectral-like domain. FiLM-Ensemble [13] extends the mechanism for uncertainty quantification by using FiLM layers to generate ensemble diversity, a capability relevant to quantifying prediction uncertainty in chaotic PDE systems such as Kuramoto-Sivashinsky. In physics-informed settings, hypernetwork-based meta-learning approaches [14] generalize FiLM by using one network to generate the full parameters of another (a PINN) conditioned on PDE input parameters. These works demonstrate that hypernetwork-style conditioning --- of which FiLM is a lightweight special case --- produces meaningful adaptation across PDE families, validating our choice of FiLM as the primary conditioning method.

Our project also investigates spectral gating as an alternative conditioning mechanism. While FiLM operates on spatial feature maps after each spectral block, spectral gating directly modulates Fourier coefficients via a learned, frequency-dependent gate: gate_k = 1 + alpha * tanh(MLP(z_cond)_k). This design is motivated by the observation that different PDEs have distinct spectral signatures --- diffusion-dominated equations decay rapidly in frequency space while chaotic systems like Kuramoto-Sivashinsky maintain energy across modes --- and language conditioning could learn to modulate these spectral characteristics adaptively.

## 3. Grounding Language Models in Physical Systems

A central motivation of this project is the insight from SayCan (Ahn et al. [15]) that large language models encode rich semantic knowledge about the world that can be extracted and grounded in physical systems through appropriate interfacing mechanisms. SayCan demonstrated that by combining LLM-derived task probabilities with learned affordance functions (value functions from reinforcement learning), a mobile manipulator robot could execute temporally extended, abstract natural language instructions in a real kitchen environment. The key factorization --- p(success) is proportional to p(skill is useful | instruction) * p(skill is feasible | state) --- separates semantic understanding from physical grounding, enabling each component to be improved independently. Notably, Ahn et al. showed that simply upgrading the underlying language model (from FLAN-137B to PaLM-540B) directly improved robotic task execution success from 61% to 74%, establishing that improvements in language modeling translate to improvements in physical task performance [15].

Our project adapts this philosophy to scientific computing: instead of grounding language in robotic affordances, we ground language in the spectral dynamics of PDE solvers. The LLM provides a latent representation z that encodes semantic knowledge about PDE physics (equation type, parameter regimes, qualitative behavior), while the FNO provides the computational substrate that actually solves the equation. The projection and conditioning layers serve as the interface between these two modalities, analogous to SayCan's value function scoring mechanism.

Recent work has explored the intersection of language models and PDE solving from multiple angles. Lang-PINN [16] uses multi-agent LLM systems to automatically construct and train physics-informed neural networks from natural language PDE descriptions, achieving 3-5 orders of magnitude MSE reduction versus baselines. However, Lang-PINN uses LLMs as code-generating agents rather than as latent feature extractors, making it architecturally distinct from our approach. Work on evaluating LLMs' physics simulation capabilities [17] has established baseline understanding of what physics knowledge LLMs encode, revealing both surprising capabilities and systematic limitations in reasoning about dynamical systems. Most relevantly, recent findings demonstrate that text-trained LLMs can zero-shot extrapolate spatiotemporal PDE dynamics via in-context learning [18], revealing a three-stage learning mechanism and providing direct evidence that LLM representations encode useful physics priors. This supports our core hypothesis that mean-pooled LLM hidden states carry meaningful information about PDE behavior even without explicit physics training.

The bilevel optimization framework of Ma et al. [19], which combines LLM abstract reasoning with simulation computational strength, represents yet another paradigm for LLM-simulation integration. CodePDE [20] introduces a framework for generating PDE solvers using LLMs, evaluating their reasoning, debugging, and self-refinement capabilities. These approaches treat LLMs as reasoning or code-generation engines; our approach instead treats the LLM as a feature extractor whose internal representations serve as conditioning signals, keeping the system fully differentiable and avoiding the brittleness of generated code or multi-agent pipelines.

## 4. Foundation Models and Multi-Task Learning for PDEs

A key question our project addresses is whether language conditioning enables a single neural operator to generalize across PDE families. This connects to the broader effort toward foundation models for scientific computing. Poseidon by Herde et al. [21] is a multiscale operator transformer pretrained on diverse fluid dynamics PDEs using time-conditioned layer norms --- a mechanism similar in spirit to FiLM. Poseidon outperforms baselines on 15 downstream tasks and generalizes to unseen physics, but its conditioning is limited to temporal information rather than rich semantic descriptions of PDE physics.

The DPOT framework by Hao et al. [22] pretrains operator transformers with up to 1 billion parameters on 10+ PDE datasets using denoising objectives with Fourier attention, achieving state-of-the-art results on standard PDE benchmarks. CoDA-NO [23] takes a different approach, tokenizing functions along the codomain (channel) space and using self-supervised pretraining on fluid flow snapshots, with 36% improvement on fluid-structure interaction problems via few-shot fine-tuning. MPP (Multiple Physics Pretraining) by McCabe et al. [24] introduces shared embedding and normalization layers for training a single transformer across heterogeneous physical systems, where a single pretrained model matches or outperforms task-specific baselines.

These foundation model approaches share a common limitation: they condition on task identity implicitly through learned embeddings or normalization statistics, without access to the rich semantic structure that natural language provides. Our language conditioning approach offers a complementary advantage --- by encoding PDE descriptions in natural language, we can express not just task identity but also qualitative physics (e.g., "chaotic spatiotemporal dynamics," "balances shock steepening against viscous smoothing") and quantitative parameters (e.g., "diffusion coefficient 0.008") in a unified representation space. If successful, this would provide a more interpretable and extensible conditioning interface than learned task embeddings.

In-Context Operator Networks (ICON) by Liu et al. [25] and Zebra by Serrano et al. [26] represent a parallel conditioning paradigm where the model adapts to new PDEs by conditioning on example trajectories provided at inference time (in-context learning). ICON handles 19 distinct problem types without fine-tuning, while Zebra additionally provides built-in uncertainty quantification. These approaches condition on data examples rather than language, requiring access to solved instances of the target PDE. Language conditioning could complement in-context approaches by providing high-level guidance about the PDE type and expected behavior, potentially reducing the number of example trajectories needed for effective adaptation.

## 5. Parameter-Efficient Adaptation and LoRA

The planned Phase 5 of this project involves enabling gradient flow from the PDE trajectory loss into the LLM via Low-Rank Adaptation (LoRA), introduced by Hu et al. [27]. LoRA adds trainable low-rank decomposition matrices to frozen pretrained weights, typically in the attention layers: W' = W + BA, where B and A are low-rank matrices with rank r much smaller than the model dimension. This enables fine-tuning with orders of magnitude fewer parameters while maintaining the full expressiveness of the pretrained model. Hu et al. showed that LoRA achieves performance comparable to full fine-tuning on GPT-3 175B with only 0.01% of the trainable parameters [27].

In our setting, LoRA adaptation is particularly compelling because it allows the PDE trajectory loss to reshape the LLM's internal representations without catastrophic forgetting of the language prior. The hypothesis is that LoRA-adapted hidden states will produce conditioning vectors z that are not merely static physics descriptions but dynamically optimized latent control signals for the neural operator. This connects to the broader theme of the project --- that language, when embedded through a differentiable pipeline and adapted end-to-end, can transcend its role as a static information carrier and become an active modulation signal for scientific computation.

Batched LoRA approaches [28] extend this to multi-task settings where different PDE tasks might benefit from task-specific LoRA adapters within a shared LLM backbone, enabling efficient scaling as the number of PDE scenarios grows.

---

## Positioning and Novelty

This project sits at a unique intersection of the surveyed literature. While neural operators [1-8] provide powerful PDE solvers and FiLM [11] provides an effective conditioning mechanism, no prior work has used LLM latent representations as the conditioning signal for neural operators. Foundation model approaches [21-26] achieve multi-PDE generalization through learned embeddings or in-context examples, but none leverage the rich semantic structure of natural language descriptions. Language-PDE interfaces [16-20] use LLMs as code generators or reasoning agents, but none extract and inject latent representations into the operator's computational pipeline in a fully differentiable manner.

The core contribution is therefore the complete, differentiable pipeline: text -> LLM -> hidden states -> mean pooling -> z -> projection -> FiLM/spectral conditioning -> FNO -> prediction, where language acts as a latent control signal rather than a reasoning output.

---

## References

[1] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar, "Fourier Neural Operator for Parametric Partial Differential Equations," *ICLR*, 2021.

[2] L. Lu, P. Jin, G. Pang, Z. Zhang, and G. E. Karniadakis, "Learning Nonlinear Operators via DeepONet Based on the Universal Approximation Theorem of Operators," *Nature Machine Intelligence*, vol. 3, pp. 218-229, 2021.

[3] J. Kossaifi, N. Kovachki, K. Azizzadenesheli, and A. Anandkumar, "Multi-Grid Tensorized Fourier Neural Operator for High-Resolution PDEs," *arXiv:2310.00120*, 2023.

[4] Z. Li et al., "Geometry-Informed Neural Operator for Large-Scale 3D PDEs," *NeurIPS*, 2023.

[5] M. A. Rahman, Z. E. Ross, and K. Azizzadenesheli, "U-NO: U-shaped Neural Operators," *Transactions on Machine Learning Research (TMLR)*, 2022.

[6] Z. Li, H. Zheng, N. Kovachki, D. Jin, H. Chen, B. Liu, K. Azizzadenesheli, and A. Anandkumar, "Physics-Informed Neural Operator for Learning Partial Differential Equations," *ACM/IMS Journal of Data Science*, 2023.

[7] T. Wang et al., "Latent Neural Operator for Solving Forward and Inverse PDE Problems," *NeurIPS*, 2024.

[8] K. Azizzadenesheli, N. Kovachki, Z. Li, M. Liu-Schiaffini, J. Kossaifi, and A. Anandkumar, "Neural Operators for Accelerating Scientific Simulations and Design," *Nature Reviews Physics*, 2024.

[9] M. Takamoto et al., "PDEBench: An Extensive Benchmark for Scientific Machine Learning," *NeurIPS Datasets and Benchmarks*, 2022.

[10] F. Koehler et al., "APEBench: A Benchmark for Autoregressive Neural Emulators of PDEs," *NeurIPS Datasets and Benchmarks*, 2024.

[11] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, "FiLM: Visual Reasoning with a General Conditioning Layer," *AAAI*, 2018.

[12] H. Liu et al., "AudioLDM: Text-to-Audio Generation with Latent Diffusion Models," *ICML*, 2023.

[13] "FiLM-Ensemble: Probabilistic Deep Learning via Feature-wise Linear Modulation," *OpenReview*, 2022.

[14] "Hypernetwork-based Meta-Learning for Low-Rank Physics-Informed Neural Networks," *NeurIPS*, 2023.

[15] M. Ahn, A. Brohan, N. Brown, Y. Chebotar, et al., "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances," *arXiv:2204.01691*, 2022.

[16] "Lang-PINN: From Language to Physics-Informed Neural Networks via a Multi-Agent Framework," *arXiv:2510.05158*, 2024.

[17] "Physics Simulation Capabilities of LLMs," *arXiv:2312.02091*, 2023.

[18] "Text-Trained LLMs Can Zero-Shot Extrapolate PDE Dynamics," *arXiv:2509.06322*, 2025.

[19] Y. Ma et al., "LLM and Simulation as Bilevel Optimizers: A New Paradigm to Advance Physical Scientific Discovery," *ICML*, 2024.

[20] "CodePDE: An Inference Framework for LLM-driven PDE Solver Generation," *arXiv:2505.08783*, 2025.

[21] M. Herde, B. Raonic, et al., "Poseidon: Efficient Foundation Models for PDEs," *NeurIPS*, 2024.

[22] Z. Hao et al., "DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training," *ICML*, 2024.

[23] "Pretraining Codomain Attention Neural Operators for Solving Multiphysics PDEs (CoDA-NO)," *NeurIPS*, 2024.

[24] M. McCabe, B. Regaldo-Saint Blancard, et al., "Multiple Physics Pretraining for Spatiotemporal Surrogate Models," *NeurIPS*, 2024.

[25] Y. Liu et al., "In-Context Operator Networks," *Journal of Computational Physics*, 2024.

[26] L. Serrano, A. Kassai Koupai, T. X. Wang, P. Erbacher, and P. Gallinari, "Zebra: In-Context and Generative Pretraining for Solving Parametric PDEs," *ICML*, 2025.

[27] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," *ICLR*, 2022.

[28] "Batched Low-Rank Adaptation of Foundation Models," *NeurIPS*, 2024.
