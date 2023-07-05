# SRAM-Based Compute-In-Memory

<div align=center><img src=./imgs/various-memory.png title="Fujiki et al. 2022 MICRO" width=50%/></div>

Compute-in-memory has attracted growing attention in the AI era. Among different types of memory, SRAM features high speed, precision, and compatibility with digital CMOS technology. Here is a collection of recent work on SRAM-based compute-in-memory. Most of these work is focused on AI acceleration.

## List of Papers

- 2023: [ISSCC](#2023-isscc), [ASPLOS](#2023-asplos), [HPCA](#2023-hpca), [JSSC](#2023-jssc), [TCAD](#2023-tcad)
- 2022: [ISSCC](#2022-isscc), [VLSI](#2022-vlsi), [MICRO](#2022-micro), [DAC](#2022-dac), [GLSVLSI](#2022-glsvlsi), [JSSC](#2022-jssc), [TC](#2022-tc)
- 2021: [ISSCC](#2021-isscc), [VLSI](#2021-vlsi), [ASSCC](#2021-asscc), [JSSC](#2021-jssc)
- 2020: [ISSCC](#2020-isscc), [DAC](#2020-dac), [ISCAS](#2020-iscas), [JSSC](#2020-jssc), [TC](#2020-tc), [SSC-L](#2020-ssc-l)
- 2019: [ISSCC](#2019-isscc), [ISCA](#2019-isca), [HPCA](#2019-hpca), [ESSCIRC](#2019-esscirc), [JSSC](#2019-jssc), [TVLSI](#2019-tvlsi)
- 2018: [ISSCC](#2018-isscc), [ISCA](#2018-isca), [JSSC](#2018-jssc)
- 2017: [ISSCC](#2017-isscc), [MICRO](#2017-micro), [HPCA](#2017-hpca), [JSSC](#2017-jssc)
- 2016: [JSSC](#2016-jssc)

### 2023 ISSCC

- (7.1) A 22nm 832Kb Hybrid-Domain Floating-Point SRAM In-Memory-Compute Macro with 16.2-70.2TFLOPS/W for High-Accuracy AI-Edge Devices [Floating-point, hybrid-domain for multiplication]
- (7.2) A 28nm 64-kb 31.6-TFLOPS/W Digital-Domain Floating-Point-Computing-Unit and Double-Bit 6T-SRAM Computing-in-Memory Macro for Floating-Point CNNs [Floating-point, double-bitcell, approximation]
- (7.3) A 28nm 38-to-102-TOPS/W 8b Multiply-Less Approximate Digital SRAM Compute-In-Memory Macro for Neural-Network Inference
- (7.4) A 4nm 6163-TOPS/W/b 4790-TOPS/mm2/b SRAM Based Digital-Computing-in-Memory Macro Supporting Bit-Width Flexibility and Simultaneous MAC and Weight Update
- (7.5) A 28nm Horizontal-Weight-Shift and Vertical-Feature-ShiftBased Separate-WL 6T-SRAM Computation-in-Memory Unit-Macro for Edge Depthwise Neural-Networks
- (7.6) A 70.85–86.27TOPS/W PVT-Insensitive 8b Word-Wise ACIM with Post-Processing Relaxation
- (7.7) CV-CIM: A 28nm XOR-Derived Similarity-Aware Computation-in-Memory for Cost-Volume Construction
- (7.8) A 22nm Delta-Sigma Computing-In-Memory (ΔΣCIM) SRAM Macro with Near-Zero-Mean Outputs and LSB-First ADCs Achieving 21.38TOPS/W for 8b-MAC Edge AI Processing
- (7.9) CTLE-Ising: A 1440-Spin Continuous-Time Latch-Based Ising Machine with One-Shot Fully-Parallel Spin Updates Featuring Equalization of Spin States
- (16.1) MulTCIM: A 28nm 2.24μJ/Token Attention-Token-Bit Hybrid Sparse Digital CIM-Based Accelerator for Multimodal Transformers
- (16.2) A 28nm 53.8TOPS/W 8b Sparse Transformer Accelerator with In-Memory Butterfly Zero Skipper for Unstructured-Pruned NN and CIM-Based Local-Attention-Reusable Engine
- (16.3) A 28nm 16.9-300TOPS/W Computing-in-Memory Processor Supporting Floating-Point NN Inference/Training with Intensive-CIM Sparse-Digital Architecture [Floating-point, computation division]
- (16.4) TensorCIM: A 28nm 3.7nJ/Gather and 8.3TFLOPS/W FP32 Digital-CIM Tensor Processor for MCM-CIM-Based Beyond-NN Acceleration [Floating-point, chiplet]
- (16.7) A 40-310TOPS/W SRAM-Based All-Digital Up to 4b In-Memory Computing Multi-Tiled NN Accelerator in FD-SOI 18nm for Deep-Learning Edge Applications
- (29.1) A 32.5mW Mixed-Signal Processing-in-Memory-Based k-SAT Solver in 65nm CMOS with 74.0% Solvability for 30-Variable 126-Clause 3-SAT Problems
- (29.2) Snap-SAT: A One-Shot Energy-Performance-Aware All-Digital Compute-in-Memory Solver for Large-Scale Hard Boolean Satisfiability Problems

### 2023 ASPLOS

- Infinity Stream: Portable and Programmer-Friendly In-/Near-Memory Fusion [Compiler]

### 2023 HPCA

- EVE: Ephemeral Vector Engines [General-purpose]

### 2023 JSSC

- (Extension version of F. Tu et al. *ISSCC*, pp. 254-255, 2022) ReDCIM: Reconfigurable Digital Computing- In -Memory Processor With Unified FP/INT Pipeline for Cloud AI Acceleration
- A Charge Domain SRAM Compute-in-Memory Macro With C-2C Ladder-Based 8-Bit MAC Unit in 22-nm FinFET Process for Edge Inference

### 2023 TCAD

- SDP: Co-Designing Algorithm, Dataflow, and Architecture for In-SRAM Sparse NN Acceleration [SW/HW co-design]

TranCIM: Full-Digital Bitline-Transpose CIM-based Sparse Transformer Accelerator With Pipeline/Parallel Reconfigurable Modes

- Extension version of [F. Tu et al. *ISSCC*, pp. 466-467, 2022]

### 2022 ISSCC

- (11.5) Single-Mode CMOS 6T-SRAM Macros with Keeper-Loading-Free Peripherals and Row-Separate Dynamic Body Bias Achieving 2.53fW/bit Leakage for AIoT Sensing Platforms
- (11.6) A 5-nm 254-TOPS/W 221-TOPS/mm2 Fully-Digital Computing-in-Memory Macro Supporting Wide-Range Dynamic-Voltage-Frequency Scaling and Simultaneous MAC and Write Operations
- (11.7) A 1.041-Mb/mm2 27.38-TOPS/W Signed-INT8 Dynamic-Logic-Based ADC-less SRAM Compute-In-Memory Macro in 28nm with Reconfigurable Bitwise Operation for AI and Embedded Applications
- (11.8) A 28nm 1Mb Time-Domain Computing-in-Memory 6T-SRAM Macro with a 6.6ns Latency, 1241GOPS and 37.01TOPS/W for 8b-MAC Operations for Edge-AI Devices
- (15.3) COMB-MCM: Computing-on-Memory-Boundary NN Processor with Bipolar Bitwise Sparsity Optimization for Scalable Multi-Chiplet-Module Edge Machine Learning
- (15.5) A 28nm 29.2TFLOPS/W BF16 and 36.5TOPS/W INT8 Reconfigurable Digital CIM Processor with Unified FP/INT Pipeline and Bitwise In-Memory Booth Multiplication for Cloud Deep Learning Acceleration [Floating-point, pre-alignment]
- (15.6) DIANA: An End-to-End Energy-Efficient DIgital and ANAlog Hybrid Neural Network SoC
- (16.1) DIMC: 2219TOPS/W 2569F2/b Digital In-Memory Computing Macro in 28nm Based on Approximate Arithmetic Hardware
- (29.3) A 28nm 15.59μJ/Token Full-Digital Bitline-Transpose CIM-Based Sparse Transformer Accelerator with Pipeline/Parallel Reconfigurable Modes [Transpose SRAM]

### 2022 VLSI

- A 12nm 121-TOPS/W 41.6-TOPS/mm2 All Digital Full Precision SRAM-based Compute-in-Memory with Configurable Bit-width For AI Edge Applications

### 2022 MICRO

- Multi-Layer In-Memory Processing [Compiler]

### 2022 DAC

- Processing-in-SRAM acceleration for ultra-low power visual 3D perception [Visual odometry]

### 2022 GLSVLSI

- Energy-Efficient In-SRAM Accumulation for CMOS-based CNN Accelerators

### 2022 JSSC

- Two-Way Transpose Multibit 6T SRAM Computing-in-Memory Macro for Inference-Training AI Edge Chips

### 2022 TC

- Eidetic: An In-Memory Matrix Multiplication Accelerator for Neural Networks

### 2021 ISSCC

- (15.1) A Programmable Neural-Network Inference Accelerator Based on Scalable In-Memory Computing [Charge-domain]
- (15.2) A 2.75-to-75.9TOPS/W Computing-in-Memory NN Processor Supporting Set-Associate Block-Wise Zero Skipping and Ping-Pong CIM with imultaneous Computation and Weight Updating
- (15.4) A 5.99-to-691.1TOPS/W Tensor-Train In-Memory-Computing Processor Using Bit-Level-Sparsity-Based Optimization and Variable-Precision Quantization
- (16.3) A 28nm 384kb 6T-SRAM Computation-in-Memory Macro with 8b Precision for AI Edge Chips
- (16.4) An 89TOPS/W and 16.3TOPS/mm2 All-Digital SRAM-Based Full-Precision Compute-In Memory Macro in 22nm for Machine-Learning Edge Applications [TSMC]

### 2021 VLSI

- A 13.7 TFLOPS/W Floating-point DNN Processor using Heterogeneous Computing Architecture with Exponent-Computing-in-Memory [Floating-point]
- PIMCA: A 3.4-MB Programmable In-Memory Computing Accelerator in 28nm For On-Chip DNN Inference

### 2021 ASSCC

- A 16Kb Transpose 6T SRAM In-Memory-Computing Macro based on Robust Charge-Domain Computing

### 2021 JSSC

- (Extension version of X. Si et al. *ISSCC*, pp. 246-248, 2020) A Local Computing Cell and 6T SRAM-Based Computing-in-Memory Macro With 8-b MAC Operation for Edge AI Chips

### 2020 ISSCC

- (15.2) A 28nm 64Kb Inference-Training Two-Way Transpose Multibit 6T SRAM Compute-in-Memory Macro for AI Edge Chips
- (15.3) A 351TOPS/W and 372.4GOPS Compute-in-Memory SRAM Macro in 7nm FinFET CMOS for Machine-Learning Applications
- (15.5) A 28nm 64Kb 6T SRAM Computing-in-Memory Macro with 8b MAC Operation for AI Edge Chips

### 2020 DAC

- Bit parallel 6T SRAM in-memory computing with reconfigurable bit-precision [BL boosting]

### 2020 ISCAS

- Towards a Reconfigurable Bit-Serial/Bit-Parallel Vector Accelerator using In-Situ Processing-In-SRAM

### 2020 JSSC

- A Programmable Heterogeneous Microprocessor Based on Bit-Scalable In-Memory Computing
- (extension version of J. Wang et al. *ISSCC*, pp. 224-226, 2019) A 28-nm Compute SRAM With Bit-Serial Logic/Arithmetic Operations for Programmable In-Memory Vector Computing [General-purpose]

### 2020 TC

- CIMAT: A Compute-In-Memory Architecture for On-chip Training Based on Transpose SRAM Arrays [Transpose SRAM]

### 2020 SSC-L

- A 35.6 TOPS/W/mm² 3-Stage Pipelined Computational SRAM With Adjustable Form Factor for Highly Data-Centric Applications

### 2019 ISSCC

- (7.5) A 65nm 0.39-to-140.3TOPS/W 1-to-12b Unified NeuralNetwork Processor Using Block-Circulant-Enabled Transpose-Domain Acceleration with 8.1× Higher TOPS/mm2 and 6T HBST-TRAM-Based 2D Data-Reuse Architecture [Transpose SRAM]
- (14.2) A Compute SRAM with Bit-Serial Integer/Floating-Point Operations for Programmable In-Memory Vector Acceleration [General-purpose, floating-point, transpose SRAM]
- (24.5) A Twin-8T SRAM Computation-In-Memory Macro for Multiple-Bit CNN-Based Machine Learning

### 2019 ISCA

- Duality cache for data parallel acceleration [Compiler]

### 2019 HPCA

- Bit Prudent In-Cache Acceleration of Deep Convolutional Neural Networks

### 2019 ESSCIRC

- A 1-16b Precision Reconfigurable Digital In-Memory Computing Macro Featuring Column-MAC Architecture and Bit-Serial Computation

### 2019 JSSC

- A 64-Tile 2.4-Mb In-Memory-Computing CNN Accelerator Employing Charge-Domain Compute [Charge-domain]

### 2019 TVLSI

- 8T SRAM Cell as a Multibit Dot-Product Engine for Beyond Von Neumann Computing

### 2018 ISSCC

- (31.1) Conv-RAM: An energy-efficient SRAM with embedded convolution computation for low-power CNN-based machine learning applications
- (31.2) A 42pJ/Decision 3.12TOPS/W Robust In-Memory Machine Learning Classifier with On-Chip Training
- (31.5) A 65nm 4Kb Algorithm-Dependent Computing-in-Memory SRAM Unit-Macro with 2.3ns and 55.8TOPS/W Fully Parallel Product-Sum Operation for Binary DNN Edge Processors

### 2018 ISCA

- PROMISE: An End-to-End Design of a Programmable Mixed-Signal Accelerator for Machine-Learning Algorithms
- Neural Cache: Bit-Serial In-Cache Acceleration of Deep Neural Networks
- Bit Fusion: Bit-Level Dynamically Composable Architecture for Accelerating Deep Neural Network

### 2018 JSSC

- Recryptor: A Reconfigurable Cryptographic Cortex-M0 Processor With In-Memory and Near-Memory Computing for IoT Security

### 2017 ISSCC

- (14.6) A 0.62mW ultra-low-power convolutional-neural-network face-recognition processor and a CIS integrated with always-on haar-like face detector [Transpose SRAM]

### 2017 MICRO

- Cache Automaton

### 2017 HPCA

- Compute Caches

### 2017 JSSC

- In-Memory Computation of a Machine-Learning Classifier in a Standard 6T SRAM Array

### 2016 JSSC

- A 28 nm Configurable Memory (TCAM/BCAM/SRAM) Using Push-Rule 6T Bit Cell Enabling Logic-in-Memory
