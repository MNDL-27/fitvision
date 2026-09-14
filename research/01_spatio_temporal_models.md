# 01. Spatio-Temporal Graph Neural Networks & Motion Transformers for Exercise Analysis

## Abstract
Traditional computer vision pipelines for fitness evaluation rely on hand-crafted 2D joint angle heuristics. While computationally lightweight, these methods suffer from viewpoint sensitivity, depth ambiguity, and an inability to model temporal rhythm and inter-joint coordination. This paper surveys and formulates the modern paradigm of **Spatio-Temporal Skeletal Modeling**, comparing **Spatial-Temporal Graph Convolutional Networks (ST-GCN, 2s-AGCN)** and **Motion Transformers (PoseFormer, MotionBERT)** for physical form evaluation and exercise segmentation.

---

## 1. Mathematical Formulation of Skeletal Graphs

A human body pose sequence across $T$ frames with $N$ joints is represented as a spatio-temporal graph $G = (V, E)$.
- **Node Set** $V = \{v_{t,i} \mid t = 1, \dots, T; i = 1, \dots, N\}$ represents joint $i$ at time $t$, with input features $F(v_{t,i}) \in \mathbb{R}^C$ (typically $x, y, z$ coordinates, visibility confidence $c$, and velocity $\Delta x, \Delta y$).
- **Edge Set** $E$ comprises two subsets:
  1. *Intra-frame skeletal edges* $E_S = \{(v_{t,i}, v_{t,j}) \mid (i, j) \in H\}$, where $H$ denotes physical anatomical bones.
  2. *Inter-frame temporal edges* $E_T = \{(v_{t,i}, v_{t+1,i})\}$, connecting the same physical joint across consecutive frames.

### 1.1 Spatial Graph Convolution
The spatial graph convolution operation on node $v_i$ at frame $t$ is formulated as:

$$f_{out}(v_{ti}) = \sum_{v_{tj} \in B(v_{ti})} \frac{1}{Z_{ti}(v_{tj})} f_{in}(v_{tj}) \cdot \mathbf{W}(l_{ti}(v_{tj}))$$

Where:
- $B(v_{ti})$ is the 1-hop neighboring node set of joint $i$.
- $l_{ti}: B(v_{ti}) \to \{0, 1, \dots, K-1\}$ partitions neighbors into spatial subsets (e.g., centrifugal, centripetal, and root partitions).
- $\mathbf{W}$ is the learnable weight tensor.
- $Z_{ti}$ is the normalization factor equivalent to the subset cardinality.

In matrix notation over the entire skeletal graph:

$$\mathbf{X}^{(l+1)} = \sigma \left( \sum_{k=0}^{K-1} \mathbf{\tilde{D}}_k^{-\frac{1}{2}} \mathbf{\tilde{A}}_k \mathbf{\tilde{D}}_k^{-\frac{1}{2}} \mathbf{X}^{(l)} \mathbf{W}_k \right)$$

Where $\mathbf{\tilde{A}}_k = \mathbf{A}_k + \mathbf{I}_N$ is the spatial adjacency matrix for partition $k$, and $\mathbf{\tilde{D}}_k$ is its degree matrix.

### 1.2 Two-Stream Adaptive GCN (2s-AGCN)
Standard ST-GCN uses fixed adjacency matrices determined solely by human anatomy. In physical rehabilitation and athletic training, implicit kinetic correlations (e.g., hip-knee coupling during squats) are critical. 2s-AGCN introduces an adaptive topology:

$$\mathbf{A}_k = \mathbf{A}_{static} + \mathbf{B}_k + \mathbf{C}_k$$

- $\mathbf{A}_{static}$: Physical anatomical connections.
- $\mathbf{B}_k$: Globally learned parameter matrix optimizing inter-joint correlations.
- $\mathbf{C}_k$: Sample-specific self-attention matrix computed via normalized embedded Gaussian:

$$\mathbf{C}_k(i, j) = \frac{\exp \left( \theta(\mathbf{x}_i)^T \phi(\mathbf{x}_j) \right)}{\sum_{m=1}^N \exp \left( \theta(\mathbf{x}_i)^T \phi(\mathbf{x}_m) \right)}$$

---

## 2. Motion Transformers (PoseFormer & MotionBERT)

### 2.1 Spatial-Temporal Self-Attention
Rather than constraining receptive fields through graph convolutions, Motion Transformers apply self-attention across both spatial and temporal axes:

1. **Spatial Transformer Block**: Embeds all $N=33$ landmarks in frame $t$. Joints attend to each other regardless of anatomical distance, capturing full-body posture balance:
   $$\mathbf{Z}_S = \text{Softmax}\left(\frac{\mathbf{Q}_S \mathbf{K}_S^T}{\sqrt{d_k}}\right) \mathbf{V}_S$$
2. **Temporal Transformer Block**: Operates across the sequence of frame embeddings $T$, learning cadence, acceleration phases (concentric vs. eccentric), and velocity profiles.

### 2.2 Pre-training with Masked Motion Autoencoders (MotionBERT)
MotionBERT trains on massive unlabeled motion capture datasets (AMASS, Human3.6M) by randomly masking 30–50% of 3D skeletal joints and learning to reconstruct the full kinematic trajectory. The pre-trained latent space encodes valid physiological kinematics, making downstream fine-tuning for exercise anomaly detection robust against camera noise.

---

## 3. Comparative Architecture Analysis

| Architecture | Paradigm | Parameter Count | Latency (CPU) | Latency (TensorRT) | Accuracy (NTU-60) | Suitability for Real-Time Client |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Heuristic Trigonometry** | Direct Vector Math | 0 | < 0.1 ms | N/A | ~70% (Brittle) | Highest (Immediate) |
| **TCN (Temporal ConvNet)** | 1D Dilated Convolutions | ~0.4 M | 3.2 ms | 0.8 ms | 86.4% | Very High |
| **ST-GCN** | Spatio-Temporal Graph CNN | 3.1 M | 14.5 ms | 2.1 ms | 88.7% | High (ONNX/WASM) |
| **2s-AGCN** | Adaptive Graph Attention | 6.9 M | 28.0 ms | 4.2 ms | 91.2% | Moderate |
| **PoseFormer** | Spatial-Temporal Transformer | 9.6 M | 68.0 ms | 9.8 ms | 92.4% | Server / Cloud Edge |
| **MotionBERT** | Dual-Stream Pretrained VLM | 32.4 M | 220.0 ms | 26.0 ms | 94.8% | Offline / Post-Workout |

---

## 4. Implementation Strategy for FitVision

To maintain zero-latency client performance while capturing full kinematic intelligence, we recommend a **Hybrid Hierarchical Architecture**:
1. **Edge/Client Level (60 FPS)**:
   - MediaPipe BlazePose extracts $N=33$ landmarks.
   - An exported lightweight ONNX **Spatial-Temporal TCN** (128 channels, 4 residual blocks) runs directly in the browser via `onnxruntime-web` with WebGL/WebGPU backend.
   - Evaluates real-time rep cadence and flags anomalous kinetic spikes.
2. **Cloud/NIM Level (Asynchronous Keyframe Audit)**:
   - On rep completion or anomaly detection, sends the multi-frame sequence to **NVIDIA NIM** for biomechanical interpretation and natural-language coaching.
