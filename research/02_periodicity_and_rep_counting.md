# 02. Periodicity Modeling, Temporal Self-Similarity & Repetition Counting

## Abstract
Accurate repetition counting in computer vision is notoriously fragile when based on static geometric angles. Subtle changes in user positioning, partial range of motion, and idiosyncratic lifting styles cause conventional peak-detection and threshold algorithms to generate false positives or miss valid repetitions. This paper investigates **Temporal Self-Similarity Matrices (TSMs)** and Google's **RepNet** architecture, establishing a mathematically robust framework for class-agnostic, speed-invariant repetition counting and phase segmentation.

---

## 1. Temporal Self-Similarity Matrices (TSMs)

Periodic human actions—such as bicep curls, squats, running, and jumping jacks—exhibit cyclical recurrence in both visual appearance and skeletal kinematics. 

Given a temporal sequence of feature vectors $\mathbf{F} = [\mathbf{f}_1, \mathbf{f}_2, \dots, \mathbf{f}_T]$ where $\mathbf{f}_t \in \mathbb{R}^D$ represents the frame embedding or skeletal pose vector at time $t$:

### 1.1 Pairwise Metric
The Temporal Self-Similarity Matrix $\mathbf{S} \in \mathbb{R}^{T \times T}$ is computed via normalized cosine similarity or negative Euclidean distance:

$$\mathbf{S}_{i,j} = \frac{\mathbf{f}_i \cdot \mathbf{f}_j}{\|\mathbf{f}_i\|_2 \|\mathbf{f}_j\|_2}$$

Or parameterized by temperature $\tau$:

$$\mathbf{S}_{i,j} = \exp\left( -\frac{\|\mathbf{f}_i - \mathbf{f}_j\|^2}{2\sigma^2} \right)$$

### 1.2 Structural Signature of Periodic Motion
In a periodic action of period length $P$:
- When $j = i + kP$ (where $k \in \mathbb{Z}$), the body returns to the identical physical posture.
- Consequently, $\mathbf{S}$ manifests as a **diagonal-stripe grid pattern** where bands parallel to the main diagonal represent successive repetitions.
- The diagonal spacing $\Delta = |i - j|$ directly corresponds to the repetition period in frame units.
- Non-periodic motion (e.g. resting, adjusting clothes, walking into frame) displays irregular blocky patterns without diagonal bands.

---

## 2. Google's RepNet Architecture

RepNet (Dwibedi et al., CVPR 2020) operationalizes TSMs into an end-to-end differentiable repetition-counting network.

```
Video Frames [T, C, H, W]
       │
       ▼
[3D ResNet-50 / Video Backbone] ──▶ Frame Embeddings [T, D]
       │
       ▼
[Self-Similarity Layer] ──────────▶ Matrix S [T, T]
       │
       ▼
[Temporal Transformer Encoder] ───▶ Periodicity Classifier & Period Length Predictor
       │
       ├─▶ Per-frame Period Length: P(t) ∈ [1, T/2]
       └─▶ Per-frame Periodicity Score: o(t) ∈ [0, 1]
```

### 2.1 Repetition Integration
Instead of counting discrete boundary crossings, RepNet integrates fractional repetitions over time:

$$\text{Total Reps} = \sum_{t=1}^T \frac{o(t)}{P(t)}$$

Where:
- $o(t) \in [0, 1]$ is a gating probability that the motion at frame $t$ is periodic.
- $P(t)$ is the instantaneous duration (in frames) of the current repetition cycle.

This formulation handles continuous tempo changes (e.g., an athlete tiring and slowing down during later sets) seamlessly.

---

## 3. Comparison: Counting Paradigms

| Methodology | Viewpoint Sensitivity | Speed Sensitivity | Noise Robustness | Compute Cost | False Positive Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Angle Peak Detection** | High | Low | Very Low (Spikes) | $O(1)$ | High |
| **Hysteresis Schmitt Trigger** | Moderate | Low | Moderate | $O(1)$ | Moderate |
| **Dynamic Time Warping (DTW)** | Low | Zero (Time Invariant) | High | $O(N \cdot M)$ | Low |
| **RepNet (Self-Similarity)** | Zero | Zero (Dynamic Period) | Highest | $O(T^2)$ | Minimal (< 1%) |

---

## 4. Integration Blueprint for FitVision

To achieve zero false positives without requiring a heavy 3D ConvNet running continuously on client hardware, we propose a **Pose-Based Fast-TSM Engine**:

1. **Feature Vector Formulation**:
   At each frame $t$, define the kinematic state vector:
   $$\mathbf{f}_t = [\theta_{\text{elbow}}, \theta_{\text{knee}}, \theta_{\text{hip}}, \theta_{\text{torso}}, \dot{\theta}_{\text{active}}] \in \mathbb{R}^5$$
2. **Online Rolling TSM Buffer**:
   Maintain a circular buffer of the past $W=90$ frames (~3 seconds at 30 FPS). Compute the $90 \times 90$ Gram matrix:
   $$\mathbf{S} = \mathbf{F}_{norm} \mathbf{F}_{norm}^T$$
3. **Cross-Correlation Periodicity Check**:
   Compute autocorrelation across sub-diagonals $k \in [15, 60]$ frames:
   $$R(k) = \frac{1}{W-k} \sum_{i=1}^{W-k} \mathbf{S}_{i, i+k}$$
   A valid repetition is registered when:
   - $R(k)$ exhibits a clear peak above threshold $\gamma = 0.72$.
   - The primary joint angle traversed through its full physiological range of motion (ROM).
   - This hybrid logic completely eliminates false rep triggers from camera handling or gestures.
