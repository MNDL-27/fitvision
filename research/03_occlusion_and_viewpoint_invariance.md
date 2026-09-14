# 03. 2D-to-3D Pose Lifting, Viewpoint Invariance, & Occlusion Compensation

## Abstract
Single-camera monocular motion detection is subject to two major sources of geometric distortion: **perspective foreshortening** (where joint angles appear compressed depending on camera elevation or obliquity) and **out-of-frame occlusion** (where limbs disappear due to camera framing or body self-occlusion). This paper analyzes mathematical solutions to achieve true viewpoint-invariant joint tracking through **2D-to-3D Pose Lifting (VideoPose3D, SemGCN)**, **Procrustes Anatomical Normalization**, and **Kinematic Inverse Kinematics (IK) Constraint Solvers**.

---

## 1. 2D-to-3D Pose Lifting

Monocular 2D keypoints $(x, y)$ discard depth $z$. A 2D elbow angle observed from a 45° oblique camera angle can register as 110° when the true 3D anatomical angle is 90°.

### 1.1 Temporal Dilated Convolutional Lifting (VideoPose3D)
Pioneered by Pavllo et al. (CVPR 2019), VideoPose3D exploits temporal context to disambiguate depth:

Given 2D input trajectories $\mathbf{X} \in \mathbb{R}^{T \times 2N}$, a sequence of 1D temporal convolutions with exponentially increasing dilation factors ($d = 1, 3, 9, 27$) generates a receptive field spanning up to 243 frames without losing temporal resolution:

$$\hat{\mathbf{Y}}_t = \mathcal{F}_{\text{dilated}}(\mathbf{X}_{t - B : t + B}) \in \mathbb{R}^{3N}$$

Where $\hat{\mathbf{Y}}_t$ represents the estimated 3D root-relative coordinates $(x, y, z)$ for all $N$ joints.

### 1.2 Loss Formulation with Kinematic Priors
Training combines 3D Mean Per Joint Position Error (MPJPE) with bone length consistency constraints:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \|\mathbf{y}_i - \hat{\mathbf{y}}_i\|_2 + \lambda_{\text{bone}} \sum_{(j, k) \in E} \left| \|\hat{\mathbf{y}}_j - \hat{\mathbf{y}}_k\|_2 - L_{j,k} \right| + \lambda_{\text{smooth}} \|\hat{\mathbf{y}}_t - 2\hat{\mathbf{y}}_{t-1} + \hat{\mathbf{y}}_{t-2}\|_2$$

The bone length penalty $L_{j,k}$ enforces physical rigidity: human upper arm and femur bones do not stretch during movement.

---

## 2. Viewpoint & Scale Invariance Normalization

To make an exercise evaluation engine completely agnostic to whether the user is 1 meter or 3 meters from the camera, or whether the phone is resting on the floor tilted upward:

### 2.1 Anatomical Torso Scaling
1. Compute the shoulder center $S_{mid} = \frac{\mathbf{P}_{11} + \mathbf{P}_{12}}{2}$ and hip center $H_{mid} = \frac{\mathbf{P}_{23} + \mathbf{P}_{24}}{2}$.
2. Define the Torso Metric Reference $L_{\text{torso}} = \|S_{mid} - H_{mid}\|_2$.
3. Translate all joint coordinates to the mid-hip origin:
   $$\mathbf{P}'_i = \frac{\mathbf{P}_i - H_{mid}}{L_{\text{torso}}}$$
All subsequent distances, velocities, and bounding boxes are now expressed in **Torso Units (TU)**.

### 2.2 Camera Tilt / Pitch Compensation
When the camera is angled upward (e.g. laptop on a low desk or phone on the ground):
1. The vertical gravity vector $\mathbf{g}$ is inferred from the vector between the midpoint of the shoulders and midpoint of the ankles during the starting neutral stance.
2. An affine rotation matrix $\mathbf{R}_{\text{tilt}}$ aligns the estimated gravity vector with the world vertical axis $[0, 1]^T$:
   $$\mathbf{P}_{\text{aligned}} = \mathbf{R}_{\text{tilt}} \mathbf{P}'$$
3. Joint angles computed on $\mathbf{P}_{\text{aligned}}$ reflect pure anatomical biomechanics, free of camera tilt artifacts.

---

## 3. Occlusion Handling & Missing Joint Compensation

When a user exercises at close range or in constrained environments, feet, knees, or wrists periodically exit the video frame.

### 3.1 Strict Visibility Filtering vs. Kinematic Imputation
1. **Rule of Integrity**: Never guess or compute metrics for occluded joints unless specifically required.
   - Visibility confidence threshold: $c \ge 0.55$.
   - Any angle dependent on a joint with $c < 0.55$ must return `None` rather than an interpolated guess.
2. **Kinematic Boundary Clamping (Forward Kinematics)**:
   When joints must be estimated for continuous rendering:
   - Apply physical anatomical constraints:
     - Knee flexion: $[0^\circ, 160^\circ]$ (cannot invert backwards).
     - Elbow flexion: $[0^\circ, 155^\circ]$.
     - Neck rotation: $[-80^\circ, +80^\circ]$.
   - A predicted landmark violating these constraints is projected onto the nearest physiologically admissible manifold.

### 3.2 Unscented Kalman Filtering (UKF) for Limbs
For brief occlusions (< 6 frames / 200 ms, such as an arm swinging behind a dumbbell):
A Constant Acceleration kinematic UKF models the joint state:

$$\mathbf{x}_t = [p_x, p_y, v_x, v_y, a_x, a_y]^T$$

When visibility drops, the filter runs in pure prediction mode, coasting the trajectory smoothly until visual re-acquisition occurs.

---

## 4. Implementation Checklist for FitVision

- [x] Strict visibility gating ($c \ge 0.55$) in `movement_analyzer.py` to eliminate ghost limb calculations.
- [x] Torso-height normalization for distance-invariant progress curves.
- [ ] Export a lightweight VideoPose3D ONNX model for 2D-to-3D coordinate lifting.
- [ ] Integrate anatomical limit clamping for zero-jitter rendering during dumbbell occlusions.
