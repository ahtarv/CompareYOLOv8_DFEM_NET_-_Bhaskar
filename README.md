# ⚡ Comparative Analysis: YOLOv8 vs. DFEM-Net vs. BhaskarNet

## 📌 Abstract

This repository hosts a comprehensive comparative study between the state-of-the-art **YOLOv8** object detection model and two novel custom architectures designed for adverse weather traffic analysis:

1. **DFEM-Net** (Dynamic Adaptive Feature Extraction Network) - *The 2026 Research Baseline*
2. **BhaskarNet** (Balanced Hardware-aware Architecture for Speed, Kinetic Analysis, and Recognition) - *Our Novel Optimization*

The project evaluates the efficacy of integrating advanced mechanisms such as **Deformable Convolutions (DCN)** and **3D Convolution-based Multi-scale Fusion** into the YOLO framework. A key focus is the **hardware-aware optimization** of these heavy modules to enable edge deployment on standard CPUs.

## 📂 Repository Structure

```text
CompareYOLOv8_DFEM_NET_-_Bhaskar/
├── 01_YOLOv8_Baseline/       # Standard Industry Benchmark (Ultralytics)
│   └── benchmark_yolo.py     # Latency test for YOLOv8n
│
├── 02_DFEM_Net_Original/     # The "Heavy" Research Implementation
│   ├── dfem_net.yaml         # Config with 3D Convs & Full DCN
│   └── benchmark.py          # Demonstrates the high computational cost
│
└── 03_BhaskarNet_Optimized/  # The "Lite" Novel Architecture
    ├── bhaskar_net.yaml      # Config with Pseudo-3D & Dilated Kernels
    ├── benchmark.py          # Demonstrates the 10x speedup
    └── ...                   # Custom modules (TuaBottleneck, Scalseq)

```

## 🧠 Methodologies & Architectures

### 1. DFEM-Net (The Original Research)

A "Teacher" network designed for heavy occlusion and fog.

* **Mechanism:** Uses **3D Convolutions** in the neck (Scalseq) to fuse features across scales (P3, P4, P5) and **Deformable Convolutions** in the backbone.
* **The Problem:** Extremely high computational cost on CPUs due to non-contiguous memory access patterns in 3D operations.

### 2. BhaskarNet (Our Proposed Solution)

**BhaskarNet** (**B**alanced **H**ardware-aware **A**rchitecture for **S**peed, **K**inetic **A**nalysis, and **R**ecognition) is a novel optimization of DFEM-Net designed for edge devices.

* **Pseudo-3D Fusion:** Replaces heavy `Conv3d` layers with **Scale-Sequence Attention (1x1 Conv)** to reduce FLOPs by ~96%.
* **Sparse Kernels:** Replaces computationally expensive Deformable Convolutions with **Dilated Convolutions**, maintaining the receptive field (seeing through fog) without the offset calculation overhead.
* **Hardware Alignment:** Optimizes channel dimensions for CPU cache hierarchy.

## 📊 Performance Analysis (The "Big Table")

Benchmarks were conducted on an **Intel i3-1215U CPU** to simulate edge device performance (non-GPU environment).

| Architecture | Avg Latency (ms) | FPS | Stability | Speedup Factor |
| --- | --- | --- | --- | --- |
| **YOLOv8n (Baseline)** | **102 ms** | ~9.8 | High | (Reference) |
| **Original DFEM-Net** | 2,901 ms | 0.3 | Low (Memory Spikes) | 1x |
| **BhaskarNet (Ours)** | **274 ms** | **~3.6** | **High** | **10.6x Faster** |

> **Key Finding:** BhaskarNet achieves a **10.6x speedup** over the original DFEM-Net architecture while maintaining the advanced multi-scale fusion logic, making it viable for real-time traffic monitoring (3+ FPS) on standard hardware.

## 🚀 Installation & Usage

**1. Install Dependencies**

```bash
pip install torch torchvision ultralytics numpy matplotlib

```

**2. Run the Comparison Tournament**
To reproduce the benchmark table:

```bash
# Test the Baseline
cd 01_YOLOv8_Baseline
python benchmark_yolo.py

# Test the Optimization
cd ../03_BhaskarNet_Optimized
python benchmark.py

```

## 📜 Citation & Credits

If you use this code or architecture, please cite the original DFEM-Net paper:

> *"DFEM-Net: A dynamic adaptive feature extraction network based deep learning model for pedestrian and vehicle detection"* (2026).

**BhaskarNet** was developed as a specialized optimization study by [Your Name] at DJ Sanghvi College of Engineering.

---

*This repository is for research and educational purposes.*
