# Comparative Analysis of YOLOv8 and Advanced Feature Extraction Networks (DFEM-Net, Bhaskar-Net)

## 📌 Abstract

This repository hosts a comprehensive comparative study between the state-of-the-art **YOLOv8** object detection model and two novel custom architectures: **DFEM-Net** (Dynamic Adaptive Feature Extraction Network) and **BHASKAR-Net** (Boosted Hybrid Attention with Sparse Kernels for Adaptive Recognition). The project aims to evaluate the efficacy of integrating advanced mechanisms such as Deformable Convolutions (DCN), 3D Convolution-based Multi-scale Fusion (Scalseq), and Triple-Encoding Feature Boosting (Zoomcat) into the YOLO framework.

The primary focus is on enhancing detection performance in challenging scenarios involving:
- **Small Object Detection**
- **Heavy Occlusion**
- **Complex Environmental Conditions** (e.g., Fog, Rain)

## 📂 Repository Structure

The codebase is organized into three distinct modules, each representing a specific model architecture or baseline:

```
CompareYOLOv8_DFEM_NET_-_Bhaskar/
├── YOLOv8/           # Baseline Ultralytics YOLOv8 implementation
├── DFEM_Net/         # Implementation of DFEM-Net with TuaBackbone, Scalseq, and Zoomcat
│   ├── dfem_net.yaml # Model configuration
│   ├── benchmark.py  # Latency benchmarking script
│   └── ...           # Custom modules (TuaBottleneck, etc.)
└── Bhaskar_NET/      # Implementation of Bhaskar-Net (Variant of DFEM-Net)
    ├── dfem_net.yaml # Model configuration
    └── ...
```

## 🧠 Methodologies & Architectures

### 1. DFEM-Net (Dynamic Adaptive Feature Extraction Network)
DFEM-Net introduces a paradigm shift from standard CNNs by incorporating:
- **TuaNet Backbone**: Replaces standard convolutions with **Deformable Convolutions (DCN)** and **TuaAttention** mechanisms to adaptively model geometric transformations.
- **Scalseq (Scale Sequence)**: A novel neck architecture that stacks multi-scale features (P3, P4, P5) into a 3D tensor and applies **3D Convolutions** to learn inter-scale correlations effectively.
- **Zoomcat**: A specialized module for retaining high-frequency details, crucial for small object detection.

### 2. BHASKAR-Net (Boosted Hybrid Attention with Sparse Kernels for Adaptive Recognition)
**BHASKAR-Net** represents a further evolution of the adaptive feature extraction paradigm. It is designed to optimize the trade-off between computational efficiency and detection accuracy by integrating:
- **Boosted Hybrid Attention**: A composite attention mechanism that refines feature selection across multiple scales.
- **Sparse Kernels**: Utilizes efficient, sparse structural elements (inspired by Deformable Convolutions) to reduce redundancy while maintaining receptive field integrity.
- **Adaptive Recognition**: dynamically adjusts its feature response based on the input scene's complexity (e.g., occlusion levels).

## 🚀 Installation & Requirements

Ensure you have a Python 3.8+ environment. Install the necessary dependencies:

```bash
pip install torch torchvision ultralytics numpy
```

## 🛠️ Usage

### Running Benchmarks
Each model directory contains a `benchmark.py` script to evaluate inference latency and throughput.

**To benchmark DFEM-Net:**
```bash
cd DFEM_Net
python benchmark.py
```

**To benchmark Bhaskar-Net:**
```bash
cd Bhaskar_NET
python benchmark.py
```

### Model Configuration
The model architectures are defined in `dfem_net.yaml` within their respective directories. These YAML files follow the Ultralytics configuration format but utilize custom modules (`TuaBottleneck`, `Scalseq`, `Zoomcat`) which are dynamically registered at runtime.

### Inference & Training
You can use the standard Ultralytics CLI or Python API, provided you import the custom modules first (as done in the `main_full.py` and `benchmark.py` scripts).

**Python Example:**
```python
import sys
# Add the module directory to path if necessary or run from within the directory
# sys.path.append('DFEM_Net')

from ultralytics import YOLO
# Ensure custom modules are imported/registered here
# (See main_full.py in subdirectories for registration logic)

model = YOLO("DFEM_Net/dfem_net.yaml")
model.train(data="coco128.yaml", epochs=100)
```

## 📊 Performance Analysis

Preliminary benchmarking on Intel i3-1215U (CPU) reveals important insights regarding the computational cost of advanced feature extraction:
- **Baseline YOLOv8n**: Achieves extremely low latency (~80-120ms), benefiting from highly optimized 2D convolutions.
- **DFEM-Net / Bhaskar-Net**: Exhibit higher latency due to the presence of unoptimized Deformable Convolutions and 3D Convolutions on CPU. These architectures are recommended for **GPU-accelerated environments** where parallelization can be fully leveraged.

## 📜 Citation

If you use this code or architecture in your research, please cite the original DFEM-Net paper:

> *"DFEM-Net: A dynamic adaptive feature extraction network based deep learning model for pedestrian and vehicle detection"* (2026).

## 🤝 Acknowledgments
- **Ultralytics** for the YOLOv8 framework.
- The authors of the DFEM-Net paper for the structural concepts.

---
*This repository is for research and educational purposes.*
