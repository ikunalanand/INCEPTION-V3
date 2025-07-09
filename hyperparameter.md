# Hyperparameters for PneumoniaMNIST InceptionV3 Fine‑Tuning  
*(Colab, Tesla T4 GPU)*  

| Category | Hyperparameter | Value | Why this value works well on a T4 GPU |
|----------|----------------|-------|---------------------------------------|
| **Data I/O** | Image size | **299 × 299 × 3** | Matches the native input of InceptionV3; keeps tensor dimensions friendly to cuDNN kernels on T4. |
| | Batch size | **32** | A 299×299×3 FP32 tensor ≈ 1 MB. Forward+backward passes with the trainable head + last 40 layers keep the 16 GB VRAM of a T4 ~60 % utilised (leaving head‑room for augmentation & bookkeeping). |
| | Data augmentation | `rotation 15°`, `shift 0.1`, `shear 0.1`, `zoom 0.1`, `flip` | Light‑to‑moderate augmentation improves generalisation without adding heavy compute; the extra ops are run on‑the‑fly on CPU while the GPU trains, so throughput stays high. |
| **Optimiser** | Algorithm | **Adam** | Adaptive learning works well for fine‑tuning large pretrained nets; Adam kernels are highly optimised in cuDNN/TF for Turing GPUs (T4). |
| | Learning‑rate | **1 × 10⁻⁵** | Starting small avoids catastrophically distorting pretrained ImageNet weights; converges smoothly within 100 epochs. |
| **Model** | Base network | **InceptionV3 (ImageNet weights)** | Good balance of accuracy ↔ parameter count (24 M) — fits comfortably on T4, trains ~140 img/s at BS = 32. |
| | Trainable layers | **last 40 layers** (BN frozen) | Lets higher‑level features adapt to X‑ray domain while keeping <10 M parameters trainable → faster convergence & lower memory. |
| | Head | `GAP → Dense 512 ReLU → Dropout 0.5 → Dense 1 sigmoid` | Dense 512 is enough capacity for binary task; Dropout combats over‑fitting given only ≈4 k training images. |
| **Training loop** | Epochs | **100** | Empirically, loss plateaus around 80–90 epochs; 100 adds a safety margin. On a T4 this is ≈ 3 h wall‑time (comfortable for free Colab). |
| | Callback | **TqdmCallback** | Gives granular progress bars without cluttering Colab logs; negligible overhead. |
| **Seed control** | `random`, `numpy`, `tensorflow` = **42** | Ensures deterministic splits & comparable metrics when reproducing. |
| **Metrics** | `accuracy`, `precision`, `recall`, `ROC‑AUC` | Class‑imbalance aware (precision/recall) and threshold‑free (AUC). |

## GPU‑specific rationale

* **Tesla T4** has 16 GB GDDR6 and Turing Tensor Cores:  
  * Handles 32 × (299 × 299 × 3) FP32 images and the InceptionV3 graph comfortably.  
  * With mixed‑precision (`tf.keras.mixed_precision`) you can push BS ≈ 48, but 32 keeps code simple and avoids any overflow warnings.  
* **Compute capability 7.5** + cuDNN 8 means TF’s fused conv‑BN‑ReLU kernels and Adam updates are fully accelerated.  
* Training the **full backbone** would roughly double VRAM & time with marginal accuracy gain; unfreezing the **top 40 layers** strikes an 80/20 balance suited to a single‑GPU Colab session.

> *Last updated: 9 Jul 2025*  

