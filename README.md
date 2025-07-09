# PneumoniaMNIST InceptionV3 Fine‑Tuning

This repo contains a **reproducible Colab workflow** for binary
classification of chest X‑ray images (pneumonia vs normal) using a
fine‑tuned InceptionV3 network.

## 1 . Set up environment
```bash
pip install -r requirements.txt
```

## 2 . Get the dataset
Download **`PneumoniaMNIST.npz`** from the official MedMNIST
repository and upload it when prompted in Colab.

## 3 . Training script
The Colab notebook (`train_pneumonia.ipynb`) trains for **100 epochs**
with data augmentation and a TQDM progress bar:

```python
!pip -q install --upgrade scikit-learn tqdm matplotlib numpy tensorflow
# ... (full script from the chat) ...
history = model.fit(
    train_gen,
    epochs=100,
    validation_data=val_gen,
    verbose=0,
    callbacks=[TqdmCallback(verbose=1)]
)
```

## 4 . Reproducing results
* GPU: any NVIDIA GPU with ≥4 GB VRAM (Colab “GPU” runtime works).
* Expected test accuracy ≈ 0.88–0.90 after 100 epochs.
* See the notebook for learning‑curve plots, classification report,
  confusion matrix, ROC‑AUC, and a sample misclassified X‑ray.

## 5 . Tips
* Tune **batch size**, **learning rate**, or **unfreeze more layers** of
  InceptionV3 for potentially higher accuracy.
* For faster experimentation on CPU‑only machines, reduce image size to
  150×150 and replace InceptionV3 with MobileNetV2.

---

*Last updated: 9 July 2025*
