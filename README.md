# Perancangan-Model-Prediksi-Skor-RULA
#RULA Score Classification using Dual-Input CNN with Advanced Oversampling & Traditional Oversampling

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Repository ini berisi implementasi sistem klasifikasi skor RULA (Rapid Upper Limb Assessment) menggunakan arsitektur dual-input CNN dengan berbagai teknik fine-tuning dan oversampling advanced untuk menangani data imbalance.

## 📋 Abstract

Penelitian ini mengembangkan model deep learning untuk mengklasifikasikan skor RULA (1-7) berdasarkan gambar postur tubuh dan keypoints. Metode yang diusulkan menggunakan:
- **Dual-Input Architecture**: ResNet18 untuk image features + Keypoints coordinates
- **5 Fine-Tuning Strategies**: head_only, last_block, last_2blocks, last_3blocks, full_layer
- **Traditional Oversampling**: use 3000 & 8000 sample per class, last_3blocks fine tuning
- **Advanced Oversampling**: IB-GAN (Information Bottleneck Generative Adversarial Networks) dan MGVAE (Multi-class Guided Variational Autoencoder )

## 🚀 Features

- 🎯 **5 Fine-Tuning Strategies** untuk transfer learning optimal
- 🤖 **IB-GAN**: In-batch generative oversampling untuk minority classes
- 🔄 **MG-VAE**: Multi-class conditional VAE untuk data synthesis
- 📊 **Comprehensive Evaluation**: Accuracy, Confusion Matrix, Class-wise metrics
- ⚡ **Efficient Training**: Early stopping, LR scheduling, Gradient accumulation
