# Domain Generalization Methods

## 🔍 Overview
- **Survey**: [A Survey of Data Augmentation in Domain Generalization](https://link.springer.com/article/10.1007/s11063-025-11747-9)
- **Note**: Most cited papers are listed below across domain-level and image-level augmentation strategies.

---

## 🧠 Domain-Level Methods

### 🎨 Generative Modeling-Based

#### **GAN-Based**
- **DLOW**: [Domain Flow for Adaptation and Generalization (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gong_DLOW_Domain_Flow_for_Adaptation_and_Generalization_CVPR_2019_paper.pdf)  
  _Citations: 447_
  - transforms images from source domain to target domain (with intermediate states = mixture of source and target domain)
  - used in paper for:
    - cross-domain semantic segmentation
      - Experiments:
        - 2 Datasets (GTA5, Cityscapes)
        - apply DFLOW to training images to improve results
    - style generalization
      Experiments:
      - transform source Image in target domains
      - measure User preference of transformed images compared to other such methods 
  - code available on [GitHub](https://github.com/ETHRuiGong/DLOW)
  - **Critical Thinking**
    - does it really force the model to learn domain invariant features and thus improve domain generalization or will this method only expand the domain space
<br/><br/>
- **Domain Randomization + Pyramid Consistency**:  
  [~~Simulation-to-Real Generalization Without Accessing Target Domain Data (ICCV 2019)~~](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yue_Domain_Randomization_and_Pyramid_Consistency_Simulation-to-Real_Generalization_Without_Accessing_Target_ICCV_2019_paper.pdf)  
  _Citations: 490_

### 🌀 Domain Randomization-Based
- [~~Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World (2017)~~](https://arxiv.org/pdf/1703.06907)  
  _Citations: 3,808_
- [~~Training Deep Networks with Synthetic Data: Bridging the Reality Gap (CVPRW 2018)~~](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w14/Tremblay_Training_Deep_Networks_CVPR_2018_paper.pdf)  
  _Citations: 1,159_

### ⚔️ Adversarial Training-Based
- [Cross-Gradient Training for Generalizing Across Domains (2018)](https://arxiv.org/pdf/1804.10745)  
  _Citations: 641_
  - Experiments on character recognition (font = domain)
- [Generalizing to Unseen Domains via Adversarial Data Augmentation (NeurIPS 2018)](https://proceedings.neurips.cc/paper/2018/file/1d94108e907bb8311d8802b48fd54b4a-Paper.pdf)  
  _Citations: 969_
  - transform a image to a fictive distribution
  - could be applied but looks complicated
#### 🕸️ Adversarial Network
- [~~Learning to Learn Single Domain Generalization (CVPR 2020)~~](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qiao_Learning_to_Learn_Single_Domain_Generalization_CVPR_2020_paper.pdf)  
  _Citations: 566_


### 📶 Frequency-Domain-Based
- [A Fourier-Based Framework for Domain Generalization (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_A_Fourier-Based_Framework_for_Domain_Generalization_CVPR_2021_paper.pdf)  
  _Citations: 566_
  - Idea: Fourier Phase contains high level semantics more resistant to domain shift
  - Interpolation of Fourier Phase of 2 Images
  - Method Teacher-Student Model with ResNet
  - Code available on [GitHub](https://github.com/MediaBrain-SJTU/FACT)

---

## 🖼️ Image-Level Methods

### 🧪 Traditional Augmentation
- [A Simple Framework for Contrastive Learning of Visual Representations (SimCLR, ICML 2020)](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf)  
  _Citations: 23,663_
- 
  - code available on [GitHub](https://github.com/google-research/simclr)
- [Image Augmentation is All You Need: Regularizing Deep Reinforcement Learning from Pixels (2021)](https://openreview.net/pdf?id=GY6-6sTvGaf)  
  _Citations: 525_

### 🧩 Self-Supervised Learning-Based
- [Domain Generalization by Solving Jigsaw Puzzles (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Carlucci_Domain_Generalization_by_Solving_Jigsaw_Puzzles_CVPR_2019_paper.pdf)  
  _Citations: 1,068_

### 🧬 Mixup-Based
- [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty (2019)](https://arxiv.org/pdf/1912.02781)  
  _Citations: 1,584_
