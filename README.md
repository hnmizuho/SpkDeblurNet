# The official implementation code of Enhancing Motion Deblurring in High-Speed Scenes with Spike Streams (NeurIPS 2023).
> Traditional cameras produce desirable vision results but struggle with motion blur in high-speed scenes due to long exposure windows. Existing frame-based deblurring algorithms face challenges in extracting useful motion cues from severely blurred images. Recently, an emerging bio-inspired vision sensor known as the spike camera has achieved an extremely high frame rate while preserving rich spatial details, owing to its novel sampling mechanism. However, typical binary spike streams are relatively low-resolution, degraded image signals devoid of color information, making them unfriendly to human vision. In this paper, we propose a novel approach that integrates the two modalities from two branches, leveraging spike streams as auxiliary visual cues for guiding deblurring in high-speed motion scenes. We propose the first spike-based motion deblurring model with bidirectional information complementarity. We introduce a content-aware motion magnitude attention module that utilizes learnable mask to extract relevant information from blurry images effectively, and we incorporate a transposed cross-attention fusion module to efficiently combine features from both spike data and blurry RGB images. Furthermore, we build two extensive synthesized datasets for training and validation purposes, encompassing high-temporal-resolution spikes, blurry images, and corresponding sharp images. The experimental results demonstrate that our method effectively recovers clear RGB images from highly blurry scenes and outperforms state-of-the-art deblurring algorithms in multiple settings.

## Datasets
[Spk-GoPro](https://pan.baidu.com/s/13j4NLpyrrEL1VH2wgiaGng?pwd=kxva)   [Spk-X4K1000FPS](https://pan.baidu.com/s/1XryVqgbrknUU6LGyPHX3Lg?pwd=n3ss)

## Quick Start
```
python test_real.py
```

## Usage
```
python train.py -opt options/train/train_deblur.yml
```

```
python valid.py -opt options/test/test_deblur.yml
```

```
python test.py -opt options/test/test_deblur.yml
```

## Citation
```
@article{chen2024enhancing,
  title={Enhancing Motion Deblurring in High-Speed Scenes with Spike Streams},
  author={Chen, Shiyan and Zhang, Jiyuan and Zheng, Yajing and Huang, Tiejun and Yu, Zhaofei},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

pth