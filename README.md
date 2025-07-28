> **본 프로젝트는 한화에어로스페이스 스마트 국방 데이터 분석과정 3기의 '4채널을 활용한 위장 객체 탐지 고도화' 프로젝트의 서브프로젝트로 수행되었습니다.**
<img src="imgs/black_poodle.gif" align="right" width="384">


<br><br><br>
<br><br><br>

# Thermal-CycleGAN
**Thermal-CycleGAN**은 RGB 이미지와 해당 Segmentation Map을 결합한 4채널을 입력으로 받아 대응되는 IR(열화상) 이미지를 생성하는 Unpaired Image-to-Image Translation 모델입니다.

CycleGAN 아키텍처를 기반으로 하며 입력 채널 수와 출력 채널 수를 수정하여 **RGB+Seg → IR** 변환에 특화된 구조로 개선되었습니다.

## Motivation
객체 탐지 모델(YOLOv8)을 RGB+IR 4채널 입력 구조로 확장하기 위해 IR(열화상) 이미지 데이터를 별도로 확보해야 하는 과제가 있었고 이를 해결하고자 RGB 이미지와 해당 Segmentation 정보를 활용해 대응되는 IR 이미지를 생성하는 CycleGAN 기반의 변환 모델을 개발하게 되었습니다.

또한, 실제 열화상 촬영 없이도 다양한 IR 이미지 데이터를 생성할 수 있어, 국방 분야의 위장 객체 탐지 성능 향상 및 이미지 데이터 증강에 기여할 수 있는 가능성을 열어주었습니다.

## Project Environment
- OS: Ubuntu 25.04
- Python: 3.10
- Pytorch:2.2.0
- CUDA:12.1
- GPU: NVIDIA TITAN Xp / A100 (Colab)

## Colab Notebook
Thermal-CycleGAN Tutorial: [Google Colab](https://colab.research.google.com/drive/151BsdW-YFtof58BDCZL86fjbLWB4eYJ_?usp=drive_link) | [Code](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb)

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/scksh/Thermal-CycleGAN
cd Thermal-CycleGAN
```
- For pip users, please type the command `pip install -r requirements.txt`.

### Download Dataset (HuggingFace)
- You can download the `RGBSeg2IR` dataset directly from Hugging Face Hub using the following Python code:
```python
from huggingface_hub import hf_hub_download
import zipfile

# Download dataset
zip_path = hf_hub_download(
    repo_id="SUMMERZETT/RGBSeg2IR",
    filename="RGBSeg2IR.zip",
    repo_type="dataset"
)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("./datasets")
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with train and test script

### Download pre-trained model (Thermal-CycleGAN)
- To download the pretrained Thermal-CycleGAN model from Hugging Face, use the following code:
```python
from huggingface_hub import hf_hub_download

# Download pretrained model
model_path = hf_hub_download(
    repo_id="SUMMERZETT/Thermal-CycleGAN",
    filename="thermal_cyclegan.pth",
    repo_type="model",
    local_dir="./pretrained",
    local_dir_use_symlinks=False
)
```
### Thermal-CycleGAN Train
- Train a model:
```bash
!python train.py --dataroot ./datasets/RGBSeg2IR --name RGBSeg2IR --model cycle_gan --direction AtoB --dataset_mode aligned --input_nc 4 --output_nc 1 --gpu_ids 0 --n_epochs 0 --n_epochs_decay 10 --lambda_identity 0 --lr_policy linear
```

### Apply a pre-trained mode (Thermal-CycleGAN)
- Generate the results:
```bash
!python generate.py --rgb_dir ./datasets/RGBSeg2IR/testA --seg_dir ./datasets/RGBSeg2IR/testSegA --model_path ./pretrained/thermal_cyclegan.pth --output_dir ./results/generated_IR --gpu_ids 0
```

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```

## Other Languages
[Spanish](docs/README_es.md)

## Related Projects
**[contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) (CUT)**<br>
**[CycleGAN-Torch](https://github.com/junyanz/CycleGAN) |
[pix2pix-Torch](https://github.com/phillipi/pix2pix) | [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)|
[BicycleGAN](https://github.com/junyanz/BicycleGAN) | [vid2vid](https://tcwang0509.github.io/vid2vid/) | [SPADE/GauGAN](https://github.com/NVlabs/SPADE)**<br>
**[iGAN](https://github.com/junyanz/iGAN) | [GAN Dissection](https://github.com/CSAILVision/GANDissect) | [GAN Paint](http://ganpaint.io/)**

## Cat Paper Collection
If you love cats, and love reading cool graphics, vision, and learning papers, please check out the Cat Paper [Collection](https://github.com/junyanz/CatPapers).

## Acknowledgments
Our code is inspired by [pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan).
