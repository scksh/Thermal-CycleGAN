import torch
from PIL import Image
import torchvision.transforms as transforms
from models import networks
import os
import sys
sys.path.append('/home/ricky/Data/Test_Images')  # 예시

# 모델 정의 (입력 4채널, 출력 3채널 or 1채널 등 상황에 따라 조정)
netG = networks.define_G(input_nc=4, output_nc=1, ngf=64, netG='resnet_9blocks',
                         norm='instance', use_dropout=False, init_type='normal',
                         init_gain=0.02, gpu_ids=[])
# netG = networks.define_G(input_nc=4, output_nc=1, ngf=64, netG='resnet_9blocks',
#                          norm='instance', use_dropout=False, init_type='normal',
#                          init_gain=0.02, gpu_ids=[0])

# 저장된 모델 로드 
model_path = '/home/ricky/Data/checkpoints/case1/20_net_G_A.pth'
netG.load_state_dict(torch.load(model_path, map_location='cuda:0'))
# netG.load_state_dict(torch.load(model_path, map_location='cuda:0'))
netG.eval()

# 전처리 정의 (RGB와 Seg에 동일한 변환)
transform_rgb = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_seg = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Grayscale 1채널용 정규화
])

# 테스트 이미지 파일 이름
filename = '26.jpg'

# 경로 설정
rgb_path = os.path.join('Test_Images', 'RGB', filename)
seg_path = os.path.join('Test_Images', 'SEG', filename)
output_path = os.path.join('Test_Images', 'Generated', filename)

# 디렉토리 없으면 생성
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 이미지 로드 및 전처리
rgb_img = Image.open(rgb_path).convert('RGB')
seg_img = Image.open(seg_path).convert('L')  # Grayscale mask

rgb_tensor = transform_rgb(rgb_img)           # [3, 256, 256]
seg_tensor = transform_seg(seg_img)           # [1, 256, 256]

# RGB + Seg concat → [4, 256, 256]
input_tensor = torch.cat([rgb_tensor, seg_tensor], dim=0).unsqueeze(0)  # [1, 4, 256, 256]

# 추론
with torch.no_grad():
    output_tensor = netG(input_tensor)

# 후처리 및 저장
output_image = output_tensor.squeeze().detach().cpu()
output_image = (output_image + 1) / 2.0  # [-1,1] → [0,1]
output_image = transforms.ToPILImage()(output_image)
output_image.save(output_path)

print(f"변환된 thermal-like 이미지 저장 완료: {output_path}")