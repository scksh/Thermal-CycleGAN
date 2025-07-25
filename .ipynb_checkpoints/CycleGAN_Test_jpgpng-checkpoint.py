import torch
from PIL import Image
import torchvision.transforms as transforms
from models import networks
import os
import sys
sys.path.append('/home/ricky/Data')  # 예시

# 모델 정의
netG = networks.define_G(input_nc=4, output_nc=1, ngf=64, netG='resnet_9blocks',
                         norm='instance', use_dropout=False, init_type='normal',
                         init_gain=0.02, gpu_ids=[])

# 모델 로드
model_path = r'/home/ricky/Data/checkpoints/person/latest_net_G_A.pth'
netG.load_state_dict(torch.load(model_path, map_location='cuda:0'))
netG.eval()

# 정규화 transform만 정의
normalize_rgb = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
normalize_seg = transforms.Normalize((0.5,), (0.5,))

# 경로 설정
base_dir = '/home/ricky/Data/datasets/person0/valid'
rgb_dir = os.path.join(base_dir, 'images_a')
seg_dir = os.path.join(base_dir, 'images_seg_a')
output_dir = os.path.join(base_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# 파일 리스트 획득
file_list = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.jpg', '.png'))])

for filename in file_list:
    rgb_path = os.path.join(rgb_dir, filename)
    
    # [변경] GT 파일명을 .png로 강제 매핑
    seg_filename = os.path.splitext(filename)[0] + '.png'
    seg_path = os.path.join(seg_dir, seg_filename)

    output_path = os.path.join(output_dir, filename)

    if not os.path.exists(seg_path):
        print(f"[경고] 대응하는 Seg 파일 없음: {seg_path}")
        continue

    rgb_img = Image.open(rgb_path).convert('RGB')
    seg_img = Image.open(seg_path).convert('L').resize(rgb_img.size, resample=Image.NEAREST)

    rgb_tensor = transforms.ToTensor()(rgb_img)
    seg_tensor = transforms.ToTensor()(seg_img)

    rgb_tensor = normalize_rgb(rgb_tensor)
    seg_tensor = normalize_seg(seg_tensor)

    input_tensor = torch.cat([rgb_tensor, seg_tensor], dim=0).unsqueeze(0)

    with torch.no_grad():
        output_tensor = netG(input_tensor)

    output_image = output_tensor.squeeze().cpu()
    output_image = (output_image + 1) / 2.0
    output_image = transforms.ToPILImage()(output_image)
    output_image.save(output_path)

    print(f"→ 저장 완료: {output_path}")

print("모든 이미지 변환 완료.")
