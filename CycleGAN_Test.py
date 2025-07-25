import torch
from PIL import Image
import torchvision.transforms as transforms
from models import networks
import os
import sys

# 경로 설정
base_dir = r"/home/ricky"
rgb_dir = os.path.join(base_dir, "images")
seg_dir = os.path.join(base_dir, "output_masks")
out_dir = os.path.join(base_dir, "thermal")
model_path = os.path.join(base_dir, "latest_net_G_A.pth")

# 디렉토리 생성
os.makedirs(out_dir, exist_ok=True)

# 모델 정의 및 불러오기 (입력 4채널, 출력 1채널)
netG = networks.define_G(
    input_nc=4, output_nc=1, ngf=64, netG='resnet_9blocks',
    norm='instance', use_dropout=False, init_type='normal',
    init_gain=0.02, gpu_ids=[]
)
netG.load_state_dict(torch.load(model_path, map_location='cpu'))  # GPU 사용시 map_location='cuda:0'
netG.eval()

# 전처리 정의
transform_rgb = transforms.Compose([
    transforms.Resize((1280, 720)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_seg = transforms.Compose([
    transforms.Resize((1280, 720)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 🔁 모든 이미지 처리
for filename in os.listdir(rgb_dir):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    rgb_path = os.path.join(rgb_dir, filename)
    seg_path = os.path.join(seg_dir, filename.replace(".jpg", "_combined_mask.png"))  # 마스크는 _mask 붙음
    output_path = os.path.join(out_dir, filename.replace(".jpg", "_thermal.png"))

    if not os.path.exists(seg_path):
        print(f"❌ 마스크 없음: {seg_path}")
        continue

    try:
        rgb_img = Image.open(rgb_path).convert('RGB')
        seg_img = Image.open(seg_path).convert('L')
    except Exception as e:
        print(f"⚠️ 파일 처리 오류: {filename}, 에러: {e}")
        continue

    # 전처리
    rgb_tensor = transform_rgb(rgb_img)
    seg_tensor = transform_seg(seg_img)

    # [1, 4, 256, 256]
    input_tensor = torch.cat([rgb_tensor, seg_tensor], dim=0).unsqueeze(0)

    # 추론
    with torch.no_grad():
        output_tensor = netG(input_tensor)

    # 후처리 및 저장
    output_image = output_tensor.squeeze().detach().cpu()
    output_image = (output_image + 1) / 2.0  # [-1, 1] → [0, 1]
    output_image = transforms.ToPILImage()(output_image)
    output_image.save(output_path)

    print(f"✅ 저장 완료: {output_path}")

print("🎉 모든 이미지 변환 완료.")

