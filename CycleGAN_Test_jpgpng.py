import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from models import networks

# === 설정 ===
PAD = 30
SAVE_FORMAT = "jpg"  # 또는 "png"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# === 경로 설정 ===
base_dir = "/home/ricky/Data/drone_video/445_exp"
img_dir = os.path.join(base_dir, "images")
label_dir = os.path.join(base_dir, "labels")
mask_dir = os.path.join(base_dir, "masks")  # 원본크기 마스크 위치
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# === 모델 로드 ===
netG = networks.define_G(
    input_nc=4, output_nc=1, ngf=64, netG='resnet_9blocks',
    norm='instance', use_dropout=False, init_type='normal',
    init_gain=0.02, gpu_ids=[]
)
model_path = "/home/ricky/Data/checkpoints/person/latest_net_G_A.pth"
netG.load_state_dict(torch.load(model_path, map_location=device))
netG.to(device)
netG.eval()

normalize_rgb = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
normalize_seg = transforms.Normalize((0.5,), (0.5,))

img_files = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))])

for img_file in tqdm(img_files, desc="열화상 생성 중"):
    name = os.path.splitext(img_file)[0]
    img_path = os.path.join(img_dir, img_file)
    label_path = os.path.join(label_dir, name + ".txt")
    mask_path = os.path.join(mask_dir, name + ".png")  # 마스크는 원본 크기 기준
    output_path = os.path.join(output_dir, f"{name}.{SAVE_FORMAT}")
    alt_ext = "png" if SAVE_FORMAT == "jpg" else "jpg"
    output_path_alt = os.path.join(output_dir, f"{name}.{alt_ext}")

    # 결과가 이미 존재하면 스킵
    if os.path.exists(output_path) or os.path.exists(output_path_alt):
        print(f"⏩ 스킵: {name} (이미 존재)")
        continue

    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print(f"[경고] 이미지 로드 실패: {img_path}")
        continue

    h, w = orig_img.shape[:2]

    # === 객체 유무 판단
    has_object = False
    if os.path.exists(label_path):
        with open(label_path) as f:
            has_object = len(f.read().strip()) > 0

    if not has_object:
        # 객체 없음: 패딩 추가 + 가짜 객체 마스크 생성
        padded_img = cv2.copyMakeBorder(orig_img, 0, 0, PAD, 0, cv2.BORDER_REFLECT)
        mask = np.zeros((h, w + PAD), dtype=np.uint8)
        cv2.rectangle(mask, (5, h // 2 - 15), (25, h // 2 + 15), color=255, thickness=-1)
        crop_output = True
    else:
        # 👉 객체 있음: 원본 이미지 + 원본 크기 마스크 사용
        padded_img = orig_img
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[경고] 마스크 로드 실패 → 스킵: {mask_path}")
            continue
        crop_output = False

    # === 열화상 생성 ===
    try:
        rgb_img = Image.fromarray(cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)).convert('RGB')
        seg_img = Image.fromarray(mask).convert('L')

        rgb_tensor = normalize_rgb(transforms.ToTensor()(rgb_img))
        seg_tensor = normalize_seg(transforms.ToTensor()(seg_img))
        input_tensor = torch.cat([rgb_tensor, seg_tensor], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = netG(input_tensor)

        output_image = output_tensor.squeeze().cpu()
        output_image = (output_image + 1) / 2.0
        output_np = (output_image.numpy() * 255).astype(np.uint8)

        if crop_output:
            output_np = output_np[:, PAD:PAD + w]  # 좌측 PAD 잘라냄
        output_np = np.clip(output_np, 0, 255)

        if SAVE_FORMAT == "jpg":
            cv2.imwrite(output_path, output_np, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(output_path, output_np)

        print(f"저장 완료: {output_path}")

    except Exception as e:
        print(f"오류 발생: {img_file} | {e}")

print("열화상 생성 완료!")

output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# === 모델 로드 ===
netG = networks.define_G(
    input_nc=4, output_nc=1, ngf=64, netG='resnet_9blocks',
    norm='instance', use_dropout=False, init_type='normal',
    init_gain=0.02, gpu_ids=[]
)
model_path = "/home/ricky/Data/checkpoints/person/latest_net_G_A.pth"
netG.load_state_dict(torch.load(model_path, map_location=device))
netG.to(device)
netG.eval()

normalize_rgb = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
normalize_seg = transforms.Normalize((0.5,), (0.5,))

img_files = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))])

for img_file in tqdm(img_files, desc="열화상 생성 중"):
    name = os.path.splitext(img_file)[0]
    img_path = os.path.join(img_dir, img_file)
    label_path = os.path.join(label_dir, name + ".txt")
    mask_path = os.path.join(mask_dir, name + ".png")  # 마스크는 원본 크기 기준
    output_path = os.path.join(output_dir, f"{name}.{SAVE_FORMAT}")
    alt_ext = "png" if SAVE_FORMAT == "jpg" else "jpg"
    output_path_alt = os.path.join(output_dir, f"{name}.{alt_ext}")

    # 결과가 이미 존재하면 스킵
    if os.path.exists(output_path) or os.path.exists(output_path_alt):
        print(f"⏩ 스킵: {name} (이미 존재)")
        continue

    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print(f"[경고] 이미지 로드 실패: {img_path}")
        continue

    h, w = orig_img.shape[:2]

    # === 객체 유무 판단
    has_object = False
    if os.path.exists(label_path):
        with open(label_path) as f:
            has_object = len(f.read().strip()) > 0

    if not has_object:
        # 객체 없음: 패딩 추가 + 가짜 객체 마스크 생성
        padded_img = cv2.copyMakeBorder(orig_img, 0, 0, PAD, 0, cv2.BORDER_REFLECT)
        mask = np.zeros((h, w + PAD), dtype=np.uint8)
        cv2.rectangle(mask, (5, h // 2 - 15), (25, h // 2 + 15), color=255, thickness=-1)
        crop_output = True
    else:
        # 객체 있음: 원본 이미지 + 원본 크기 마스크 사용
        padded_img = orig_img
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[경고] 마스크 로드 실패 → 스킵: {mask_path}")
            continue
        crop_output = False

    # === 열화상 생성 ===
    try:
        rgb_img = Image.fromarray(cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)).convert('RGB')
        seg_img = Image.fromarray(mask).convert('L')

        rgb_tensor = normalize_rgb(transforms.ToTensor()(rgb_img))
        seg_tensor = normalize_seg(transforms.ToTensor()(seg_img))
        input_tensor = torch.cat([rgb_tensor, seg_tensor], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = netG(input_tensor)

        output_image = output_tensor.squeeze().cpu()
        output_image = (output_image + 1) / 2.0
        output_np = (output_image.numpy() * 255).astype(np.uint8)

        if crop_output:
            output_np = output_np[:, PAD:PAD + w]  # 좌측 PAD 잘라냄
        output_np = np.clip(output_np, 0, 255)

        if SAVE_FORMAT == "jpg":
            cv2.imwrite(output_path, output_np, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(output_path, output_np)

        print(f"저장 완료: {output_path}")

    except Exception as e:
        print(f"오류 발생: {img_file} | {e}")

print("열화상 생성 완료!")
