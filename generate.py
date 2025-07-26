import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from models.networks import define_G  # Import your generator definition

def load_input_image(rgb_path, seg_path, transform):
    """Load and preprocess RGB and segmentation image, then concatenate."""
    rgb = Image.open(rgb_path).convert('RGB')
    seg = Image.open(seg_path).convert('L')  # Grayscale

    rgb_tensor = transform(rgb)
    seg_tensor = transform(seg)

    input_tensor = torch.cat([rgb_tensor, seg_tensor], dim=0)  # Shape: [4, H, W]
    return input_tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_dir', type=str, required=True, help='Directory containing RGB test images')
    parser.add_argument('--seg_dir', type=str, required=True, help='Directory containing segmentation maps')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained generator .pth file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save generated IR images')
    parser.add_argument('--gpu_ids', type=str, default='0', help='Comma-separated GPU IDs, e.g. "0" or "-1" for CPU')
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse GPU IDs
    str_ids = args.gpu_ids.split(',')
    gpu_ids = [int(id) for id in str_ids if int(id) >= 0]
    device = torch.device(f'cuda:{gpu_ids[0]}' if gpu_ids and torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    # Define transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # Converts to [0,1]
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # For RGB
    ])

    # Define generator
    netG = define_G(input_nc=4, output_nc=1, ngf=64, netG='resnet_9blocks',
                    norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
    netG.to(device)

    # Load weights
    state_dict = torch.load(args.model_path, map_location=device)
    netG.load_state_dict(state_dict)
    netG.eval()

    # List of RGB images
    rgb_filenames = sorted(os.listdir(args.rgb_dir))
    
    for filename in rgb_filenames:
        rgb_path = os.path.join(args.rgb_dir, filename)
        seg_path = os.path.join(args.seg_dir, filename)

        if not os.path.exists(seg_path):
            print(f"[WARNING] Segmentation map not found for {filename}, skipping.")
            continue

        input_tensor = load_input_image(rgb_path, seg_path, transform)
        input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dim

        with torch.no_grad():
            output = netG(input_tensor)[0].cpu()

        # Rescale from [-1, 1] to [0, 1]
        output = (output + 1) / 2.0
        output_image = transforms.ToPILImage()(output)

        save_path = os.path.join(args.output_dir, filename)
        output_image.save(save_path)
        print(f"Saved: {save_path}")


if __name__ == '__main__':
    main()
