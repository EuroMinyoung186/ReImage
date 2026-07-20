"""Generate edited (tampered) test images by inpainting the masked region
with Stable Diffusion or SDXL.

Given the original images and binary masks of the region to edit (white =
edited region), this produces one edited image per input, keeping the same
filename so the three datalists stay aligned.

Usage:
    pip install diffusers transformers accelerate

    # Stable Diffusion inpainting
    python scripts/generate_edited.py --model sd \
        --img_dir /path/to/test_img --mask_dir /path/to/test_mask \
        --out_dir /path/to/sd_edited

    # SDXL inpainting
    python scripts/generate_edited.py --model sdxl \
        --img_dir /path/to/test_img --mask_dir /path/to/test_mask \
        --out_dir /path/to/sdxl_edited
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

MODEL_IDS = {
    'sd': 'stable-diffusion-v1-5/stable-diffusion-inpainting',
    'sdxl': 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
}

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def build_pipeline(model, device):
    if model == 'sd':
        from diffusers import StableDiffusionInpaintPipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL_IDS['sd'], torch_dtype=torch.float16, safety_checker=None)
    else:
        from diffusers import StableDiffusionXLInpaintPipeline
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            MODEL_IDS['sdxl'], torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=['sd', 'sdxl'], required=True)
    parser.add_argument('--img_dir', required=True, help='original (cover) images')
    parser.add_argument('--mask_dir', required=True, help='binary masks (white = region to edit)')
    parser.add_argument('--out_dir', required=True, help='where to save the edited images')
    parser.add_argument('--prompt', default='', help='inpainting prompt (default: empty)')
    parser.add_argument('--size', type=int, default=512, help='resolution fed to the pipeline')
    parser.add_argument('--steps', type=int, default=50, help='denoising steps')
    parser.add_argument('--seed', type=int, default=20)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = build_pipeline(args.model, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    img_dir, mask_dir = Path(args.img_dir), Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f'No images found in {img_dir}')

    for i, img_path in enumerate(images, 1):
        matches = [mask_dir / (img_path.stem + ext) for ext in IMG_EXTS]
        mask_path = next((m for m in matches if m.exists()), None)
        if mask_path is None:
            print(f'[skip] no mask for {img_path.name}')
            continue

        out_path = out_dir / (img_path.stem + '.png')
        if out_path.exists():
            continue

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        orig_size = image.size

        result = pipe(
            prompt=args.prompt,
            image=image.resize((args.size, args.size)),
            mask_image=mask.resize((args.size, args.size)),
            num_inference_steps=args.steps,
            generator=generator,
        ).images[0]

        result = result.resize(orig_size)
        # Composite so only the masked region is actually modified
        result = Image.composite(result, image, mask.point(lambda v: 255 if v > 127 else 0))
        result.save(out_path)

        if i % 50 == 0 or i == len(images):
            print(f'{i}/{len(images)} done')


if __name__ == '__main__':
    main()
