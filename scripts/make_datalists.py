"""Generate the datalist txt files consumed by the dataset loaders.

Each output file lists absolute image paths, one per line, sorted by
filename. The three test lists must stay aligned line-by-line
(i-th image <-> i-th mask <-> i-th edited image), which sorting guarantees
as long as corresponding files share the same name across the three
directories.

Usage:
    # Training list (e.g. COCO train2017)
    python scripts/make_datalists.py --train_dir /path/to/coco/train2017

    # Test lists
    python scripts/make_datalists.py \
        --test_img_dir /path/to/test/original \
        --test_mask_dir /path/to/test/mask \
        --test_edited_dir /path/to/test/edited
"""

import argparse
from pathlib import Path

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}


def list_images(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f'Not a directory: {directory}')
    paths = sorted(
        p for p in directory.rglob('*')
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if not paths:
        raise FileNotFoundError(f'No images found under: {directory}')
    return paths


def write_list(paths, out_file):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        f.write('\n'.join(str(p) for p in paths) + '\n')
    print(f'{out_file}: {len(paths)} images')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train_dir', help='directory of training images (-> train.txt)')
    parser.add_argument('--test_img_dir', help='directory of original (cover) images (-> test_img.txt)')
    parser.add_argument('--test_mask_dir', help='directory of binary edit-region masks (-> test_mask.txt)')
    parser.add_argument('--test_edited_dir', help='directory of edited images (-> test_edited.txt)')
    parser.add_argument('--out_dir', default='./datalists', help='output directory (default: ./datalists)')
    args = parser.parse_args()

    if not any([args.train_dir, args.test_img_dir, args.test_mask_dir, args.test_edited_dir]):
        parser.error('give --train_dir and/or the three --test_*_dir options')

    test_args = [args.test_img_dir, args.test_mask_dir, args.test_edited_dir]
    if any(test_args) and not all(test_args):
        parser.error('--test_img_dir, --test_mask_dir and --test_edited_dir must be given together')

    out_dir = Path(args.out_dir)

    if args.train_dir:
        write_list(list_images(args.train_dir), out_dir / 'train.txt')

    if all(test_args):
        img = list_images(args.test_img_dir)
        mask = list_images(args.test_mask_dir)
        edited = list_images(args.test_edited_dir)
        if not (len(img) == len(mask) == len(edited)):
            raise SystemExit(
                f'Count mismatch: {len(img)} images, {len(mask)} masks, '
                f'{len(edited)} edited images — the three lists must align line-by-line.'
            )
        for a, b, c in zip(img, mask, edited):
            if not (a.stem == b.stem == c.stem):
                print(f'[warn] name mismatch: {a.name} / {b.name} / {c.name}')
        write_list(img, out_dir / 'test_img.txt')
        write_list(mask, out_dir / 'test_mask.txt')
        write_list(edited, out_dir / 'test_edited.txt')


if __name__ == '__main__':
    main()
