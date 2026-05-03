import argparse
from pathlib import Path

from PIL import Image, ImageDraw

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output", default="results/diffusion/sample_grid.png")
    parser.add_argument("--images-per-folder", type=int, default=4)
    parser.add_argument("--thumb-size", type=int, default=160)
    return parser.parse_args()


def list_images(folder, limit):
    paths = [
        path
        for path in sorted(Path(folder).rglob("*"))
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return paths[:limit]


def main():
    args = parse_args()
    labels = args.labels or [Path(folder).name for folder in args.folders]
    if len(labels) != len(args.folders):
        raise ValueError("--labels must have the same length as --folders.")

    rows = []
    for folder, label in zip(args.folders, labels):
        images = []
        for path in list_images(folder, args.images_per_folder):
            image = Image.open(path).convert("RGB")
            image.thumbnail((args.thumb_size, args.thumb_size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (args.thumb_size, args.thumb_size), "white")
            x = (args.thumb_size - image.width) // 2
            y = (args.thumb_size - image.height) // 2
            canvas.paste(image, (x, y))
            images.append(canvas)
        if not images:
            raise FileNotFoundError(f"No images found under {folder}")
        rows.append((label, images))

    label_width = max(120, max(len(label) for label in labels) * 8 + 20)
    row_height = args.thumb_size + 20
    width = label_width + args.images_per_folder * args.thumb_size
    height = len(rows) * row_height
    grid = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(grid)

    for row_idx, (label, images) in enumerate(rows):
        y = row_idx * row_height
        draw.text((10, y + args.thumb_size // 2 - 8), label, fill="black")
        for col_idx, image in enumerate(images):
            x = label_width + col_idx * args.thumb_size
            grid.paste(image, (x, y))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)
    print(output)


if __name__ == "__main__":
    main()
