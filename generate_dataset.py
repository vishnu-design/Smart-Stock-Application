"""
SmartStock — Synthetic Dataset Generator
Generates realistic warehouse product images: undamaged vs damaged
Damage types: scratches, cracks, dents (discolouration), crush marks, corner damage
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from pathlib import Path

random.seed(42)
np.random.seed(42)

# ── Output dirs ────────────────────────────────────────────────────────────────
BASE = Path("/home/claude/dataset")
for split in ["train", "val", "test"]:
    for cls in ["undamaged", "damaged"]:
        (BASE / split / cls).mkdir(parents=True, exist_ok=True)

# ── Product colour palettes (simulate cardboard boxes / plastic crates) ────────
PRODUCT_PALETTES = [
    [(210, 180, 140), (195, 160, 120), (225, 200, 160)],   # cardboard brown
    [(180, 200, 220), (160, 185, 210), (200, 215, 230)],   # plastic blue-grey
    [(220, 210, 190), (205, 195, 175), (235, 225, 205)],   # light beige
    [(160, 180, 160), (145, 165, 145), (175, 195, 175)],   # industrial green
    [(230, 220, 200), (215, 205, 185), (245, 235, 215)],   # pale box
]

IMG_SIZE = 224  # MobileNetV2 & YOLOv8 standard input


# ── Base product image ─────────────────────────────────────────────────────────
def make_base_product(palette):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(240, 238, 232))
    draw = ImageDraw.Draw(img)

    # Box body
    bx1, by1 = random.randint(20, 45), random.randint(20, 45)
    bx2, by2 = random.randint(175, 200), random.randint(175, 200)
    base_col = palette[0]
    draw.rectangle([bx1, by1, bx2, by2], fill=base_col,
                   outline=tuple(max(0, c - 40) for c in base_col), width=2)

    # Top face shading
    draw.rectangle([bx1, by1, bx2, by1 + 18],
                   fill=palette[2], outline=tuple(max(0, c - 30) for c in base_col), width=1)

    # Label strip
    lx1, lx2 = bx1 + 10, bx2 - 10
    ly1, ly2 = by1 + 30, by1 + 65
    draw.rectangle([lx1, ly1, lx2, ly2], fill=(245, 245, 245),
                   outline=(180, 180, 180), width=1)

    # Barcode lines on label
    for i in range(8):
        lw = random.choice([1, 2])
        x = lx1 + 8 + i * ((lx2 - lx1 - 16) // 8)
        draw.line([(x, ly1 + 4), (x, ly2 - 8)], fill=(30, 30, 30), width=lw)

    # Seam lines
    mid_x = (bx1 + bx2) // 2
    draw.line([(mid_x, by1 + 18), (mid_x, by2)],
              fill=tuple(max(0, c - 25) for c in base_col), width=1)
    draw.line([(bx1, by1 + 18), (bx2, by1 + 18)],
              fill=tuple(max(0, c - 25) for c in base_col), width=1)

    # Subtle noise
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 3, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    return img, (bx1, by1, bx2, by2)


# ── Damage functions ───────────────────────────────────────────────────────────
def add_scratches(img, box):
    draw = ImageDraw.Draw(img)
    bx1, by1, bx2, by2 = box
    n = random.randint(2, 6)
    for _ in range(n):
        x1 = random.randint(bx1, bx2)
        y1 = random.randint(by1, by2)
        length = random.randint(15, 60)
        angle = random.uniform(0, 360)
        import math
        x2 = int(x1 + length * math.cos(math.radians(angle)))
        y2 = int(y1 + length * math.sin(math.radians(angle)))
        col = tuple(random.randint(60, 100) for _ in range(3))
        draw.line([(x1, y1), (x2, y2)], fill=col, width=random.randint(1, 3))
    return img


def add_cracks(img, box):
    draw = ImageDraw.Draw(img)
    bx1, by1, bx2, by2 = box
    import math
    for _ in range(random.randint(1, 3)):
        sx = random.randint(bx1 + 10, bx2 - 10)
        sy = random.randint(by1 + 10, by2 - 10)
        pts = [(sx, sy)]
        angle = random.uniform(0, 360)
        for _ in range(random.randint(4, 10)):
            step = random.randint(8, 20)
            angle += random.uniform(-45, 45)
            nx = int(pts[-1][0] + step * math.cos(math.radians(angle)))
            ny = int(pts[-1][1] + step * math.sin(math.radians(angle)))
            pts.append((nx, ny))
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=(40, 35, 30), width=2)
            # hairline branch
            if random.random() > 0.5:
                ba = angle + random.choice([-60, 60])
                bx = int(pts[i][0] + 10 * math.cos(math.radians(ba)))
                by_ = int(pts[i][1] + 10 * math.sin(math.radians(ba)))
                draw.line([pts[i], (bx, by_)], fill=(55, 50, 45), width=1)
    return img


def add_dent_discolouration(img, box):
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    bx1, by1, bx2, by2 = box
    cx = random.randint(bx1 + 20, bx2 - 20)
    cy = random.randint(by1 + 20, by2 - 20)
    rx, ry = random.randint(15, 40), random.randint(10, 30)
    col = random.choice([(80, 60, 40, 120), (100, 80, 60, 100),
                          (140, 110, 80, 90), (60, 50, 40, 130)])
    draw.ellipse([(cx - rx, cy - ry), (cx + rx, cy + ry)], fill=col)
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay).convert("RGB")
    return img


def add_crush_deformation(img, box):
    bx1, by1, bx2, by2 = box
    arr = np.array(img)
    # Warp a horizontal band to simulate crush
    band_y = random.randint(by1 + 20, by2 - 30)
    band_h = random.randint(10, 25)
    shift = random.randint(3, 10)
    arr[band_y:band_y + band_h, bx1:bx2] = np.roll(
        arr[band_y:band_y + band_h, bx1:bx2], shift, axis=1)
    # Darken the crushed region
    arr[band_y:band_y + band_h, bx1:bx2] = (
        arr[band_y:band_y + band_h, bx1:bx2] * 0.75).astype(np.uint8)
    return Image.fromarray(arr)


def add_corner_damage(img, box):
    draw = ImageDraw.Draw(img)
    bx1, by1, bx2, by2 = box
    corner = random.choice(["tl", "tr", "bl", "br"])
    sz = random.randint(15, 35)
    bg = (240, 238, 232)  # matches canvas background
    if corner == "tl":
        pts = [(bx1, by1), (bx1 + sz, by1), (bx1, by1 + sz)]
    elif corner == "tr":
        pts = [(bx2, by1), (bx2 - sz, by1), (bx2, by1 + sz)]
    elif corner == "bl":
        pts = [(bx1, by2), (bx1 + sz, by2), (bx1, by2 - sz)]
    else:
        pts = [(bx2, by2), (bx2 - sz, by2), (bx2, by2 - sz)]
    draw.polygon(pts, fill=bg)
    # Ragged torn edge
    for pt in pts:
        draw.ellipse([pt[0] - 3, pt[1] - 3, pt[0] + 3, pt[1] + 3],
                     fill=tuple(max(0, c - 30) for c in bg))
    return img


DAMAGE_FNS = [add_scratches, add_cracks, add_dent_discolouration,
              add_crush_deformation, add_corner_damage]


# ── Apply augmentations (brightness, blur, rotation) ──────────────────────────
def augment(img):
    if random.random() > 0.4:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
    if random.random() > 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    if random.random() > 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    if random.random() > 0.5:
        img = img.rotate(random.uniform(-12, 12), expand=False,
                         fillcolor=(240, 238, 232))
    return img


# ── Image generators ──────────────────────────────────────────────────────────
def make_undamaged():
    palette = random.choice(PRODUCT_PALETTES)
    img, box = make_base_product(palette)
    img = augment(img)
    return img


def make_damaged():
    palette = random.choice(PRODUCT_PALETTES)
    img, box = make_base_product(palette)
    # Apply 1-3 damage types
    chosen = random.sample(DAMAGE_FNS, k=random.randint(1, 3))
    for fn in chosen:
        img = fn(img, box)
    img = augment(img)
    return img


# ── Generate splits ───────────────────────────────────────────────────────────
SPLIT_COUNTS = {
    "train": {"undamaged": 400, "damaged": 400},
    "val":   {"undamaged": 80,  "damaged": 80},
    "test":  {"undamaged": 60,  "damaged": 60},
}

total = 0
for split, counts in SPLIT_COUNTS.items():
    for cls, n in counts.items():
        fn = make_undamaged if cls == "undamaged" else make_damaged
        for i in range(n):
            img = fn()
            out_path = BASE / split / cls / f"{cls}_{i:04d}.jpg"
            img.save(out_path, "JPEG", quality=90)
            total += 1
        print(f"  [{split}/{cls}] {n} images generated")

print(f"\nDataset complete — {total} total images at {BASE}")
print(f"  Train: 800  |  Val: 160  |  Test: 120")
