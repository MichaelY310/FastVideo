# SPDX-License-Identifier: Apache-2.0
"""CPU-only 2x2 synchronized video contact sheets from the saved MP4 outputs."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    font = ImageFont.load_default()
    for candidate in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, 17)
            break
    for directory in sorted(args.root.glob("prompt*")):
        if not directory.is_dir():
            continue
        settings = [("dense", "Dense RDD"), ("c2f50", "C2F keep 50%"),
                    ("c2f20", "C2F keep 20%"), ("c2f12", "C2F keep 12.5%")]
        readers = [imageio.get_reader(str(directory / f"{key}.mp4")) for key, _ in settings]
        try:
            with imageio.get_writer(str(directory / "comparison.mp4"), fps=16, codec="libx264", quality=8,
                                    macro_block_size=16) as writer:
                for frame_index in range(29):
                    canvas = Image.new("RGB", (832, 512), "#172033")
                    draw = ImageDraw.Draw(canvas)
                    for panel, ((_, label), reader) in enumerate(zip(settings, readers, strict=False)):
                        x, y = (panel % 2)*416, (panel // 2)*256
                        frame = Image.fromarray(reader.get_data(frame_index)).resize((416, 224), Image.Resampling.LANCZOS)
                        canvas.paste(frame, (x, y+32))
                        draw.text((x+9, y+5), label, font=font, fill="white")
                    writer.append_data(np.asarray(canvas))
                    if frame_index == 14:
                        canvas.save(directory / "comparison.jpg")
        finally:
            for reader in readers:
                reader.close()
        print(directory / "comparison.mp4")


if __name__ == "__main__":
    main()
