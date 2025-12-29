import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ------------------------------------------------------------
# Font discovery
# ------------------------------------------------------------

FONT_DIRS = [
    ".",  # local node dir
    "/usr/share/fonts",
    "/usr/share/fonts/truetype",
    "C:/Windows/Fonts",
]

def find_fonts():
    fonts = []
    for base in FONT_DIRS:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith((".ttf", ".otf")):
                    fonts.append(os.path.join(root, f))
    return sorted(set(fonts))


font_list = find_fonts()


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def hex_to_rgba(hex_color):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (r, g, b, 255)
    elif len(hex_color) == 8:
        r, g, b, a = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4, 6))
        return (r, g, b, a)
    else:
        raise ValueError(f"Invalid hex color: {hex_color}")


def compute_position(position, img_w, img_h, text_w, text_h, padding):
    if position == "top-left":
        return (padding, padding)
    if position == "top-right":
        return (img_w - text_w - padding, padding)
    if position == "bottom-left":
        return (padding, img_h - text_h - padding)
    if position == "bottom-right":
        return (img_w - text_w - padding, img_h - text_h - padding)
    if position == "center":
        return (
            (img_w - text_w) // 2,
            (img_h - text_h) // 2,
        )
    return (padding, padding)


# ------------------------------------------------------------
# Node
# ------------------------------------------------------------

class OverlayFrameNumber:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_number": ("INT", {"default": 1, "min": 1, "max": 99999}),
                "position": (
                    ["top-left", "top-right", "bottom-left", "bottom-right", "center"],
                    {"default": "bottom-right"},
                ),
                "font_size": ("INT", {"default": 48, "min": 12, "max": 200}),
                "font_color": ("STRING", {"default": "#FFFFFF"}),
                "background_color": ("STRING", {"default": "#00000000"}),
                "outline_color": ("STRING", {"default": "#000000"}),
                "outline_width": ("INT", {"default": 2, "min": 0, "max": 10}),
                "font_file": (["None"] + sorted(font_list), {"default": "None"}),
                "text_padding": ("INT", {"default": 10, "min": 0, "max": 50}),
                "num_padding": ("INT", {"default": 4, "min": 1, "max": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/postprocessing"

    def process(
        self,
        images,
        frame_number,
        position,
        font_size,
        font_color,
        background_color,
        outline_color,
        outline_width,
        font_file,
        text_padding,
        num_padding,
    ):

        # ----------------------------------------------------
        # Normalize IMAGE input (batch tensor OR list)
        # ----------------------------------------------------

        input_was_tensor = False
        image_list = []

        if isinstance(images, torch.Tensor) and images.ndim == 4:
            input_was_tensor = True
            for i in range(images.shape[0]):
                image_list.append(images[i])
        elif isinstance(images, (list, tuple)):
            image_list = list(images)
        else:
            raise ValueError("Unsupported IMAGE input type")

        # ----------------------------------------------------
        # Font
        # ----------------------------------------------------

        if font_file != "None" and os.path.isfile(font_file):
            font = ImageFont.truetype(font_file, font_size)
        else:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

        font_rgba = hex_to_rgba(font_color)
        bg_rgba = hex_to_rgba(background_color)
        outline_rgba = hex_to_rgba(outline_color)

        output_images = []

        # ----------------------------------------------------
        # Process frames
        # ----------------------------------------------------

        for i, img_tensor in enumerate(image_list):
            img_np = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np).convert("RGBA")

            draw = ImageDraw.Draw(pil_img)

            label = f"{frame_number + i:0{num_padding}d}"

            text_bbox = draw.textbbox((0, 0), label, font=font, stroke_width=outline_width)
            text_w = text_bbox[2] - text_bbox[0] + text_padding * 2
            text_h = text_bbox[3] - text_bbox[1] + text_padding * 2

            x, y = compute_position(
                position,
                pil_img.width,
                pil_img.height,
                text_w,
                text_h,
                text_padding,
            )

            # Background box
            if bg_rgba[3] > 0:
                draw.rectangle(
                    [x, y, x + text_w, y + text_h],
                    fill=bg_rgba,
                )

            # Text
            draw.text(
                (x + text_padding, y + text_padding),
                label,
                fill=font_rgba,
                font=font,
                stroke_width=outline_width,
                stroke_fill=outline_rgba,
            )

            out_np = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
            out_tensor = torch.from_numpy(out_np)

            output_images.append(out_tensor)

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        if input_was_tensor:
            return (torch.stack(output_images, dim=0),)
        else:
            return (output_images,)


NODE_CLASS_MAPPINGS = {
    "OverlayFrameNumber": OverlayFrameNumber
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OverlayFrameNumber": "Overlay Frame Number"
}
