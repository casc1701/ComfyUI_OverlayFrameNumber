import torch
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import glob
import platform

class OverlayFrameNumber:
    @classmethod
    def INPUT_TYPES(cls):
        # Dynamically get available system fonts
        font_list = cls._get_system_fonts()
        if not font_list:
            font_list = ["Arial", "DejaVuSans", "Times New Roman", "Courier New"]  # Fallback

        return {
            "required": {
                "images": ("IMAGE",),
                "font_size": ("INT", {"default": 32, "min": 8, "max": 200}),
                "font_color": (["white", "black", "red", "green", "blue", "yellow", "cyan", "magenta"],),
                "font": (font_list,),
                "h_position": (["left", "center", "right"], {"default": "left"}),
                "v_position": (["top", "center", "bottom"], {"default": "top"}),
                "h_padding": ("INT", {"default": 20, "min": 0, "max": 1000}),
                "v_padding": ("INT", {"default": 20, "min": 0, "max": 1000}),
                "num_padding": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
                "prefix_text": ("STRING", {"default": "Frame", "multiline": False}),
                "outline_enabled": ("BOOLEAN", {"default": False}),
                "outline_color": (["none", "white", "black", "red", "green", "blue", "yellow", "cyan", "magenta"], {"default": "black"}),
                "stroke_width": ("INT", {"default": 1, "min": 0, "max": 5, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "Image/Processing"
    OUTPUT_IS_LIST = (False,)

    @staticmethod
    def _get_system_fonts():
        """Dynamically retrieve available TrueType fonts from the system."""
        font_list = []
        font_paths = []

        # Common font directories based on OS
        if platform.system() == "Windows":
            font_dirs = [os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")]
        elif platform.system() == "Darwin":  # macOS
            font_dirs = ["/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts")]
        else:  # Linux/Unix-like
            font_dirs = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts")]

        # Add fontconfig fonts if available (Linux)
        try:
            import subprocess
            result = subprocess.run(["fc-list", ":file"], capture_output=True, text=True)
            if result.returncode == 0:
                font_paths.extend([line.split(":")[0] for line in result.stdout.splitlines()])
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Add fonts from common directories
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                font_paths.extend(glob.glob(os.path.join(font_dir, "*.ttf")))

        # Extract font names (without path or extension)
        font_list = sorted(set(os.path.splitext(os.path.basename(f))[0] for f in font_paths))

        # Fallback to common fonts if none found
        if not font_list:
            font_list = ["Arial", "DejaVuSans", "Times New Roman", "Courier New"]

        return font_list

    def process(self, images, font_size, font_color, font, h_position, v_position, h_padding, v_padding, num_padding, prefix_text, outline_enabled, outline_color, stroke_width):
        # Validate inputs
        if not isinstance(images, torch.Tensor) or len(images.shape) != 4:
            raise ValueError("Input 'images' must be a 4D tensor (batch, height, width, channels)")

        # Map font_color and outline_color to RGB
        color_map = {
            "white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0),
            "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
            "cyan": (0, 255, 255), "magenta": (255, 0, 255), "none": None
        }
        font_color_rgb = color_map[font_color]
        outline_color_rgb = color_map[outline_color] if outline_enabled and outline_color != "none" else None

        # Map font names to system font files (with common paths or aliases)
        font_map = {
            "Arial": "arial.ttf",
            "DejaVuSans": "DejaVuSans.ttf",
            "Times New Roman": "times.ttf",
            "Courier New": "cour.ttf"
        }
        font_file = font_map.get(font, font + ".ttf")  # Try font name directly as fallback

        # Try to load the font, with fallback to default
        try:
            font_path = ImageFont.truetype(font_file, font_size)
        except (OSError, IOError):
            # Try to find the font in system directories
            font_found = False
            font_dirs = (
                [os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")] if platform.system() == "Windows" else
                ["/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts")] if platform.system() == "Darwin" else
                ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts")]
            )
            for font_dir in font_dirs:
                font_path_candidate = os.path.join(font_dir, font_file)
                if os.path.exists(font_path_candidate):
                    try:
                        font_path = ImageFont.truetype(font_path_candidate, font_size)
                        font_found = True
                        break
                    except (OSError, IOError):
                        continue
            if not font_found:
                print(f"Font {font_file} not found, using default font")
                font_path = ImageFont.load_default()

        # Process each frame
        output_images = []
        for i in range(images.shape[0]):
            # Convert tensor to PIL image
            img_tensor = images[i]  # Shape: (height, width, channels)
            img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)  # Convert to uint8
            if img_array.shape[2] == 3:  # RGB
                img = Image.fromarray(img_array, mode="RGB")
            else:
                raise ValueError("Input images must have 3 channels (RGB)")

            # Calculate text position
            draw = ImageDraw.Draw(img)
            # Format text: prefix + zero-padded number, or just number if prefix is empty
            frame_number = f"{i + 1:0{num_padding}d}"
            text = f"{prefix_text} {frame_number}" if prefix_text.strip() else frame_number
            try:
                text_bbox = draw.textbbox((0, 0), text, font=font_path)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:  # Fallback for older PIL versions
                text_width, text_height = draw.textsize(text, font=font_path)

            img_width, img_height = img.size

            # Horizontal position
            if h_position == "left":
                x = h_padding
            elif h_position == "center":
                x = (img_width - text_width) // 2
            else:  # right
                x = img_width - text_width - h_padding

            # Vertical position
            if v_position == "top":
                y = v_padding
            elif v_position == "center":
                y = (img_height - text_height) // 2
            else:  # bottom
                y = img_height - text_height - v_padding

            # Draw text with optional outline
            try:
                # Use stroke_width and stroke_fill if outline is enabled and supported (Pillow 9.0+)
                if outline_enabled and outline_color_rgb is not None and stroke_width > 0:
                    draw.text((x, y), text, font=font_path, fill=font_color_rgb, stroke_width=stroke_width, stroke_fill=outline_color_rgb)
                else:
                    draw.text((x, y), text, font=font_path, fill=font_color_rgb)
            except TypeError:  # Fallback for older Pillow versions without stroke_width
                print("Text outline not supported in this Pillow version; rendering without outline")
                draw.text((x, y), text, font=font_path, fill=font_color_rgb)

            # Convert back to tensor
            img_array = np.array(img).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(img_array))

        # Stack images back into a batch tensor
        output_tensor = torch.stack(output_images, dim=0)

        return (output_tensor,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Re-run if inputs change
        return True

    @classmethod
    def OUTPUT_UI(cls, outputs):
        # No UI output for image batches (handled by downstream nodes like Preview Image)
        return {}

NODE_CLASS_MAPPINGS = {
    "OverlayFrameNumber": OverlayFrameNumber
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OverlayFrameNumber": "Overlay Frame Number"
}