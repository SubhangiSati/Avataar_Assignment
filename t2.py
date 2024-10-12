import logging
import argparse
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Pose Editing for Object in Image")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--mask', type=str, required=True, help='Path to the mask image')
    parser.add_argument('--object_class', type=str, required=True, help='Object class to modify (e.g., chair)')
    parser.add_argument('--azimuth', type=float, required=True, help='Azimuth angle for pose adjustment')
    parser.add_argument('--polar', type=float, required=True, help='Polar angle for pose adjustment')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output image')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the inpainting model
    logger.info("Loading Stable Diffusion Inpainting model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting").to(device)

    # Load the input image and mask
    logger.info(f"Loading image from {args.image}")
    image = Image.open(args.image).convert("RGB")
    
    logger.info(f"Loading mask from {args.mask}")
    mask = Image.open(args.mask).convert("L")  # Grayscale mask

    # Create a prompt that is specific to preserving the object while adjusting its pose
    prompt = (f"A {args.object_class} rotated by azimuth {args.azimuth} degrees and polar {args.polar} degrees, "
              f"while preserving its original look and blending into the existing scene.")

    logger.info("Processing image for pose editing")

    # Use the inpainting pipeline with the prompt and mask to adjust only the masked region (the object)
    edited_image = pipe(prompt=prompt, image=image, mask_image=mask).images[0]

    # Post-process: Blend the inpainted result into the original image where the mask is applied
    original_np = np.array(image)
    edited_np = np.array(edited_image)
    mask_np = np.array(mask) / 255.0  # Convert mask to range 0-1

    # Combine the original image and the edited image based on the mask (keeping unmasked areas unchanged)
    final_image_np = (mask_np[..., None] * edited_np + (1 - mask_np[..., None]) * original_np).astype(np.uint8)
    
    # Save the final output image
    final_image = Image.fromarray(final_image_np)
    final_image.save(args.output)
    
    logger.info(f"Edited image saved at {args.output}")

if __name__ == "_main_":
    main()