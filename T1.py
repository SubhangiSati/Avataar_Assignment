import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import argparse

# Pre-defined COCO classes used in the model
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife',
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(img_path, threshold=0.5):
    # Load the pre-trained Mask R-CNN model
    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Load and preprocess the image
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        predictions = model(img_tensor)  # Make predictions

    # Get the masks, boxes, labels, and scores
    pred_scores = predictions[0]['scores'].numpy()
    pred_labels = predictions[0]['labels'].numpy()
    pred_boxes = predictions[0]['boxes'].numpy()
    pred_masks = predictions[0]['masks'].squeeze(1).numpy()  # Remove extra dimension for masks

    # Filter the predictions based on the threshold score
    pred_filtered = [(pred_labels[i], pred_boxes[i], pred_masks[i]) for i in range(len(pred_scores)) if pred_scores[i] > threshold]

    return pred_filtered

def mask_image(img_path, object_class, output_path):
    # Load the image
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # Get the object's index from COCO categories
    object_idx = COCO_INSTANCE_CATEGORY_NAMES.index(object_class)

    # Get the prediction from the model
    pred_filtered = get_prediction(img_path)

    # Create a mask for the specified object
    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)  # Mask initialized as zeros

    for label, box, obj_mask in pred_filtered:
        if label == object_idx:
            mask[obj_mask > 0.5] = 255  # Mask the detected object with white (255)

    # Convert the mask to RGB red overlay on the original image
    img_np[mask == 255] = [255, 0, 0]  # Red color for the masked object

    # Save the resulting image
    output_img = Image.fromarray(img_np)
    output_img.save(output_path)
    print(f"Masked image saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment an object in an image.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image")
    parser.add_argument('--clas', type=str, required=True, help="Object class to segment (e.g., chair, dog)")
    parser.add_argument('--output', type=str, required=True, help="Path to save the output image")

    args = parser.parse_args()

    # Perform the object segmentation
    mask_image(args.image, args.clas, args.output)
#python3.11 T1.py --image ./chair.jpg --clas "chair" --output ./generated.png
