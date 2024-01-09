import clip
import numpy as np
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

# Load test image
test_image_path = "./my_dataset/test_image.png"
test_image = transform(Image.open(test_image_path)).unsqueeze(0).to(device)

# Different prompts
prompts = [
    "a photo of a cat",
    "cat",
    "nice cat",
    "a photo of a black cat",
    "a photo of a cat sitting on a chair",
]

# Probability calculation for each prompt
for prompt in prompts:
    text = clip.tokenize([prompt]).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(test_image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print(f"Prompt: {prompt}")
    print("Label probabilities:", probs)
    predicted_label = np.argmax(probs)

    # Get the class labels from the tokenizer
    class_labels = [label.replace("a photo of a ", "") for label in prompt.split(", ")]

    print("Predicted label:", predicted_label, f"({class_labels[predicted_label]})")
    print()
   