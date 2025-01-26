import os
import json

# Path to your JSON file
json_path = "data.json"

# Directory where your images (and captions) will be stored for training
image_dir = "./data"
os.makedirs(image_dir, exist_ok=True)

# Your chosen generic prompt
prompt = "thread error in the textile cloth"

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for entry in data["image_data"]:
    target_image_path = entry["target_image"].lstrip('/')
    # Extract the image file name (e.g., "image_1.jpg")
    image_name = os.path.basename(target_image_path)
    
    # Create a .txt filename with the same base name
    base_name, ext = os.path.splitext(image_name)
    caption_file_name = base_name + ".txt"

    caption_file_path = os.path.join(image_dir, caption_file_name)
    with open(caption_file_path, 'w', encoding='utf-8') as cfile:
        cfile.write(prompt)

print("Caption files generated successfully.")
