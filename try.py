#%%
import os
from pdf2image import convert_from_path

# Input and output directories
input_dir = "images"
output_dir = "image"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all PDF files
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        
        print(f"Converting {filename}...")
        
        # Convert PDF to a list of image objects
        images = convert_from_path(pdf_path)
        
        for i, image in enumerate(images):
            image_filename = f"{base_name}_page_{i+1}.png"
            image_output_path = os.path.join(output_dir, image_filename)
            image.save(image_output_path, "PNG")

print("Conversion complete.")
# %%
