import os
from pdf2image import convert_from_path

# Path containing the PDFs
pdf_dir = "/home/ics-security/ICS-Detection/outputs/all_pdfs_collected"

# Loop through files in the directory
for filename in os.listdir(pdf_dir):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, filename)
        
        # Convert PDF to images (one per page)
        try:
            images = convert_from_path(pdf_path, dpi=300)  # Higher DPI = better quality
            base_name = os.path.splitext(filename)[0]
            
            for i, image in enumerate(images):
                output_path = os.path.join(pdf_dir, f"{base_name}_page_{i+1}.png")
                image.save(output_path, "PNG")
                print(f"Saved: {output_path}")
        
        except Exception as e:
            print(f"Error converting {filename}: {e}")

print("Conversion completed.")
