import os
import shutil

# Define root and destination directory
root_dir = '/home/bkcs/HDD/Dao_Cuong/ICS-Detection'
target_dir = os.path.join(root_dir, 'all_pdfs_collected')
os.makedirs(target_dir, exist_ok=True)

# Helper to clean up the name
def build_new_filename(rel_path_parts, filename):
    parts = [p for p in rel_path_parts if p]  # remove empty strings
    name_prefix = '_'.join(parts)
    return f"{name_prefix}_{filename}" if name_prefix else filename

# Walk and collect PDFs
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.lower().endswith('.pdf'):
            source_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(dirpath, root_dir)
            rel_parts = rel_path.split(os.sep)

            # Avoid copying files already in the target dir
            if os.path.commonpath([dirpath, target_dir]) == target_dir:
                continue

            # Build a clean name
            new_name = build_new_filename(rel_parts, filename)
            dest_path = os.path.join(target_dir, new_name)

            # Prevent overwrite
            base, ext = os.path.splitext(new_name)
            counter = 1
            while os.path.exists(dest_path):
                new_name = f"{base}_{counter}{ext}"
                dest_path = os.path.join(target_dir, new_name)
                counter += 1

            shutil.copy2(source_path, dest_path)
            print(f"Copied: {source_path} → {dest_path}")
