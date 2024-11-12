import os
import shutil

TRAIN_DIR = 'Training'

def create_data_directory(input_dir, output_dir, required_type='color'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        if required_type in root:
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    
                    relative_path = os.path.relpath(root, input_dir).replace(os.sep, "_")
                    output_img_name = f"{relative_path}_{file}"
                    output_img_path = os.path.join(output_dir, output_img_name)
                    
                    if os.path.exists(output_img_path):
                        os.remove(output_img_path)
                    
                    shutil.copy(img_path, output_img_path)

create_data_directory(os.path.join(TRAIN_DIR, 'real_part'), 'Training_Color/real_part', required_type='color')
create_data_directory(os.path.join(TRAIN_DIR, 'fake_part'), 'Training_Color/fake_part', required_type='color')
