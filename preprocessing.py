from PIL import Image
import os

data_dir = 'dataset'
output_dir = 'processed_dataset'
image_size = (200, 200)

os.makedirs(output_dir, exist_ok=True)

def resize(data_dir, output_dir, image_size):
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
    
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    try:
                        img = Image.open(os.path.join(subdir_path, filename))
                        if img.mode == 'RGBA':
                            img = img.convert('RGB')
                    
                        img_resized = img.resize(image_size)
                    
                        output_subdir = os.path.join(output_dir, subdir)
                        os.makedirs(output_subdir, exist_ok=True)
                    
                        img_resized.save(os.path.join(output_subdir, filename))
                    except Exception as e:
                        print(f"Error processing image {filename}: {str(e)}")


def check_image_sizes(directory):
    sizes = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                img = Image.open(os.path.join(dirpath, filename))
                if img.size not in sizes:
                    sizes.append(img.size)
    return len(sizes) == 1

directory = r'processed_dataset'
if check_image_sizes(directory):
    print("All images are the same size.")
else:
    print("Not all images are the same size.")
