import os
import tarfile

def extract_tar_files(tar_dir, output_dir):
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all .tar files in the specified directory
    for tar_file in os.listdir(tar_dir):
        if tar_file.endswith('.tar'):
            # Get category directory
            class_name = tar_file[:-4]  # class name, i.e. strip '.tar'
            class_dir = os.path.join(output_dir, class_name)

            # Create class directory
            os.makedirs(class_dir, exist_ok=True)

            # Extract file to the corresponding class directory
            try:
                with tarfile.open(os.path.join(tar_dir, tar_file), 'r') as tar_ref:
                    tar_ref.extractall(class_dir)
                print(f"Successfully extracted {tar_file} to {class_name} directory.")
            except tarfile.TarError as e:
                print(f"tarfile error while extracting {tar_file}: {e}")
            except Exception as e:
                print(f"Error while extracting {tar_file}: {e}")

# Set your input and output directories
tar_directory = r'C:\baidunetdiskdownload\train_tar'  # replace with your .tar file directory
output_directory = 'train'   # replace with the directory where you want to store the extracted images

# Call the function to extract
extract_tar_files(tar_directory, output_directory)