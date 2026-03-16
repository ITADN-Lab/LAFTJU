import os
import tarfile

def extract_tar_files(tar_dir, output_dir):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历指定目录中的所有 .tar 文件
    for tar_file in os.listdir(tar_dir):
        if tar_file.endswith('.tar'):
            # 获取类别目录
            class_name = tar_file[:-4]  # 类别名，即去掉 '.tar'
            class_dir = os.path.join(output_dir, class_name)

            # 创建类别目录
            os.makedirs(class_dir, exist_ok=True)

            # 解压缩文件到对应类别目录
            try:
                with tarfile.open(os.path.join(tar_dir, tar_file), 'r') as tar_ref:
                    tar_ref.extractall(class_dir)
                print(f"解压缩 {tar_file} 到 {class_name} 目录成功。")
            except tarfile.TarError as e:
                print(f"解压 {tar_file} 时发生 tarfile 错误: {e}")
            except Exception as e:
                print(f"解压 {tar_file} 时发生错误: {e}")

# 设置你的输入和输出目录
tar_directory = r'C:\baidunetdiskdownload\train_tar'  # 替换为你的 .tar 文件目录
output_directory = 'train'   # 替换为你想要存放解压图像的目录

# 调用函数进行解压
extract_tar_files(tar_directory, output_directory)