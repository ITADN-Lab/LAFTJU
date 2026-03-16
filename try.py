import os
num_cpus = os.cpu_count()  # 获取总的 CPU 核心数
print(f"可用的 CPU 核心数: {num_cpus}")