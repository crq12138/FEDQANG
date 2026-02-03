import numpy as np
import os

def shrink_dataset(file_path, target_size=12000):
    """
    读取npy文件，随机抽取指定数量的样本，并保存为新文件。
    支持处理：
    1. 简单的 numpy 数组 (N, ...)
    2. 包含 'data'/'images' 和 'label'/'labels' 的字典
    """
    
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件 {file_path}")
        return

    print(f"🔄 正在加载 {file_path} ...")
    # allow_pickle=True 是必须的，因为如果里面存的是字典或对象，默认是不允许的
    dataset = np.load(file_path, allow_pickle=True)

    # 2. 分析数据结构并获取原始样本量 N
    data_type = type(dataset)
    N = 0
    
    # 既然是联邦学习的client数据，通常是一个字典或者对象，或者直接是数组
    # 我们先尝试转换成字典处理（如果是0-d array包裹的字典）
    if dataset.ndim == 0 and dataset.dtype == 'O':
        # 解包 0-d array
        dataset = dataset.item()
        data_type = type(dataset)

    # 逻辑分流
    if isinstance(dataset, dict):
        # 假设字典里有 'images' 或 'x' 这种键，取第一个键的长度作为 N
        keys = list(dataset.keys())
        first_key = keys[0]
        N = len(dataset[first_key])
        print(f"ℹ️  检测到字典结构，包含键: {keys}")
    elif isinstance(dataset, (np.ndarray, list)):
        N = len(dataset)
        print(f"ℹ️  检测到数组结构")
    else:
        print(f"❌ 未知的数据结构类型: {data_type}")
        return

    print(f"📊 原始样本数量: {N}")

    # 3. 检查是否需要缩减
    if N <= target_size:
        print(f"⚠️  原始数量 ({N}) 小于或等于目标数量 ({target_size})，无需缩减。")
        return

    # 4. 生成随机索引 (核心步骤)
    # replace=False 表示不放回抽样，保证数据不重复
    # 这种方式可以在大数据量下保持原始的类别分布（等比例）
    print(f"✂️  正在随机抽取 {target_size} 个样本...")
    indices = np.random.choice(N, target_size, replace=False)
    indices.sort() # 排序索引，有时候能提升后续读取效率或保持时序（如果有的话）

    # 5. 根据索引构建新数据
    new_dataset = None

    if isinstance(dataset, dict):
        new_dataset = {}
        for k, v in dataset.items():
            # 确保字典里的每个值都是数组且长度一致，才能切片
            if isinstance(v, (np.ndarray, list)) and len(v) == N:
                new_dataset[k] = np.array(v)[indices]
            else:
                # 如果是 metadata（如 'description': 'dataset info'），则直接复制
                new_dataset[k] = v
    else:
        # 纯数组直接切片
        new_dataset = dataset[indices]

    # 6. 保存新文件
    filename, ext = os.path.splitext(file_path)
    new_file_path = f"{filename}_small{ext}"
    
    np.save(new_file_path, new_dataset)
    
    print(f"✅ 处理完成！")
    print(f"💾 新文件已保存为: {new_file_path}")
    print(f"📉 最终样本数量: {target_size}")

if __name__ == "__main__":
    # 配置
    TARGET_FILE = './medmnist/pathmnist_exp1_client_5.npy'
    TARGET_SIZE = 12000
    
    shrink_dataset(TARGET_FILE, TARGET_SIZE)