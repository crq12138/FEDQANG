import torch
from copy import deepcopy
from mnist_cnn_model import MNISTCNNModel

def average_models(models):
    """
    对多个节点模型的参数进行平均，生成全局模型。
    
    :param models: List[torch.nn.Module]，包含多个节点的模型实例。
    :return: torch.nn.Module，全局平均模型。
    """
    # 确保至少有一个模型
    if not models:
        raise ValueError("模型列表为空，无法计算平均值。")
    
    # 深拷贝第一个模型作为全局模型的初始结构
    global_model = deepcopy(models[0])
    
    # 初始化全局模型的参数为0
    for param in global_model.parameters():
        param.data.zero_()
    
    # 遍历所有模型，将它们的参数相加
    for model in models:
        for global_param, local_param in zip(global_model.parameters(), model.parameters()):
            global_param.data.add_(local_param.data)
    
    # 计算平均值
    num_models = len(models)
    for param in global_model.parameters():
        param.data.div_(num_models)
    
    return global_model

# 示例使用
if __name__ == "__main__":
    # 假设有3个节点，每个节点都初始化了自己的模型
    node_models = [MNISTCNNModel() for _ in range(3)]
    
    # 为每个节点的模型随机初始化参数
    for model in node_models:
        for param in model.parameters():
            param.data.uniform_(-1, 1)  # 参数随机值 [-1, 1]
    
    # 计算全局模型
    global_model = average_models(node_models)
    
    # 打印全局模型的第一个参数以验证
    print("全局模型的第一个参数:", next(global_model.parameters()))