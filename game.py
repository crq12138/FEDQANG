import numpy as np
import math
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import os

class Node:
    def __init__(self, index, x, q, p_n, max_data_size):
        """
        初始化节点。

        :param index: 节点索引
        :param x: 初始数据集大小
        :param q: 质量分数
        :param p_n: p_n 值
        :param max_data_size: 最大数据集大小
        """
        self.index = index
        self.x = x
        self.q = q
        self.p_n = p_n
        self.max_data_size = max_data_size

    def capacity_function(self, T, time_upload, time_download, cpu_cycle_per_data_D=0.00947555555, local_train=1):
        """
        计算处理能力。

        :param T: 总时间
        :param time_upload: 上传时间
        :param time_download: 下载时间
        :param cpu_cycle_per_data_D: CPU周期数
        :param local_train: 本地训练轮数
        :return: 处理能力
        """
        return self.x * cpu_cycle_per_data_D * local_train / (T - (time_upload + time_download))

    def cost_function(self, cost_upload, cost_download, cost_investment, cost_energy_consumption,
                      T, time_upload, time_download, cpu_cycle_per_data_D=0.00947555555, local_train=1):
        """
        计算成本函数。

        :param cost_upload: 上传成本
        :param cost_download: 下载成本
        :param cost_investment: 投资成本
        :param cost_energy_consumption: 能耗成本
        :param T: 总时间
        :param time_upload: 上传时间
        :param time_download: 下载时间
        :param cpu_cycle_per_data_D: CPU周期数
        :param local_train: 本地训练轮数
        :return: 成本
        """
        processing_capacity_f = self.capacity_function(T, time_upload, time_download, cpu_cycle_per_data_D, local_train)
        cost = cost_upload + cost_download \
               + cost_investment * processing_capacity_f \
               + cost_energy_consumption * math.pow(self.x * cpu_cycle_per_data_D * local_train, 3) / math.pow((T - time_upload - time_download), 2)
        return cost

    def utility_function(self, loss):
        """
        计算效用函数。

        :param loss: 损失下降值
        :return: 效用
        """
        return self.p_n * loss

    def __str__(self):
        return f"Node {self.index}: x={self.x}, q={self.q}, p_n={self.p_n}, max_data_size={self.max_data_size}"


class FederatedLearningGame:
    def __init__(self, nodes, organization_index=0, k=0.1, T=60, epsilon=1):
        """
        初始化联邦学习博弈。

        :param nodes: 节点列表
        :param organization_index: 当前节点索引
        :param k: 当前精确度损失
        :param T: 总时间
        :param epsilon: 迭代终止阈值
        """
        self.nodes = nodes
        self.organization_index = organization_index
        self.k = k
        self.T = T
        self.epsilon = epsilon
        self.delta_x = float('inf')
        self.iteration = 0

        # 成本参数
        self.cost_upload = 20/50 * 7.6 * 0.174 / 3600000  # temp
        self.time_upload = 0.16 / 42.06  # temp
        self.cost_download = 20/50 * 7.6 * 0.174 / 3600000  # temp
        self.time_download = 0.16 / 78.26  # temp
        self.cost_investment = 0.22 * self.T / 3600  # temp
        self.cost_energy_consumption = math.pow(10, -26) * 0.174 / 3600000  # temp

        # 其他参数
        self.cpu_cycle_per_data_D = 0.00947555555
        self.local_train = 1

        # 日志路径
        self.log_dir = "C:/Users/crq/Desktop/postguaduate/persnoal experiments/experiments/incentive_py/"
        os.makedirs(self.log_dir, exist_ok=True)

        # 清空日志文件
        self.clear_node_log_files()

    def clear_node_log_files(self, log_filename_prefix="node_", log_filename_suffix="_log.log"):
        """
        清空每个节点的日志文件。

        :param log_filename_prefix: 日志文件前缀
        :param log_filename_suffix: 日志文件后缀
        """
        for node in self.nodes:
            log_filename = f"{log_filename_prefix}{node.index}{log_filename_suffix}"
            with open(os.path.join(self.log_dir, log_filename), 'w') as log_file:
                log_file.write(f"节点 {node.index} 博弈过程开始:\n")
                log_file.write("=" * 40 + "\n")  # 添加一个开始的分隔符

    def log_game_step(self, node, expected_datasize, new_datasize, delta_x, iteration, log_filename="game_log.log"):
        """
        记录每次迭代的日志。

        :param node: 节点对象
        :param expected_datasize: 期望数据大小
        :param new_datasize: 新数据大小
        :param delta_x: 本次迭代最大变化量
        :param iteration: 当前迭代次数
        :param log_filename: 日志文件名
        """
        with open(os.path.join(self.log_dir, log_filename), 'a') as log_file:
            log_file.write(f"第 {iteration} 次博弈循环:\n")
            log_file.write(f"节点 {node.index} 的期望数据大小：{expected_datasize}\n")
            log_file.write(f"节点 {node.index} 的新数据大小：{new_datasize}\n")
            log_file.write(f"本次迭代最大变化量：{delta_x}\n")
            log_file.write("-" * 40 + "\n")  # 分隔符

    def compute_Gk(self):
        """
        计算当前的 G_k。

        :return: G_k
        """
        sum_m = sum(node.q * node.x for node in self.nodes)
        denominator = 4 * (self.k ** 2) * sum_m
        molecular = (1 + np.sqrt(1 + 4 * self.k * sum_m)) ** 2
        G_k = molecular / denominator
        return G_k

    def compute_loss_decrease(self, node):
        """
        计算损失下降值。

        :param node: 当前节点
        :return: G_k, k_prime, loss_decrease
        """
        G_k = self.compute_Gk()

        # 计算 G_k + 1 时的精确度损失 k'
        sum_m_minus_n = sum(other_node.q * other_node.x for other_node in self.nodes if other_node.index != node.index)
        k_prime = 1 / (np.sqrt((node.q * node.x + sum_m_minus_n) * (G_k + 1))) + 1 / (G_k + 1)

        # 计算精确度损失的下降值 k - k'
        loss_decrease = self.k - k_prime
        return G_k, k_prime, loss_decrease

    def utility_function(self, loss, node):
        """
        计算效用函数。

        :param loss: 损失下降值
        :param node: 节点对象
        :return: 效用
        """
        return node.utility_function(loss)

    def cost_function(self, node):
        """
        计算成本函数。

        :param node: 节点对象
        :return: 成本
        """
        return node.cost_function(
            cost_upload=self.cost_upload,
            cost_download=self.cost_download,
            cost_investment=self.cost_investment,
            cost_energy_consumption=self.cost_energy_consumption,
            T=self.T,
            time_upload=self.time_upload,
            time_download=self.time_download,
            cpu_cycle_per_data_D=self.cpu_cycle_per_data_D,
            local_train=self.local_train
        )

    def compute_loss_function(self, datasize_n, node):
        """
        计算损失函数。

        :param datasize_n: 节点的新数据大小
        :param node: 节点对象
        :return: 损失函数值
        """
        original_x = node.x
        node.x = datasize_n
        G_k, k_prime, loss_decrease = self.compute_loss_decrease(node)
        Utility_n = self.utility_function(loss_decrease, node)
        Cost_n = self.cost_function(node)
        # 恢复原始x
        node.x = original_x
        return Cost_n - Utility_n

    def arg_max_for_loss_function_version2(self, node):
        """
        寻找使损失函数最大的datasize_n。

        :param node: 节点对象
        :return: 期望数据大小
        """
        bounds = (0, node.max_data_size)
        result = minimize_scalar(
            fun=lambda datasize_n: self.compute_loss_function(datasize_n, node),
            bounds=bounds,
            method='bounded',
        )
        expected_datasize = result.x
        print(f"组织节点 {node.index} 的期望数据大小：{expected_datasize}")
        return expected_datasize

    def plot_loss_function(self, node_index):
        """
        绘制指定节点的损失函数图。

        :param node_index: 节点索引
        """
        node = next((n for n in self.nodes if n.index == node_index), None)
        if node is None:
            print(f"节点 {node_index} 未找到。")
            return

        datasize_n_values = np.linspace(0, node.max_data_size, 100)
        loss_values = []
        for datasize_n in datasize_n_values:
            loss = self.compute_loss_function(datasize_n, node)
            loss_values.append(loss)
        plt.plot(datasize_n_values, loss_values)
        plt.xlabel('datasize_n')
        plt.ylabel('loss_function')
        plt.title(f'Node {node.index} Loss Function')
        plt.show()

    def compute_global_loss(self):
        """
        计算当前所有节点的损失函数值之和，用于收敛判断。
        您也可以根据需求定义其他衡量指标，如所有节点效用之和、成本之和等。
        """
        total_loss = 0.0
        for node in self.nodes:
            # 这里的 compute_loss_function 需要一个 datasize 和 node，对应 node 本身
            # 因为 compute_loss_function 会临时修改 node.x，然后再改回去
            # 为了方便，这里直接把 node.x 传进去
            loss_value = self.compute_loss_function(node.x, node)
            total_loss += loss_value
        return total_loss


    def run_game_loop(self, epsilon_loss=1e-3):
        """
        使用相对变化率作为额外收敛条件。
        :param epsilon_loss: 全局损失值相对变化率阈值
        """
        # 计算初始的全局损失值
        old_global_loss = self.compute_global_loss()
        
        while self.delta_x > self.epsilon:
            self.iteration += 1
            print(f"第 {self.iteration} 次博弈循环开始")

            old_x = [node.x for node in self.nodes]
            new_x = old_x.copy()

            # 并行（或串行）计算所有节点的 expected_datasize
            expected_datasize_list = []
            for node in self.nodes:
                expected_datasize = self.arg_max_for_loss_function_version2(node)
                expected_datasize_list.append(expected_datasize)

            # 并行（或串行）更新节点
            for idx, node in enumerate(self.nodes):
                # 这里保留原先的 0.5 步长示例，也可以使用自适应步长
                step_size = 0.5 
                diff = expected_datasize_list[idx] - old_x[node.index]

                updated_x = old_x[node.index] + step_size * diff
                updated_x = max(0, min(updated_x, node.max_data_size))
                updated_x = int(math.floor(updated_x))
                new_x[node.index] = updated_x

                print(f"节点 {node.index} 的期望数据大小：{expected_datasize_list[idx]}")
                print(f"节点 {node.index} 的新数据大小：{updated_x}")

                self.log_game_step(
                    node=node,
                    expected_datasize=expected_datasize_list[idx],
                    new_datasize=updated_x,
                    delta_x=self.delta_x,
                    iteration=self.iteration,
                    log_filename=f"node_{node.index}_log.log"
                )

            # 更新节点 x
            for node in self.nodes:
                node.x = new_x[node.index]

            # 计算新的 global_loss
            new_global_loss = self.compute_global_loss()

            # 计算相对变化率
            # 为了防止分母为0，这里加个极小值 1e-10
            relative_change = abs(new_global_loss - old_global_loss) / (abs(old_global_loss) + 1e-10)
            print(f"Global Loss Old: {old_global_loss:.5f}, New: {new_global_loss:.5f}, " 
                f"Relative Change: {relative_change:.6f}")

            # 更新 delta_x
            self.delta_x = max(abs(new_x[i] - old_x[i]) for i in range(len(self.nodes)))
            print(f"本次迭代最大变化量：{self.delta_x}\n")

            # 更新 old_global_loss
            old_global_loss = new_global_loss

            # 判断是否满足相对变化率的收敛条件
            # 这里是 “步长小于阈值” 且 “相对变化率小于阈值” 同时满足时，才停止
            if self.delta_x <= self.epsilon and relative_change <= epsilon_loss:
                print("根据步长和全局损失相对变化率，达到纳什均衡或足够近似!")
                break

        print("最终的数据大小列表为：", [node.x for node in self.nodes])
        for node in self.nodes:
            print(node)

        if len(self.nodes) > 3:
            self.plot_loss_function(3)
        else:
            print("节点数量不足以绘制节点 3 的损失函数。")

def main():
    # 初始化节点列表
    nodes = [
        Node(index=0, x=1000, q=1.00, p_n=7.9, max_data_size=5000),
        Node(index=1, x=400, q=1.05, p_n=7.8, max_data_size=2000),
        Node(index=2, x=100, q=1.002, p_n=7.878, max_data_size=2000),
        Node(index=3, x=500, q=0.96, p_n=7.87, max_data_size=1200)
    ]

    # 创建联邦学习博弈实例
    game = FederatedLearningGame(
        nodes=nodes,
        organization_index=0,  # 当前节点索引
        k=0.1,  # 当前的精确度损失
        T=60,  # 总时间
        epsilon=1  # 迭代终止阈值
    )

    # 运行博弈循环
    game.run_game_loop()


if __name__ == "__main__":
    main()
