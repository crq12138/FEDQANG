import math


def compute_cost(data_size, total_time=60.0):
    cost_upload = 20 / 50 * 7.6 * 0.174 / 3600000
    cost_download = 20 / 50 * 7.6 * 0.174 / 3600000
    cost_investment = 0.22 * total_time / 3600
    cost_energy_consumption = math.pow(10, -26) * 0.174 / 3600000
    time_upload = 0.16 / 42.06
    time_download = 0.16 / 78.26
    cpu_cycle_per_data_d = 0.00947555555
    local_train = 1
    processing_capacity = (
        data_size * cpu_cycle_per_data_d * local_train
    ) / (total_time - time_upload - time_download)
    energy_term = cost_energy_consumption * math.pow(
        data_size * cpu_cycle_per_data_d * local_train, 3
    ) / math.pow((total_time - time_upload - time_download), 2)
    return cost_upload + cost_download + cost_investment * processing_capacity + energy_term
