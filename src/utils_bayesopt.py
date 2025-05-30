import os
import numpy as np


def get_output(fun_target, bx):
    output = fun_target(bx)
    if isinstance(output, np.ndarray) and np.all(output.shape == (1, 1)):
        output = output[0, 0]

    assert isinstance(output, float)
    return output


def save_bo(path_results, dict_info, str_task):
    if not os.path.exists(path_results):
        os.mkdir(path_results)

    str_prefix = f"bo_{str_task}_{dict_info['str_target']}_model_{dict_info['str_model']}_init_{dict_info['num_init']}_iter_{dict_info['num_iter']}"

    if str_task == "sampling":
        str_file = f"{str_prefix}_sampling_{dict_info['info_sampling']}_seed_{dict_info['seed']}.npy"
    elif str_task == "pool":
        str_file = f"{str_prefix}_pool_{dict_info['size_pool']}_seed_{dict_info['seed']}.npy"
    else:
        raise ValueError

    np.save(os.path.join(path_results, str_file), dict_info)
