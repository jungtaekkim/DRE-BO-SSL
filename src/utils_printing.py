def print_separators():
    num_separators = 30
    print('=' * num_separators, flush=True)


def print_info(str_model, str_target, seed, num_init, num_iter, info_sampling):
    print_separators()
    print(f'str_model {str_model}', flush=True)
    print(f'str_target {str_target}', flush=True)
    print(f'seed {seed}', flush=True)
    print(f'num_init {num_init}', flush=True)
    print(f'num_iter {num_iter}', flush=True)
    print(f'info_sampling {info_sampling}', flush=True)
    print_separators()
    print('', flush=True)
