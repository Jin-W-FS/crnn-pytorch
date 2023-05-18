
common_config = {
    'train_data_dir': '/home/xcy/Data/plates/data/',
    'tests_data_dir': '/home/xcy/Data/plates/test/',
    'img_width': 140,
    'img_height': 32,
    'img_channel': 3,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
    'leaky_relu': False,
    'reload_checkpoint': 'weights/crnn_lpr-230518.pt',
}

train_config = {
    'epochs': 10000,
    'train_batch_size': 32,
    'eval_batch_size': 512,
    'lr': 0.000005,
    'show_interval': 10,
    'valid_interval': 500,
    'save_interval': 2000,
    'cpu_workers': 4,
    'reload_checkpoint': None,
    'valid_max_iter': 100,
    'decode_method': 'greedy',
    'beam_size': 10,
    'checkpoints_dir': 'checkpoints/'
}
train_config.update(common_config)

evaluate_config = {
    'eval_batch_size': 512,
    'cpu_workers': 4,
    'decode_method': 'beam_search',
    'beam_size': 10,
}
evaluate_config.update(common_config)
