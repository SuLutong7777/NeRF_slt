import yaml
import os

class Load_Config:
    def __init__(self, args):
        # 获取配置文件的路径
        self.config_path = args.config
        self.sys_param = {}
        # 加载yaml文件参数
        self.load_yaml()
        # 加载命令行参数
        self.load_terminal(args)

    # 加载命令行参数
    def load_terminal(self, args):
        return
    
    # 加载yaml文件参数
    def load_yaml(self):
        # 读取yaml文件
        with open(self.config_path, 'r', encoding='utf-8') as f:
            sys_param = yaml.safe_load(f)
        # 加载数据集路径
        self.sys_param['device'] = sys_param['system']['device']['device']
        self.sys_param['tb_mode'] = sys_param['system']['tensorboard']['tb_mode']

        self.sys_param['data_type'] = sys_param['system']['data']['data_type']
        self.sys_param['data_root'] = sys_param['system']['data']['data_root']
        self.sys_param['data_name'] = sys_param['system']['data']['data_name']
        self.sys_param['data_path'] = os.path.join(self.sys_param['data_root'], self.sys_param['data_name'])
        self.sys_param['img_scale'] = sys_param['system']['data']['img_scale']
        self.sys_param['pca_mode'] = sys_param['system']['data']['pca_mode']
        self.sys_param['white_background'] = sys_param['system']['data']['white_background']

        self.sys_param['output_root'] = sys_param['system']['output_params']['output_root']
        self.sys_param['output_path'] = os.path.join(self.sys_param['output_root'], self.sys_param['data_name'])

        self.sys_param['test_model_path'] = sys_param['system']['train_test_params']['test_model_path']
        self.sys_param['batch_size'] = sys_param['system']['train_test_params']['batch_size']
        self.sys_param['only_coarse'] = sys_param['system']['train_test_params']['only_coarse']
        self.sys_param['learning_rate'] = sys_param['system']['train_test_params']['learning_rate']
        self.sys_param['lr_mlp'] = sys_param['system']['train_test_params']['lr_mlp']
        self.sys_param['lr_hash'] = sys_param['system']['train_test_params']['lr_hash']
        self.sys_param['learning_rate_decay'] = sys_param['system']['train_test_params']['learning_rate_decay']
        self.sys_param['weight_decay'] = sys_param['system']['train_test_params']['weight_decay']
        
        self.sys_param['num_steps'] = sys_param['system']['steps']['num_steps']
        self.sys_param['save_steps'] = sys_param['system']['steps']['save_steps']
        self.sys_param['val_steps'] = sys_param['system']['steps']['val_steps']
        self.sys_param['rays_visul'] = sys_param['system']['steps']['rays_visul']
        self.sys_param['rays_step'] = sys_param['system']['steps']['rays_step']
        
        self.sys_param['num_level'] = sys_param['model']['encoding']['num_level']
        self.sys_param['level_dim'] = sys_param['model']['encoding']['level_dim']
        self.sys_param['base_resolution'] = sys_param['model']['encoding']['base_resolution']
        self.sys_param['level_scale'] = sys_param['model']['encoding']['level_scale']
        self.sys_param['log2_hashmap_size'] = sys_param['model']['encoding']['log2_hashmap_size']
        self.sys_param['freqs_view'] = sys_param['model']['encoding']['freqs_view']

        self.sys_param['near'] = sys_param['model']['nerf']['near']
        self.sys_param['far'] = sys_param['model']['nerf']['far']
        self.sys_param['coarse_MLP_depth'] = sys_param['model']['nerf']['coarse_MLP_depth']
        self.sys_param['coarse_MLP_width'] = sys_param['model']['nerf']['coarse_MLP_width']
        self.sys_param['coarse_MLP_skips'] = sys_param['model']['nerf']['coarse_MLP_skips']
        self.sys_param['num_coarse_samples'] = sys_param['model']['nerf']['num_coarse_samples']
        self.sys_param['fine_MLP_depth'] = sys_param['model']['nerf']['fine_MLP_depth']
        self.sys_param['fine_MLP_width'] = sys_param['model']['nerf']['fine_MLP_width']
        self.sys_param['fine_MLP_skips'] = sys_param['model']['nerf']['fine_MLP_skips']
        self.sys_param['fine_weight_threshold'] = sys_param['model']['nerf']['fine_weight_threshold']
        self.sys_param['fine_samples_scale'] = sys_param['model']['nerf']['fine_samples_scale']
        self.sys_param['sigmas_default'] = sys_param['model']['nerf']['sigmas_default']