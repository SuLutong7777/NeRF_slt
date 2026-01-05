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

        self.sys_param['rot_noise'] = sys_param['system']['noise']['rot_noise']
        self.sys_param['trans_noise'] = sys_param['system']['noise']['trans_noise']

        self.sys_param['output_root'] = sys_param['system']['output_params']['output_root']
        self.sys_param['output_path'] = os.path.join(self.sys_param['output_root'], self.sys_param['data_name'])

        self.sys_param['test_model_path'] = sys_param['system']['train_test_params']['test_model_path']
        self.sys_param['batch_size'] = sys_param['system']['train_test_params']['batch_size']
        self.sys_param['cam_lr_max'] = sys_param['system']['train_test_params']['cam_lr_max']
        self.sys_param['cam_lr_min'] = sys_param['system']['train_test_params']['cam_lr_min']
        self.sys_param['opt_lr_max'] = sys_param['system']['train_test_params']['lr_max']
        self.sys_param['opt_lr_min'] = sys_param['system']['train_test_params']['lr_min']
        self.sys_param['opt_type'] = sys_param['system']['train_test_params']['opt_type']
        self.sys_param['opt_decay'] = sys_param['system']['train_test_params']['weight_decay']
        self.sys_param['rgb_loss_weight'] = sys_param['system']['train_test_params']['rgb_loss_weight']
        self.sys_param['dist_loss_weight'] = sys_param['system']['train_test_params']['dist_loss_weight']
        self.sys_param['inter_loss_weight'] = sys_param['system']['train_test_params']['inter_loss_weight']
        self.sys_param['hash_loss_weight'] = sys_param['system']['train_test_params']['hash_loss_weight']
        self.sys_param['prop_pulse_width'] = sys_param['system']['train_test_params']['prop_pulse_width']

        self.sys_param['num_steps'] = sys_param['system']['steps']['num_steps']
        self.sys_param['save_steps'] = sys_param['system']['steps']['save_steps']
        self.sys_param['val_steps'] = sys_param['system']['steps']['val_steps']
        
        self.sys_param['near'] = sys_param['model']['near']
        self.sys_param['far'] = sys_param['model']['far']
        
        self.sys_param['prop_samples'] = sys_param['model']['prop']['samples']
        self.sys_param['prop_num_level'] = sys_param['model']['prop']['encoding']['num_level']
        self.sys_param['prop_level_dim'] = sys_param['model']['prop']['encoding']['level_dim']
        self.sys_param['prop_base_resolution'] = sys_param['model']['prop']['encoding']['base_resolution']
        self.sys_param['prop_level_scale'] = sys_param['model']['prop']['encoding']['level_scale']
        self.sys_param['prop_log2_hashmap_size'] = sys_param['model']['prop']['encoding']['log2_hashmap_size']
        self.sys_param['prop_MLP_width'] = sys_param['model']['prop']['mlp']['MLP_width']

        self.sys_param['render_samples'] = sys_param['model']['render']['samples']
        self.sys_param['render_num_level'] = sys_param['model']['render']['encoding']['num_level']
        self.sys_param['render_level_dim'] = sys_param['model']['render']['encoding']['level_dim']
        self.sys_param['render_base_resolution'] = sys_param['model']['render']['encoding']['base_resolution']
        self.sys_param['render_level_scale'] = sys_param['model']['render']['encoding']['level_scale']
        self.sys_param['render_log2_hashmap_size'] = sys_param['model']['render']['encoding']['log2_hashmap_size']
        self.sys_param['render_freqs_view'] = sys_param['model']['render']['encoding']['freqs_view']
        self.sys_param['render_MLP_width'] = sys_param['model']['render']['mlp']['MLP_width']
        self.sys_param['render_MLP_depth'] = sys_param['model']['render']['mlp']['MLP_depth']
        self.sys_param['render_MLP_skips'] = sys_param['model']['render']['mlp']['MLP_skips']
