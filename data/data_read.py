from data import Load_Blender, Load_LLFF, Load_mip360

class Data_set():
    def __init__(self, sys_param):
        self.sys_param = sys_param
        self.pca_mode = self.sys_param['pca_mode']
        # 初始化数据集
        self.data_type = self.sys_param['data_type']
        if self.data_type == 'blender':
            self.data = Load_Blender(self.sys_param)
        elif self.data_type == 'llff':
            self.data = Load_LLFF(self.sys_param)
        elif self.data_type == 'mip360':
            self.data = Load_mip360(self.sys_param)
        # 图像
        self.imgs_train, self.imgs_test, self.imgs_val = self.data.imgs_train, self.data.imgs_test, self.data.imgs_val
        # 外参
        self.c2ws_pca_train, self.c2ws_pca_test, self.c2ws_pca_val = self.data.c2ws_pca_train, self.data.c2ws_pca_test, self.data.c2ws_pca_val
        self.radius_pca, self.scene_center_pca, self.scene_radius_pca = self.data.radius_pca, self.data.scene_center_pca, self.data.scene_radius_pca
        self.c2ws_train, self.c2ws_test, self.c2ws_val = self.data.c2ws_train, self.data.c2ws_test, self.data.c2ws_val
        self.radius, self.scene_center, self.scene_radius = self.data.radius, self.data.scene_center, self.data.scene_radius
        print("radius_pca, radius: ", self.radius_pca, self.radius)
        print("scene_center_pca, scene_center: ", self.scene_center_pca, self.scene_center)
        print("scene_radius_pca, scene_radius: ", self.scene_radius_pca, self.scene_radius)
        # 内参
        self.intrs_train, self.intrs_test, self.intrs_val = self.data.intrs_train, self.data.intrs_test, self.data.intrs_val
        self.intrs_inv_train, self.intrs_inv_test, self.intrs_inv_val = self.data.intrs_inv_train, self.data.intrs_inv_test, self.data.intrs_inv_val
        # 数量 索引
        self.train_num, self.test_num, self.val_num = self.data.train_num, self.data.test_num, self.data.val_num
        self.idx_train, self.idx_test, self.idx_val = self.data.idx_train, self.data.idx_test, self.data.idx_val
        self.img_h, self.img_w = self.data.img_h, self.data.img_w
        self.update_sys_param = self.update_param(self.sys_param)

    def getTrainData(self):
        if self.pca_mode:
            return self.imgs_train, self.c2ws_pca_train, self.intrs_train, self.intrs_inv_train, self.idx_train, self.train_num
        else:
            return self.imgs_train, self.c2ws_train, self.intrs_train, self.intrs_inv_train, self.idx_train, self.train_num
    
    def getTestData(self):
        if self.pca_mode:
            return self.imgs_test, self.c2ws_pca_test, self.intrs_test, self.intrs_inv_test, self.idx_test, self.test_num
        else:
            return self.imgs_test, self.c2ws_test, self.intrs_test, self.intrs_inv_test, self.idx_test, self.test_num
    
    def getValData(self):
        if self.pca_mode:
            return self.imgs_val, self.c2ws_pca_val, self.intrs_val, self.intrs_inv_val, self.idx_val, self.val_num
        else:
            return self.imgs_val, self.c2ws_val, self.intrs_val, self.intrs_inv_val, self.idx_val, self.val_num
    
    def update_param(self, sys_param):
        if self.pca_mode:
            sys_param['radius'] = self.radius_pca
            sys_param['scene_center'] = self.scene_center_pca
            sys_param['scene_radius'] = self.scene_radius_pca
        else:
            sys_param['radius'] = self.radius
            sys_param['scene_center'] = self.scene_center
            sys_param['scene_radius'] = self.scene_radius
        return sys_param
