class Config:
    def __init__(self):
        # Number of classes in the dataset
        self.num_classes = 9

        self.architecture = 'resnext50'
        self.learning_rate = 1e-5
        self.momentum= 0.98
        self.weight_decay = 1e-4
        # self.lr_decays': {i: 0.1 ** (1 / 30) for i in range(1, max_epoch, 20)},
        # self.grad_clip_norm': 2,  # 100.0,
        self.residual_block_list = [3, 4, 6, 3]
        self.init_conv_filters = 64
        self.init_conv_kernel_size = 7
        self.init_conv_strides = 2
        self.init_pooling_pool_size = 3
        self.init_pooling_strides = 2
        self.cardinality = 32
        self.is_SENet = False
        self.reduction = 16
        self.input_shape = (None, 224, 224, 3)
