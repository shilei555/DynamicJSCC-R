import argparse  # 用来解析命令行参数和选项的
import os
import torch


class BaseOptions:
    def __init__(self):
        # 初始化,说明还没有初始化
        self.initialized = False

    def initialize(self, parser):
        # 初始化被用于训练和测试的参数

        # 基础参数
        parser.add_argument('--name', type=str, default='DynamicJSCC-R',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--save_dir', type=str, default='./checkpoints', help='models are saved here')

        # 数据集参数
        parser.add_argument('--dataroot', default='./datasets',
                            help='path to images')
        parser.add_argument('--shuffle', type=bool,
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--preprocess', type=str, default='crop',
                            help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--crop_size', type=int, default=320, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        # model
        parser.add_argument('--K_max', type=int, default=48,
                            help='Maximum code length')
        parser.add_argument('--c', type=int, default=128,
                            help='Intermediate feature channel')
        parser.add_argument('--in_channels', type=int, default=3,
                            help='Number of input channels')
        parser.add_argument('--out_channels', type=int, default=3,
                            help='Number of output channels')
        # SNR 参数
        parser.add_argument('--snr_db_max', type=float, default=28.0,
                            help='Maximum SNR in dB')
        parser.add_argument('--snr_db_min', type=float, default=0.0,
                            help='Minimum SNR in dB')

        #  CR 参数
        parser.add_argument('--cr_max', type=float, default=1.0,
                            help='Maximum compression rate')
        parser.add_argument('--cr_min', type=float, default=0.1,
                            help='Minimum compression rate')


        self.initialized = True
        return parser

    def gather_options(self, configs):
        # 初始化parser并添加基本选项
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 创建一个ArgumentParser对象
            parser = self.initialize(parser)
        # 使用config里的属性对选项进行赋值
        config_ns = argparse.Namespace(**configs)
        args = parser.parse_args([], namespace=config_ns)
        return args

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        print(message)

        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, config):
        opt = self.gather_options(config)
        # 设置GPU ID
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        return opt
