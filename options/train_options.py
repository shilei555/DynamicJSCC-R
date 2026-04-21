from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.isTrain = None

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # network saving and loading parameters
        parser.add_argument('--continue_train', type=bool, default=False)
        parser.add_argument('--weights_dir', type=str, default="./checkpoints")


        # training parameters
        parser.add_argument('--epochs', type=int, default=300, help='number of train epochs ')
        parser.add_argument('--lr_max', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--lr_min', type=float, default=0.000001, help=' eta_min for CosineAnnealingLR')
        parser.add_argument('--beta', type=float, default=200.0, help='weight for mse loss')

        # val
        parser.add_argument('--use_eval', type=int, default=1,
                            help='use eval mode during train time to save the best epoch model')

        # ==================== 可视化参数 ====================
        parser.add_argument('--save_vis', action='store_true', default=True,
                            help='Save visualization images')
        parser.add_argument('--vis_interval', type=int, default=100,
                            help='Save visualization every N batches')
        parser.add_argument('--vis_max_samples', type=int, default=8,
                            help='Maximum number of samples to visualize')
        parser.add_argument('--verbose', action='store_true', default=True,
                            help='Print detailed information')



        self.isTrain = True
        return parser
