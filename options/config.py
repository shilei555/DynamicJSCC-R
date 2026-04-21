import argparse
import yaml
import utils

required_args = ['dataroot']


class ConfigParser:
    def __init__(self, yaml_path: str = None, is_train: bool = True):
        # 读取yaml文件并解析成字典
        self.yaml_path = yaml_path
        self.config_dict = self.load_yaml()
        # 选择训练或测试选项类
        options = 'TrainOptions' if is_train else 'TestOptions'
        options = utils.find_class_using_name(options, 'options')()
        # 使用config文件对可选参数进行解析
        self.config = options.parse(self.config_dict)

    # 加载yaml文件并解析成一级字典
    def load_yaml(self):
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        config_dict = {}
        for _, val in config.items():
            config_dict.update(val)
        # 检测yaml文件中是否包含所有必需的参数
        for arg in required_args:
            if arg not in config_dict:
                raise ValueError(f"Missing required argument '%s' in %s :" % (arg, self.yaml_path))
        return config_dict

    def get_config(self):
        return self.config


# 使用示例
if __name__ == "__main__":

    config = ConfigParser(r'D:\code\image compression\mine\config\train.yaml').get_config()
    print(config)
