import os
import json
import argparse
import logging

import yaml

from .general_utils import get_random_time_stamp, makedir_if_not_exist
from .logger import logging_info

dir_configs = os.path.join(os.getcwd(), "configs")

def stage_config_updated_params():
    return {
        "debug": False,
        "mode": "inpaint",
        "seed": 42,
        "n_samples": 1,
        "use_local_guid": False,
        "use_skip_x0": False,
        "skip_stop_step": 100,
        "optimize_xt.optimize_xt": True,
        "optimize_xt.num_iteration_inp": 2,
        "optimize_xt.num_iteration_guid": 2,
        "optimize_xt.coef_guid": 1.0,
        "optimize_xt.coef_guid_decay": 1.0,
        "optimize_xt.guid_stop_step": 50,
        "optimize_xt.inp_start_step": 249,
        "optimize_xt.use_comb": True,
        "optimize_xt.comb_start_step": 249,
        "optimize_xt.comb_stop_step": 100,
        "ddim.ddim_sigma": 0.0,
        "ddim.schedule_params.num_inference_steps": 250,
        "ddim.schedule_params.schedule_type": "linear",  # linear, quad, respace
        "ddim.schedule_params.infer_step_repace": "50,50,25,25,10",
        "ddim.schedule_params.jump_length": 10,
        "ddim.schedule_params.jump_n_sample": 1,
        "ddim.schedule_params.jump_start_step": 230,
        "ddim.schedule_params.jump_end_step": 0,
    }

def smart_load(path_file):
    if path_file.endswith("json"):
        return json.load(open(path_file, "r", encoding="utf-8"))
    elif path_file.endswith("yaml") or path_file.endswith("yml"):
        return yaml.safe_load(open(path_file, "r", encoding="utf-8"))
    else:
        logging.warning(
            "Un-identified file type. It will be processed as json by default."
        )
        return json.load(open(path_file, "r", encoding="utf-8"))


class NestedDict:
    def __init__(self, nested_dict=None):
        self._nested_dict = dict()
        if nested_dict is not None:
            self._nested_dict = nested_dict

    def __getitem__(self, key): # 使Object支持类似字典的键访问，支持 config["a.b.c"]
        ret = self._nested_dict
        for k in key.split("."):
            ret = ret[k]
        return ret

    def __setitem__(self, key, value): # 支持嵌套字典的修改，如 config["a.b.c"] = value
        key_list = key.split(".")
        cur_dict = self._nested_dict
        for i in range(len(key_list)):
            key = key_list[i]
            if i == len(key_list) - 1:
                cur_dict[key] = value
            else:
                if key in cur_dict.keys():
                    assert type(cur_dict[key]) is dict
                else:
                    cur_dict[key] = dict()
                cur_dict = cur_dict[key]

    def update(self, new_dict, prefix=None):
        for k in new_dict:
            key = ".".join([prefix, k]) if prefix is not None else k
            value = new_dict[k]
            if type(value) is dict:
                self.update(value, prefix=key) # 递归调用
            else:
                self[key] = value

    def keys(self, cur=None, prefix=None):
        if cur is None:
            cur = self._nested_dict

        ret = []
        for k in cur.keys():
            v = cur[k]
            new_prefix = ".".join([prefix, k]) if prefix is not None else k
            if type(v) is dict:
                ret += self.keys(cur=v, prefix=new_prefix)
            else:
                ret.append(new_prefix)
        return ret

    def show(self):
        """
        Show all the configs in logging. If get_logger is used before, then the outputs will also be in the log file.
        """
        logging_info(
            "\n%s"
            % json.dumps(
                self._nested_dict, sort_keys=True, indent=4, separators=(",", ": ")
            )
        )

    def to_dict(self):
        """
        Return the config as a dict
        :return: config dict
        :rtype: dict
        """
        return self._nested_dict


class Config(NestedDict):
    def __init__(
        self, default_config_file=None, default_config_dict=None, use_argparse=True
    ):
        """
        Initialize the config. Note that either default_config_dict or default_config_file in json format must be
        provided! The keys will be transferred to argument names, and the type will be automatically detected. The
        priority is ``the user specified parameter (if the use_argparse is True)'' > ``user specified config file (if
        the use_argparse is True)'' > ``default config dict'' > ``default config file''.

        Examples:
        default_config_dict = {"lr": 0.01, "optimizer": "sgd", "num_epoch": 30, "use_early_stop": False}
        Then the following corresponding arguments will be added in this function if use_argparse is True:
        parser.add_argument("--lr", type=float)
        parser.add_argument("--optimizer", type=str)
        parser.add_argument("--num_epoch", type=int)
        parser.add_argument("--use_early_stop", action="store_true", default=False)
        parser.add_argument("--no-use_early_stop", dest="use_early_stop", action="store_false")

        :param default_config_dict: the default config dict
        :type default_config_dict: dict
        :param default_config_file: the default config file path
        :type default_config_file: str
        """
        super(Config, self).__init__() # super()是对父类的继承， .__init__() 调用父类的方法/构造函数，具体为初始化方法

        # load from default config file
        if default_config_dict is None and default_config_file is None:
            if os.path.exists(os.path.join(os.getcwd(), "default_config.json")):
                default_config_file = os.path.join(os.getcwd(), "default_config.json")
            else:
                logging.error(
                    "Either default_config_file or default_config_dict must be provided!"
                )
                raise NotImplementedError

        if default_config_file is not None:
            self.update(smart_load(default_config_file))
        if default_config_dict is not None:
            self.update(default_config_dict)

        # transform the param terms into argparse
        if use_argparse:
            parser = argparse.ArgumentParser() # parser解析器; argument：函数中的参数；.ArgumentParser()是库argparse中的一个类，使用该类创建一个实例 parser 命令行参数解析器
            parser.add_argument("--config_file", type=str, default=None)

            # add argument parser
            for name_param in self.keys():
                value_param = self[name_param]
                if type(value_param) is bool:
                    parser.add_argument(
                        "--%s" % name_param,
                        action="store_true",
                        default=None,
                        # default=value_param
                    )
                    parser.add_argument(
                        "--no-%s" % name_param,
                        dest="%s" % name_param,
                        action="store_false",
                        default=None,
                    )
                elif type(value_param) is list:
                    parser.add_argument(
                        "--%s" % name_param,
                        type=type(value_param[0]),
                        # default=value_param,
                        nargs="+",
                        default=None,
                    )
                else:
                    parser.add_argument(
                        "--%s" % name_param,
                        type=type(value_param),
                        # default=value_param
                    )
            args = parser.parse_args() # 通过命令行参数解析器对象解析命令行，返回 argparse.Namespace 对象，可以使用 .参数名 的方式访问参数值

            updated_parameters = dict()#创建了一个空字典来存储更新后的参数
            self.subconfig_updated_params = dict()
            args_dict = vars(args) # 将args对象的所有属性、方法转换为一个字典
            for k in vars(args):
                if (
                    k not in ["config_file",] # 新添加的命令行参数解析在这里判断
                    and self[k] != args_dict[k]
                    and args_dict[k] is not None
                ):
                    updated_parameters[k] = args_dict[k]
                    if k in stage_config_updated_params().keys():
                        self.subconfig_updated_params[k] = updated_parameters[k]
            self.update(updated_parameters)

        """取消动态绑定"""
        # for k in self._nested_dict.keys(): # 将字典中的键 动态绑定到 对象的属性中，能够通过 .属性名 方法访问
        #     # assert 防止 保留属性和方法 被覆盖
        #     assert k != "_nested_dict"
        #     assert k != "keys"
        #     assert k != "__getitem__"
        #     assert k != "__setitem__"
        #     assert k != "update"
        #     assert k != "to_dict"
        #     assert k != "show"
        #     assert k != "dump"
        #     assert k != "get"
        #     if "." not in k: # 不能处理嵌套键，如 a.b.c ，只能通过 config["a.b.c"]访问
        #         setattr(self, k, self[k]) # 设置键名 k 作为对象的属性名，键值 v 作为对象的属性值
        #         # 使用的是旧值，更新dict后，属性值不会变

    def __getattr__(self, key): # 实现属性访问，并与字典访问共享一个字典，无法访问嵌套字典，嵌套字典依旧需要字典访问
        # print(key)
        if key in self.keys():
            return self[key]
        raise AttributeError(f"No attribute '{key}'")

    def dump(self, path_dump=None):
        """
        Dump the config in the path_dump.
        :param path_dump: the path to dump the config
        :type path_dump: str
        """
        if path_dump is None:
            makedir_if_not_exist(dir_configs)
            path_dump = os.path.join(dir_configs, "%s.json" % get_random_time_stamp())
        path_dump = (
            "%s.json" % path_dump if not path_dump.endswith(".json") else path_dump
        )
        assert not os.path.exists(path_dump)
        with open(path_dump, "w", encoding="utf-8") as fout:
            json.dump(self._nested_dict, fout, indent=4)

    def get(self, item, default_value=None):
        if item in self.keys():
            return self[item]
        else:
            if default_value is None:
                raise ValueError(f"No {item} in config")
        return default_value


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None)
    return parser


def parser2config(parser):
    return args2config(parser.parse_args())


def args2config(args):
    args_dict = vars(args)
    return Config(default_config_dict=args_dict, use_argparse=False)


def is_configs_same(config_a, config_b, ignored_keys=("load_epoch",)):
    config_a, config_b = config_a.to_dict(), config_b.to_dict()

    # make sure config A is always equal or longer than config B
    if len(config_a.keys()) < len(config_b.keys()):
        swap_var = config_a
        config_a = config_b
        config_b = swap_var

    if len(config_a.keys() - config_b.keys()) > 1:
        logging.error(
            "Different config numbers: %d (Existing) : %d (New)!"
            % (len(config_a.keys()), len(config_b.keys()))
        )
        return False
    elif (
        len(config_a.keys() - config_b.keys()) == 1
        and (config_a.keys() - config_b.keys())[0] != "config_file"
    ):
        logging.error(
            "Different config numbers: %d (Existing) : %d (New)!"
            % (len(config_a.keys()), len(config_b.keys()))
        )
        return False
    else:
        for i in config_a.keys() & config_b.keys():
            _ai = tuple(config_a[i]) if type(config_a[i]) == list else config_a[i]
            _bi = tuple(config_b[i]) if type(config_b[i]) == list else config_b[i]
            if _ai != _bi and i not in ignored_keys:
                logging.error(
                    "Mismatch in %s: %s (Existing) - %s (New)"
                    % (str(i), str(config_a[i]), str(config_b[i]))
                )
                return False

    return True
