import logging
import os
import json
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from PIL import Image
import argparse
import yaml
import torch.nn.functional as F

from datasets.utils import normalize
from guided_diffusion import (
    DDIMSampler,
    O_DDIMSampler,
    G_DDIMSampler,
)
from guided_diffusion import dist_util
from guided_diffusion.respace import SpacedDiffusion
from guided_diffusion.script_util import (
    model_defaults,
    create_model,
    diffusion_defaults,
    create_gaussian_diffusion,
    select_args,
    create_classifier,
    classifier_defaults,
)
from metrics import LPIPS, PSNR, SSIM, Metric
from utils import save_grid, save_image, normalize_image
from utils.config import Config
from utils.logger import get_logger, logging_info
from utils.nn_utils import get_all_paths, set_random_seed
from utils.result_recorder import ResultRecorder
from utils.timer import Timer
from datasets.masks import mask_generators

def prepare_model(algorithm, conf, device):
    # logging_info("Prepare model...")
    unet = create_model(**select_args(conf, model_defaults().keys()), conf=conf)
    SAMPLER_CLS = {
        "ddim": DDIMSampler,
        "repaint": SpacedDiffusion,
        "copaint": O_DDIMSampler,
        "guidpaint": G_DDIMSampler,
    }
    sampler_cls = SAMPLER_CLS[algorithm]
    sampler = create_gaussian_diffusion(
        **select_args(conf, diffusion_defaults().keys()),
        conf=conf,
        base_cls=sampler_cls,
    )

    logging_info(f"Loading model from {conf.model_path}...")
    unet.load_state_dict(
        dist_util.load_state_dict(
            os.path.expanduser(conf.model_path), map_location="cpu"
        ), strict=False
    )
    unet.to(device)
    if conf.use_fp16:
        unet.convert_to_fp16()
    unet.eval()
    return unet, sampler

def prepare_classifier(conf, device):
    # logging_info("Prepare classifier...")
    classifier = create_classifier(
        **select_args(conf, classifier_defaults().keys()))
    logging_info(f"Loading classifier from {conf.classifier_path}...")
    classifier.load_state_dict(
        dist_util.load_state_dict(
            os.path.expanduser(conf.classifier_path), map_location="cpu"
        )
    )
    classifier.to(device)
    classifier.eval()
    return classifier

def all_exist(paths):
    for p in paths:
        if not os.path.exists(p):
            return False
    return True

def main():
    config = Config(default_config_file="configs/config.yaml", use_argparse=True)
    if config["stage1.algorithm"] == "guidpaint":
        config.update({"use_classifier":True})
    config.show()

    # # 分配gpu，配置device
    gpu = config.get("gpu",0)
    print(f"指定的GPU: {gpu}")
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")  # 字符串格式"cuda:0",中间必须是冒号，在 PyTorch 中，设备字符串的格式如下，使用CPU为 "cpu"；使用GPU为 "cuda:0"

    # 输出目录
    outdir = os.path.join(config.output, os.path.splitext(os.path.basename(config.input))[0])
    os.makedirs(outdir, exist_ok=True)

    """
        Stage 1: stochastic sampling
    """
    logging_info("Start stage1: inpainting segmented objects...")
    ###################################################################################
    # prepare config, logger and recorder
    ###################################################################################
    config_stage1_path = config.get("stage1.config_path", "./configs/imagenet.yaml")
    config_stage1 = Config(default_config_file=config_stage1_path, use_argparse=False)
    status = []

    config_stage1.update(config.subconfig_updated_params) # 更新子配置文件的公共参数
    logging_info(f"subconf unpdate:{config.subconfig_updated_params}")

    # 更新子配置文件的特别参数
    if config["stage1.algorithm"] != "":
        config_stage1.update({"algorithm": config["stage1.algorithm"]}) # 更新的是对象的字典的一个键值对
        # config_stage1.algorithm = config.get("stage1.algorithm","copaint") # 更新的是对象的一个属性，属性名 algorithm
    status.append(config_stage1.algorithm)

    if config_stage1.algorithm in ["ddim", "copaint", "guidpaint"]:
        # num_inference_steps = config.get("ddim.schedule_params.num_inference_steps", 250)
        if config_stage1["ddim.schedule_params.schedule_type"] == "respace":
            status.append("respace"+"("+config_stage1["ddim.schedule_params.infer_step_repace"]+")")

        if config["stage1.use_timetravel"]:
            config_stage1.update({"ddim.schedule_params.use_timetravel":True})
            jump_length = config_stage1["ddim.schedule_params.jump_length"]
            jump_n_sample = config_stage1["ddim.schedule_params.jump_n_sample"]
            jump_start = config_stage1["ddim.schedule_params.jump_start_step"]
            jump_end = config_stage1["ddim.schedule_params.jump_end_step"]
            status.append("_".join(["tt", str(jump_length), str(jump_n_sample),str(jump_start), str(jump_end)]))

    if config_stage1.algorithm in ["guidpaint",]:
        if config["stage1.use_guidance"]:
            config_stage1.update({"use_guidance": True})
            if config.use_local_guid:
                status.append("local_guid" + str(config_stage1["optimize_xt.coef_guid"]))
            else:
                status.append("global_guid" + str(config_stage1["optimize_xt.coef_guid"]))

        if config_stage1["optimize_xt.optimize_xt"]:
            status.append("inp" + str(config_stage1["optimize_xt.inp_start_step"]))

        if config_stage1["optimize_xt.use_comb"]:
            status.append("comb" + str(config_stage1["optimize_xt.comb_start_step"]) +"_"+ str(config_stage1["optimize_xt.comb_stop_step"]))

    config_stage1.show()

    status = '-'.join(status)
    logging_info(f"status:{status}")

    all_paths = get_all_paths(outdir,"stage1")
    config_stage1.dump(all_paths["path_config"])
    get_logger(all_paths["path_log"], force_add_handler=True)
    recorder_stage1 = ResultRecorder(
        path_record=all_paths["path_record"],
        initial_record=config_stage1,
        use_git=config_stage1.use_git,
    )
    set_random_seed(config_stage1.seed, deterministic=False, no_torch=False, no_tf=True)

    ###################################################################################
    # prepare data
    ###################################################################################
    datas = []  # 定义一个空列表来存储分割图片、掩码和标签
    masks = []
    data_label = None
    if config.mask != "":
        if os.path.isdir(config.mask):
            mask_paths = [os.path.join(config.mask, f) for f in os.listdir(config.mask) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            for mask_path in mask_paths:
                mask = (
                    torch.from_numpy(np.array(Image.open(mask_path).resize((256, 256)).convert("1"), dtype=np.float32))
                    # .convert("1")将其转换为二值图像（黑0白1图像）,.resize()确保能够处理任意size的掩码
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                masks.append(mask)
        else:
            mask = (
                torch.from_numpy(np.array(Image.open(config.mask).resize((256, 256)).convert("1"), dtype=np.float32))
                # .convert("1")将其转换为二值图像（黑0白1图像）,.resize()确保能够处理任意size的掩码
                .unsqueeze(0)
                .unsqueeze(0)
            )
            masks.append(mask)
    elif config.mask_type != "":
        logging_info(f"mask_type: {config.mask_type}")
        mask_generator = mask_generators[config.mask_type]
        shape = config.get("shape", 256)
        mask = mask_generator((shape, shape))
        masks.append(mask)
    else:
        logging.warning("Must provide mask_type or mask_path or mask_dir for inpainting")

    masks = torch.cat(masks, dim=1)
    logging_info(f"mask num: {masks.shape[1]}, masks shape: {masks.shape}")

    # 获取输入的gt图像的路径
    if not os.path.isdir(config.input):  # 如果不是目录，即单张图片
        targets = [config.input]
        data_label_dir = os.path.dirname(config.input)
        for f in os.listdir(data_label_dir):
            if f.endswith(".json"):
                data_label_path = os.path.join(data_label_dir, f)
                with open(data_label_path, "r", encoding="utf-8") as f:
                    data_label = json.load(f)
    else:
        targets = [
            f for f in os.listdir(config.input) if not os.path.isdir(os.path.join(config.input, f))
        ]
        targets = [os.path.join(config.input, f) for f in targets]

    for target in targets:
        if target.endswith("json"):
            with open(target, "r", encoding="utf-8") as f:
                data_label = json.load(f)

    names = []  # 定义空列表来存储gt图像的名字
    gt_paths = []
    for target in targets:
        if target.lower().endswith((".jpg", ".jpeg", ".png")):
            # 载入初始图像
            logging_info(f"Processing {target}...")
            gt_paths.append(target)
            image_original = normalize(Image.open(target).convert("RGB"))  # 处理任意形状的图像
            logging_info(f"image_original shape: {image_original.shape}")
            name = os.path.splitext(os.path.basename(target))[0]
            names.append(name)

            # 自定义类标签 class_labels
            if config.labels != "":  # 用于对于单张图像
                class_labels = [int(label) for label in config.labels.split(",")] # 处理单张图片，不适合批处理
                if len(class_labels) < masks.shape[1]:  # 异常处理
                    logging.warning("Labels num less than masks num")
                    assert class_labels  # class_labels 不为空
                    while len(class_labels) < masks.shape[1]:
                        class_labels.append(class_labels[-1])  # 在class labels末尾添加末尾类标签
            elif data_label is not None and data_label[name]["class_label"] is not None: # 对于多个图像，适用于批处理，自定义的多个标签
                class_labels = [data_label[name]["class_label"]] * masks.shape[1]
            else:
                class_labels = [0] * masks.shape[1]
            logging_info(f"input class labels: {class_labels}")

            data = (image_original, masks, name, class_labels)
            datas.append(data)


    # ##################################################################################
    # prepare model and device and metics loss
    # ##################################################################################
    logging_info("Prepare stage1 model...")

    unet, sampler = prepare_model(config_stage1.algorithm, config_stage1, device)

    def model_fn(x, t, y=None, gt=None, **kwargs):
        # assert y is not None
        """
        对于CelebA-HQ,Places数据集上预训练的Diffusion Model是无条件的（Unconditional），没有标签y
        对于ImageNet数据集上预训练的Diffusion Model是有条件的（Conditional），有标签y
        """
        return unet(x, t, y if config_stage1.class_cond else None, gt=gt)

    if config.use_classifier:
        classifier = prepare_classifier(config_stage1, device)
        def cond_fn(x, t, guid_y=None, gt=None, gt_keep_mask=None, **kwargs):
            assert guid_y is not None
            # assert gt_keep_mask is not None
            logits = classifier(x, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), guid_y.view(-1)] # y.view(-1),将张量 y 展平（flatten）成一维形式
            return selected
    else:
        cond_fn = None

    METRICS = {
        "lpips": Metric(LPIPS("alex", device)),
        "psnr": Metric(PSNR(), eval_type="max"),
        "ssim": Metric(SSIM(), eval_type="max"),
    }
    final_loss = []

    # ###################################################################################
    # # start sampling
    # ###################################################################################
    logging_info("Start stage1 sampling...")
    timer, num_image = Timer(), 0
    batch_size = config_stage1.n_samples

    for i, d in enumerate(tqdm(datas)):
        image, masks, image_name, labels = d

        logging_info(f"Stochastic Sampling '{image_name}'...")

        # prepare save dir
        outpath = os.path.join(outdir, image_name)
        os.makedirs(outpath, exist_ok=True)
        sample_dir = os.path.join(outpath, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        base_count = len(os.listdir(sample_dir))
        if config.debug:
            grid_count = max(len(os.listdir(outpath)) - 8, 0)
        else:
            grid_count = max(len(os.listdir(outpath)) - 7, 0)


        # prepare batch data for processing
        batch = {
            "image": image.to(device),
            "masks": masks.to(device),
        }

        model_kwargs = {
            "gt": batch["image"].repeat(batch_size, 1, 1, 1),
            "masks": batch["masks"].repeat(batch_size, 1, 1, 1),
        }

        shape = (batch_size, 3, config_stage1.image_size, config_stage1.image_size)

        if config.use_pred_y:
            assert cond_fn is not None, f"cond_fn is None"
            # 用初始图像x，t=0 通过 Classifier() 得到预测的类标签 y_pred
            x = model_kwargs["gt"]
            t_0 = torch.tensor([0] * shape[0], device=device)
            logits = classifier(x, t_0) # (B,1000)
            log_probs = F.log_softmax(logits, dim=-1) # (B,1000)
            y_pred = torch.topk(log_probs, k=config.pred_y_top_k, dim=-1)[1]
                # torch.topk返回一个命名元组(tensor,indices)，tensor在dim维度为k,(B,k)，indices形状与tensor相同(B,k)
                # 若 B=1，k=3，则indicese 为 tensor[[18,11,20]]
            selected = log_probs[range(len(logits)), y_pred.view(-1)]
                # len(tensor) 返回的是张量的 第一个维度的大小
                # 高级索引log_probs[0,:]，第一个维度的索引0的所有对数概率,形状为(1000,)
                # log_probs[0,[18,20,11]]返回对应索引的对数概率，形状为(3,)
            logging_info(f"pred_y: {y_pred.tolist()}, log_prob_pred_y: {selected.tolist()}")
            del x, t_0, logits, log_probs
            torch.cuda.empty_cache()

            y_tuple = torch.split(y_pred,1, dim=1)
            logging_info(f"pred_y_tupe:{y_tuple}")
        else:
            if config_stage1.cond_y is not None:
                classes = torch.ones(batch_size, dtype=torch.long, device=device)
                y_tuple = (classes * config_stage1.cond_y,)
            elif config_stage1.classifier_path is not None:
                classes = torch.full((batch_size,), labels[0], device=device)
                y_tuple = (classes,)

        # 用于分类器引导的类标签：guid_y
        if config.use_pred_y:
            model_kwargs["class_ids"] = None
        else:
            class_ids = torch.tensor(labels, dtype=torch.int).unsqueeze(0)  # labels 转换为张量，并在最外层扩展一维
            class_ids = class_ids.repeat(batch_size, 1).to(device)  # 将第0维重复为Batch
            model_kwargs["class_ids"] = class_ids
            logging.info(f"class_ids shape: {class_ids.shape}")

        all_metric_paths = [
            os.path.join(outpath, i + ".last")
            for i in (list(METRICS.keys()) + ["final_loss"])
        ]
        if config_stage1.get("resume", False) and all_exist(all_metric_paths):
            for metric_name, metric in METRICS.items():
                metric.dataset_scores += torch.load(
                    os.path.join(outpath, metric_name + ".last")
                )
            logging_info("Results exists. Skip!")
        else: # 生成多样的随机中间生成空间，选择语义一致和视觉合理的中间生成作为确定性采样的初始条件
            # sample images
            samples = []
            for n, y in enumerate(y_tuple):
                model_kwargs["y"] = y.view(-1) # 转换为1维张量，-1表示根据原始维度自动推断这个维度的大小；用于U-Net，即 Conditional Diffusion Model
                logging_info(f"cond_y: {y.view(-1).tolist()}, cond_y shape:{y.view(-1).shape}")
                if config.use_pred_y:
                    model_kwargs["class_ids"] = y
                logging.info(f"guid_y: {model_kwargs['class_ids'].tolist()}, guid_y shape:{model_kwargs['class_ids'].shape}")  # f""里面只能用''单引号

                timer.start()
                result = sampler.p_sample_loop(
                    model_fn,
                    shape=shape,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                    device=device,
                    progress=True,
                    return_all=True,
                    conf=config_stage1,
                    # sample_dir=outpath if config_stage1["debug"] else None,
                    sample_dir=outpath,
                )
                timer.end()

                for metric in METRICS.values():
                    metric.update(result["sample"], batch["image"])

                if "loss" in result.keys() and result["loss"] is not None:
                    recorder_stage1.add_with_logging(
                        key=f"loss_{image_name}_{n}", value=result["loss"]
                    )
                    final_loss.append(result["loss"])
                else:
                    final_loss.append(None)

                inpainted = normalize_image(result["sample"])  # sample 在这里从[-1,1] normalize 到 [0,1]
                # logging_info(f"inpainted shape: {inpainted.shape}")
                samples.append(inpainted.detach().cpu())
            samples = torch.cat(samples)

            # save images
            # save gt images
            mask_merged = masks.prod(dim=1, keepdim=True).clamp(0.0, 1.0)
            save_grid(
                normalize_image(batch["image"]),
                os.path.join(outpath, f"gt.png")
            )
            save_grid(
                normalize_image(batch["image"] * mask_merged.to(device)),
                os.path.join(outpath, f"gt-masked.png"),
            )
            # save generations
            for sample in samples:
                # print(f"sample shape: {sample.shape}")
                save_image(
                    sample,
                    os.path.join(sample_dir, f"gen-{status}-{base_count:05}.png")
                )
                base_count += 1
            save_grid(
                samples,
                os.path.join(outpath, f"gen-{status}-{grid_count:04}.png"),
                nrow=batch_size,
            )

            # save metrics
            for metric_name, metric in METRICS.items():
                torch.save(metric.dataset_scores[-config_stage1.n_iter:], os.path.join(outpath, metric_name + ".last"))

            torch.save(
                final_loss[-config_stage1.n_iter:], os.path.join(outpath, "final_loss.last"))

            num_image += 1
            last_duration = timer.get_last_duration()
            logging_info(
                "It takes %.3lf seconds for image %s"
                % (float(last_duration), image_name)
            )

            if config_stage1.debug and config.use_skip_x0:  # 多步跳采样到x0
                mid_dir = os.path.join(outpath, "middles")
                mid_datas = []
                for f in os.listdir(mid_dir):
                    if f.startswith("mid") and not "next" in f:
                        mid_path = os.path.join(mid_dir, f)
                        t = f.split("-")[1].split(".")[0]
                        print(mid_path, t)
                        t = torch.tensor([int(t)] * batch_size, device=device)
                        x_t = normalize(Image.open(mid_path).convert("RGB")).to(device).repeat(batch_size, 1, 1, 1)
                        # print(x_t.shape)
                        if t[0].item() > 150:
                            mid_datas.append((x_t, t))
                for mid_data in tqdm(mid_datas):
                    x_t, t = mid_data
                    skip_x0 = sampler.multi_step_pred_x0(
                        model_fn,
                        x=x_t,
                        t=t,
                        model_kwargs=model_kwargs,
                        cond_fn=cond_fn,
                    )
                    save_grid(
                        normalize_image(skip_x0),
                        os.path.join(mid_dir, f"{t[0].item()}-skip_x0.png"),
                    )

            # save images
            # save gt images
            mask_merged = masks.prod(dim=1, keepdim=True).clamp(0.0, 1.0)
            save_grid(
                normalize_image(batch["image"]),
                os.path.join(outpath, f"gt.png")
            )
            save_grid(
                normalize_image(batch["image"] * mask_merged.to(device)),
                os.path.join(outpath, f"gt-masked.png"),
            )
            # save generations
            for sample in samples:
                # print(f"sample shape: {sample.shape}")
                save_image(
                    sample,
                    os.path.join(sample_dir, f"gen-{status}-{base_count:05}.png")
                )
                base_count += 1
            save_grid(
                samples,
                os.path.join(outpath, f"gen-{status}-{grid_count:04}.png"),
                nrow=batch_size,
            )

            # save metrics
            for metric_name, metric in METRICS.items():
                torch.save(metric.dataset_scores[-config_stage1.n_iter:], os.path.join(outpath, metric_name + ".last"))

            torch.save(
                final_loss[-config_stage1.n_iter:], os.path.join(outpath, "final_loss.last"))

            num_image += 1
            last_duration = timer.get_last_duration()
            logging_info(
                "It takes %.3lf seconds for image %s"
                % (float(last_duration), image_name)
            )


        ## report batch scores
        for metric_name, metric in METRICS.items():
            recorder_stage1.add_with_logging(
                key=f"{metric_name}_score_{image_name}",
                value=metric.report_batch(),
            )

    ## report over all results
    for metric_name, metric in METRICS.items():
        mean, colbest_mean = metric.report_all()
        recorder_stage1.add_with_logging(key=f"mean_{metric_name}", value=mean)
        recorder_stage1.add_with_logging(
            key=f"best_mean_{metric_name}", value=colbest_mean)
    if len(final_loss) > 0 and final_loss[0] is not None:
        recorder_stage1.add_with_logging(
            key="final_loss",
            value=np.mean(final_loss),
        )
    if num_image > 0:
        recorder_stage1.add_with_logging(
            key="mean time", value=timer.get_cumulative_duration() / num_image
        )

    logging_info(
        f"Samples are ready and waiting for you here: \n{outdir} \n"
    )
    recorder_stage1.end_recording()




if __name__ == "__main__":
    main()