import logging
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt
from tqdm import tqdm
import torch.nn.functional as F

from utils.logger import logging_info
from .gaussian_diffusion import _extract_into_tensor
from .new_scheduler import ddim_timesteps, ddim_repaint_timesteps
from .respace import SpacedDiffusion
from torch.optim.lr_scheduler import ReduceLROnPlateau



def noise_like(shape, device, repeat=False):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(
            shape[0], *((1,) * (len(shape) - 1))
        )

    def noise():
        return torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()


class DDIMSampler(SpacedDiffusion):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )
        self.ddim_sigma = conf.get("ddim.ddim_sigma", 0.0)

    def _get_et(self, model_fn, x, t, model_kwargs):
        model_fn = self._wrap_model(model_fn)
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model_fn(x, self._scale_timesteps(t), **model_kwargs)
        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, _ = torch.split(model_output, C, dim=1)
        return model_output

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(
                self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample(
        self,
        model_fn,
        x,
        t,
        prev_t,
        model_kwargs,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        **kwargs,
    ):
        B, C = x.shape[:2]
        assert t.shape == (B,)
        with torch.no_grad():
            alpha_t = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            alpha_prev = _extract_into_tensor(
                self.alphas_cumprod, prev_t, x.shape)
            sigmas = (
                self.ddim_sigma
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt((1 - alpha_t / alpha_prev))
            )

            def process_xstart(_x):
                if denoised_fn is not None:
                    _x = denoised_fn(_x)
                if clip_denoised:
                    return _x.clamp(-1, 1)
                return _x

            e_t = self._get_et(model_fn, x, t, model_kwargs)
            pred_x0 = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=e_t))

            mean_pred = (
                pred_x0 * torch.sqrt(alpha_prev)
                + torch.sqrt(1 - alpha_prev - sigmas**2) * e_t
            )
            noise = noise_like(x.shape, x.device, repeat=False)

            nonzero_mask = (t != 0).float().view(-1, *
                                                 ([1] * (len(x.shape) - 1)))
            x_prev = mean_pred + noise * sigmas * nonzero_mask

        return {
            "x_prev": x_prev,
            "pred_x0": pred_x0,
        }

    def q_sample_middle(self, x, cur_t, tar_t, no_noise=False):
        assert cur_t <= tar_t
        device = x.device
        while cur_t < tar_t:
            if no_noise:
                noise = torch.zeros_like(x)
            else:
                noise = torch.randn_like(x)
            _cur_t = torch.tensor(cur_t, device=device)
            beta = _extract_into_tensor(self.betas, _cur_t, x.shape)
            x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * noise
            cur_t += 1
        return x

    def q_sample(self, x_start, t, no_noise=False):
        if no_noise:
            noise = torch.zeros_like(x_start)
        else:
            noise = torch.randn_like(x_start)

        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod,
                                 t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def x_forward_sample(self, x0, forward_method="from_0", no_noise=False):
        x_forward = [self.q_sample(x0, torch.tensor(0, device=x0.device))]
        if forward_method == "from_middle":
            for _step in range(0, len(self.timestep_map) - 1):
                x_forward.append(
                    self.q_sample_middle(
                        x=x_forward[-1][0].unsqueeze(0),
                        cur_t=_step,
                        tar_t=_step + 1,
                        no_noise=no_noise,
                    )
                )
        elif forward_method == "from_0":
            for _step in range(1, len(self.timestep_map)):
                x_forward.append(
                    self.q_sample(
                        x_start=x0[0].unsqueeze(0),
                        t=torch.tensor(_step, device=x0.device),
                        no_noise=no_noise,
                    )
                )
        return x_forward

    def p_sample_loop(
        self,
        model_fn,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        return_all=False,
        conf=None,
        sample_dir="",
        **kwargs,
    ):
        if device is None:
            device = next(model_fn.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(shape, device=device)

        assert conf["ddim.schedule_params"] is not None
        steps = ddim_timesteps(**conf["ddim.schedule_params"])
        time_pairs = list(zip(steps[:-1], steps[1:]))

        x0 = model_kwargs["gt"]
        x_forwards = self.x_forward_sample(x0)
        mask = model_kwargs["masks"]

        x_t = img
        import os
        from utils import normalize_image, save_grid

        for cur_t, prev_t in tqdm(time_pairs):
            # replace surrounding
            x_t = x_forwards[cur_t] * mask + (1.0 - mask) * x_t
            cur_t = torch.tensor([cur_t] * shape[0], device=device)
            prev_t = torch.tensor([prev_t] * shape[0], device=device)

            output = self.p_sample(
                model_fn,
                x=x_t,
                t=cur_t,
                prev_t=prev_t,
                model_kwargs=model_kwargs,
                conf=conf,
                pred_xstart=None,
            )
            x_t = output["x_prev"]

            if conf["debug"]:
                from utils import normalize_image, save_grid

                os.makedirs(os.path.join(sample_dir, "middles"), exist_ok=True)
                save_grid(
                    normalize_image(x_t),
                    os.path.join(sample_dir, "middles",
                                 f"mid-{prev_t[0].item()}.png"),
                )
                save_grid(
                    normalize_image(output["pred_x0"]),
                    os.path.join(sample_dir, "middles",
                                 f"pred-{prev_t[0].item()}.png"),
                )

        x_t = x_t.clamp(-1.0, 1.0)
        return {
            "sample": x_t,
        }

# implemenet
class O_DDIMSampler(DDIMSampler):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )

        # assert conf.get("optimize_xt.optimize_xt",
        #                 False), "Double check on optimize"
        self.ddpm_num_steps = conf.get(
            "ddim.schedule_params.ddpm_num_steps", 250)
        self.num_inference_steps = conf.get("ddim.schedule_params.num_inference_steps", 250)
        self.coef_xt_reg = conf.get("optimize_xt.coef_xt_reg", 0.001)
        self.coef_xt_reg_decay = conf.get("optimize_xt.coef_xt_reg_decay", 1.0)
        self.num_iteration_inp = conf.get(
            "optimize_xt.num_iteration_inp", 1
        )
        self.lr_xt = conf.get("optimize_xt.lr_xt", 0.001)
        self.lr_xt_decay = conf.get("optimize_xt.lr_xt_decay", 1.0)
        self.use_smart_lr_xt_decay = conf.get(
            "optimize_xt.use_smart_lr_xt_decay", False
        )
        self.use_adaptive_lr_xt = conf.get(
            "optimize_xt.use_adaptive_lr_xt", False)
        self.mid_interval_num = int(conf.get("optimize_xt.mid_interval_num", 1))
        if conf.get("ddim.schedule_params.use_timetravel", False):
            self.steps = ddim_repaint_timesteps(**conf["ddim.schedule_params"])
        else:
            self.steps = ddim_timesteps(**conf["ddim.schedule_params"])

        self.mode = conf.get("mode", "inpaint")
        self.scale = conf.get("scale", 0)
        self.optimize_xt = conf.get("optimize_xt.optimize_xt", True)


    def p_sample(
        self,
        model_fn,
        x,
        t,
        prev_t,
        model_kwargs,
        lr_xt,
        coef_xt_reg,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        **kwargs,
    ):
        if self.mode == "inpaint":
            def loss_fn(_x0, _pred_x0, _mask):
                ret = torch.sum((_x0 * _mask - _pred_x0 * _mask) ** 2)
                # ret = torch.sum(torch.abs(_x0 * _mask - _pred_x0 * _mask))
                return ret
        elif self.mode == "super_resolution":
            size = x.shape[-1]
            downop = nn.AdaptiveAvgPool2d(
                (size // self.scale, size // self.scale))

            def loss_fn(_x0, _pred_x0, _mask):
                down_x0 = downop(_x0)
                down_pred_x0 = downop(_pred_x0)
                ret = torch.sum((down_x0 - down_pred_x0) ** 2)
                return ret
        else:
            raise ValueError("Unkown mode: {self.mode}")

        def reg_fn(_origin_xt, _xt):
            ret = torch.sum((_origin_xt - _xt) ** 2)
            return ret

        def process_xstart(_x):
            if denoised_fn is not None:
                _x = denoised_fn(_x)
            if clip_denoised:
                return _x.clamp(-1.0, 1.0)
            return _x

        def get_et(_x, _t):
            if self.mid_interval_num > 1:
                res = grad_ckpt(
                    self._get_et, model_fn, _x, _t, model_kwargs, use_reentrant=False
                )
            else:
                res = self._get_et(model_fn, _x, _t, model_kwargs)
            return res

        def get_smart_lr_decay_rate(_t, interval_num):
            int_t = int(_t[0].item())
            interval = int_t // interval_num
            steps = (
                (np.arange(0, interval_num) * interval)
                .round()[::-1]
                .copy()
                .astype(np.int32)
            )
            steps = steps.tolist()
            if steps[0] != int_t:
                steps.insert(0, int_t)
            if steps[-1] != 0:
                steps.append(0)

            ret = 1
            time_pairs = list(zip(steps[:-1], steps[1:]))
            for i in range(len(time_pairs)):
                _cur_t, _prev_t = time_pairs[i]
                ret *= self.sqrt_recip_alphas_cumprod[_cur_t] * math.sqrt(
                    self.alphas_cumprod[_prev_t]
                )
            return ret

        def multistep_predx0(_x, _et, _t, interval_num):
            int_t = int(_t[0].item())
            interval = int_t // interval_num
            steps = (
                (np.arange(0, interval_num) * interval)
                .round()[::-1]
                .copy()
                .astype(np.int32)
            )
            steps = steps.tolist()
            if steps[0] != int_t:
                steps.insert(0, int_t)
            if steps[-1] != 0:
                steps.append(0)
            time_pairs = list(zip(steps[:-1], steps[1:]))
            x_t = _x
            for i in range(len(time_pairs)):
                _cur_t, _prev_t = time_pairs[i]
                _cur_t = torch.tensor([_cur_t] * _x.shape[0], device=_x.device)
                _prev_t = torch.tensor(
                    [_prev_t] * _x.shape[0], device=_x.device)
                if i != 0:
                    _et = get_et(x_t, _cur_t)
                x_t = grad_ckpt(
                    get_update, x_t, _cur_t, _prev_t, _et, None, use_reentrant=False
                )
            return x_t

        def get_predx0(_x, _t, _et, interval_num=1):
            if interval_num == 1:
                return process_xstart(self._predict_xstart_from_eps(_x, _t, _et))
            else:
                _pred_x0 = grad_ckpt(
                    multistep_predx0, _x, _et, _t, interval_num, use_reentrant=False
                )
                return  process_xstart(_pred_x0)

        def get_update(
            _x,
            cur_t,
            _prev_t,
            _et=None,
            _pred_x0=None,
        ):
            if _et is None:
                _et = get_et(_x=_x, _t=cur_t)
            if _pred_x0 is None:
                _pred_x0 = get_predx0(_x, cur_t, _et, interval_num=1)

            alpha_t = _extract_into_tensor(self.alphas_cumprod, cur_t, _x.shape)
            alpha_prev = _extract_into_tensor(
                self.alphas_cumprod, _prev_t, _x.shape)
            sigmas = (
                self.ddim_sigma
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt((1 - alpha_t / alpha_prev))
            )
            mean_pred = (
                _pred_x0 * torch.sqrt(alpha_prev)
                + torch.sqrt(1 - alpha_prev - sigmas**2) * _et  # dir_xt
            )
            noise = noise_like(_x.shape, _x.device, repeat=False)
            nonzero_mask = (cur_t != 0).float().view(-1,
                                                     *([1] * (len(_x.shape) - 1)))
            _x_prev = mean_pred + noise * sigmas * nonzero_mask
            return _x_prev

        B, C = x.shape[:2]
        assert t.shape == (B,)
        x0 = model_kwargs["gt"]
        masks = model_kwargs["masks"]
        mask_merged = masks.prod(dim=1, keepdim=True).clamp(0.0, 1.0)  # shape: (B, 1, H, W)

        if self.use_smart_lr_xt_decay:
            lr_xt /= get_smart_lr_decay_rate(t, self.mid_interval_num)
        logging_info(f"lr_xt_smart_decay: {lr_xt:.8f}")

        # optimize
        with torch.enable_grad():
            origin_x = x.clone().detach()
            x = x.detach().requires_grad_()
            e_t = get_et(_x=x, _t=t)
            pred_x0 = get_predx0(
                _x=x, _t=t, _et=e_t, interval_num=self.mid_interval_num
            )
            prev_loss = loss_fn(x0, pred_x0, mask_merged).item()

            for step in range(self.num_iteration_inp):
                loss = loss_fn(x0, pred_x0, mask_merged) + \
                    coef_xt_reg * reg_fn(origin_x, x)
                x_grad = torch.autograd.grad(
                    loss, x, retain_graph=False, create_graph=False
                )[0].detach()
                new_x = x - lr_xt * x_grad
                logging_info(
                    f"optimize_step: {step}, loss: {loss.item():.3f}, "  # python中f"{}"没有lf，只有f
                    f"grad norm: {torch.norm(x_grad, p=2).item():.3f}"  # torch.norm()计算张量的范数，p=2表示2范数
                )
                while self.use_adaptive_lr_xt and True:
                    with torch.no_grad():
                        e_t = get_et(new_x, _t=t)
                        pred_x0 = get_predx0(
                            new_x, _t=t, _et=e_t, interval_num=self.mid_interval_num
                        )
                        new_loss = loss_fn(x0, pred_x0, mask_merged) + coef_xt_reg * reg_fn(
                            origin_x, new_x
                        )
                        if not torch.isnan(new_loss) and new_loss <= loss:
                            break
                        else:
                            lr_xt *= 0.5
                            logging_info(
                                "Loss too large (%.3lf->%.3lf)! Learning rate decreased to %.5lf."
                                % (loss.item(), new_loss.item(), lr_xt)
                            )
                            del new_x, e_t, pred_x0, new_loss
                            new_x = x - lr_xt * x_grad

                x = new_x.detach().requires_grad_()
                e_t = get_et(x, _t=t)
                pred_x0 = get_predx0(
                    x, _t=t, _et=e_t, interval_num=self.mid_interval_num
                )
                del loss, x_grad
                torch.cuda.empty_cache()

        # # after optimize
        with torch.no_grad():

            new_loss = loss_fn(x0, pred_x0, mask_merged).item()
            logging_info("Loss Change: %.3lf -> %.3lf" % (prev_loss, new_loss))
            new_reg = reg_fn(origin_x, x).item()
            logging_info("Regularization Change: %.3lf -> %.3lf" % (0, new_reg))
            del origin_x, prev_loss, mask_merged

            pred_x0, e_t, x = pred_x0.detach(), e_t.detach(), x.detach()

            x_prev = get_update(
                x,
                t,
                prev_t,
                e_t,
                _pred_x0=pred_x0 if self.mid_interval_num == 1 else None,
            )

        return {"x": x, "x_prev": x_prev, "pred_x0": pred_x0, "loss": new_loss}

    def p_sample_loop(
        self,
        model_fn,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        return_all=False,
        conf=None,
        sample_dir="",
        **kwargs, # **将字典解包，将键值对k:v 解包为关键字参数(keyword arguments(缩写kwargs)), k = v 传入函数，是独立的参数
    ):
        if device is None:
            device = next(model_fn.parameters()).device
        assert isinstance(shape, (tuple, list))

        if noise is not None:
            assert not conf["optimize_xt.filter_xT"]
            img = noise
        else:
            xT_shape = (
                shape
                if not conf["optimize_xt.filter_xT"]
                else tuple([20] + list(shape[1:]))
                #shape[]为元组（C，H，W），list()将元组转化为列表，[20]+list 列表合并，tuple()将列表转换为元组(20,C,H,W)
            )
            img = torch.randn(xT_shape, device=device)

        if conf["optimize_xt.filter_xT"]:
            xT_losses = []
            for img_i in img:
                xT_losses.append(
                    self.p_sample(
                        model_fn,
                        x=img_i.unsqueeze(0),
                        t=torch.tensor([self.steps[0]] * 1, device=device),
                        prev_t=torch.tensor([0] * 1, device=device),
                        model_kwargs=model_kwargs,
                        pred_xstart=None,
                        lr_xt=self.lr_xt,
                        coef_xt_reg=self.coef_xt_reg,
                        coef_guid=self.coef_guid,
                        cond_fn= cond_fn,
                    )["loss"]
                )
            """从20张(701行tuple中参数)噪声中选择损失最小的batch_size(shape[0])张"""
            img = img[torch.argsort(torch.tensor(xT_losses))[: shape[0]]]

        time_pairs = list(zip(self.steps[:-1], self.steps[1:]))

        x_t = img
        # set up hyper paramer for this run
        loss = None

        status = None
        for cur_t, prev_t in tqdm(time_pairs):
            lr_xt = self.lr_xt * self.lr_xt_decay ** (self.num_inference_steps - 1 - cur_t)
            coef_xt_reg = self.coef_xt_reg * self.coef_xt_reg_decay ** (self.num_inference_steps - 1 - cur_t)
            logging_info(f"cur_t: {cur_t}, next_t: {prev_t}, lr_xt: {lr_xt}, coef_xt_reg: {coef_xt_reg}")
            if cur_t > prev_t:  # denoise
                status = "reverse"
                cur_t = torch.tensor([cur_t] * shape[0], device=device)
                prev_t = torch.tensor([prev_t] * shape[0], device=device)
                output = self.p_sample(
                    model_fn,
                    x=x_t,
                    t=cur_t,
                    prev_t=prev_t,
                    model_kwargs=model_kwargs,
                    pred_xstart=None,
                    lr_xt=lr_xt,
                    coef_xt_reg=coef_xt_reg,
                    cond_fn= cond_fn,
                )
                x_t = output["x_prev"]
                loss = output["loss"]

                if conf["debug"]:
                    from utils import normalize_image, save_grid

                    os.makedirs(os.path.join(
                        sample_dir, "middles"), exist_ok=True)
                    save_grid(
                        normalize_image(output["x"]),  # 保存当前步优化后的 x_t
                        os.path.join(
                            sample_dir, "middles", f"x_{cur_t[0].item()}.png"
                        ),
                    )
                    save_grid(
                        normalize_image(output["pred_x0"]),  # 根据当前步优化后x_t预测的x0
                        os.path.join(
                            sample_dir, "middles", f"pred_x0-{cur_t[0].item()}.png"
                        ),
                    )

            else:  # time travel back
                if status == "reverse" and conf.get(
                    "optimize_xt.optimize_before_time_travel", False
                ):
                    # update xt if previous status is reverse
                    x_t = self.get_updated_xt(
                        model_fn,
                        x=x_t,
                        t=torch.tensor([cur_t] * shape[0], device=device),
                        model_kwargs=model_kwargs,
                        lr_xt=lr_xt,
                        coef_xt_reg=coef_xt_reg,
                        cond_fn=cond_fn,
                    )
                status = "forward"
                # assert prev_t == cur_t + 1, "Only support 1-step time travel back"
                # prev_t = torch.tensor([prev_t] * shape[0], device=device)
                # with torch.no_grad():
                #     x_t = self._undo(x_t, prev_t)
                temp_t = cur_t + 1
                while temp_t <= prev_t:
                    temp_t = torch.tensor([temp_t] * shape[0], device=device)
                    with torch.no_grad():
                        x_t = self._undo(x_t, temp_t)
                    temp_t += 1

                # undo lr decay
                logging_info(f"Undo step: {cur_t}")
                # lr_xt /= self.lr_xt_decay
                # coef_xt_reg /= self.coef_xt_reg_decay

        x_t = x_t.clamp(-1.0, 1.0)  # normalize
        return {"sample": x_t, "loss": loss}

    def get_updated_xt(
        self,
        model_fn,
        x,
        t,
        model_kwargs,
        lr_xt,
        coef_xt_reg,
        cond_fn=None,
    ):
        return self.p_sample(
            model_fn,
            x=x,
            t=t,
            prev_t=torch.zeros_like(t, device=t.device),
            model_kwargs=model_kwargs,
            pred_xstart=None,
            lr_xt=lr_xt,
            coef_xt_reg=coef_xt_reg,
            cond_fn=cond_fn,
        )["x"]


class G_DDIMSampler(O_DDIMSampler):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )
        self.use_guidance = conf.get("use_guidance", False)
        # self.guidance_scale = conf.get("classifier_scale", 1.0)
        self.num_iteration_guid = conf.get("optimize_xt.num_iteration_guid", 1)
        self.use_local_guid = conf.get("use_local_guid", False)
        self.coef_guid = conf.get("optimize_xt.coef_guid", 0.1)
        self.coef_guid_decay = conf.get("optimize_xt.coef_guid_decay", 1.0)
        self.guid_stop_step = conf.get("optimize_xt.guid_stop_step", 0)
        self.lr_delta = conf.get("optimize_xt.lr_delta", 0.1)
        self.lr_delta_decay = conf.get("optimize_xt.lr_delta_decay", 1.0)
        self.use_comb = conf.get("optimize_xt.use_comb", False)
        self.comb_start_step = conf.get("optimize_xt.comb_start_step", 249)
        self.comb_stop_step = conf.get("optimize_xt.comb_stop_step", 100)
        self.inp_start_step = conf.get("optimize_xt.inp_start_step", 180)
        self.use_skip_x0 = conf.get("use_skip_x0", False)
        self.skip_stop_step = conf.get("skip_stop_step", 100)

    def p_sample(
            self,
            model_fn,
            x,
            t,
            prev_t,
            model_kwargs,
            lr_xt,
            coef_xt_reg,
            coef_guid,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            **kwargs,
    ):
        def loss_fn(_x0, _pred_x0, _mask):
            ret = torch.sum((_x0 * _mask - _pred_x0 * _mask) ** 2)
            return ret

        def reg_fn(_origin_xt, _xt):
            ret = torch.sum((_origin_xt - _xt) ** 2)
            return ret

        def process_xstart(_x):
            if denoised_fn is not None:
                _x = denoised_fn(_x)
            if clip_denoised:
                return _x.clamp(-1.0, 1.0)
            return _x

        def get_et(_x, _t):
            if self.mid_interval_num > 1:
                res = grad_ckpt(
                    self._get_et, model_fn, _x, _t, model_kwargs, use_reentrant=False
                )
            else:
                res = self._get_et(model_fn, _x, _t, model_kwargs)
            return res

        def get_smart_lr_decay_rate(_t, interval_num):
            int_t = int(_t[0].item())
            interval = int_t // interval_num
            steps = (
                (np.arange(0, interval_num) * interval)
                .round()[::-1]
                .copy()
                .astype(np.int32)
            )
            steps = steps.tolist()
            if steps[0] != int_t:
                steps.insert(0, int_t)
            if steps[-1] != 0:
                steps.append(0)

            ret = 1
            time_pairs = list(zip(steps[:-1], steps[1:]))
            for i in range(len(time_pairs)):
                _cur_t, _prev_t = time_pairs[i]
                ret *= self.sqrt_recip_alphas_cumprod[_cur_t] * math.sqrt(
                    self.alphas_cumprod[_prev_t]
                )
            return ret

        def get_predx0(_x, _t, _et, interval_num=1):
            if interval_num == 1:
                return process_xstart(self._predict_xstart_from_eps(_x, _t, _et))

        def get_update(
            _x,
            cur_t,
            _prev_t,
            _et=None,
            _pred_x0=None,
        ):
            if _et is None:
                _et = get_et(_x=_x, _t=cur_t)
            if _pred_x0 is None:
                _pred_x0 = get_predx0(_x, cur_t, _et, interval_num=1)

            alpha_t = _extract_into_tensor(self.alphas_cumprod, cur_t, _x.shape)
            alpha_prev = _extract_into_tensor(
                self.alphas_cumprod, _prev_t, _x.shape)
            sigmas = (
                self.ddim_sigma
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt((1 - alpha_t / alpha_prev))
            )
            mean_pred = (
                _pred_x0 * torch.sqrt(alpha_prev)
                + torch.sqrt(1 - alpha_prev - sigmas**2) * _et  # dir_xt
            )
            noise = noise_like(_x.shape, _x.device, repeat=False)
            nonzero_mask = (cur_t != 0).float().view(-1,
                                                     *([1] * (len(_x.shape) - 1)))
            _x_prev = mean_pred + noise * sigmas * nonzero_mask
            return _x_prev

        def optimize_xt_with_guid(
                _x,
                _cur_t,
                _masks,
                coef_guid,
                model_variance,
                cond_fn=None,
                model_kwargs=None,
        ):
            t_0 = torch.tensor([0] * B, device=_cur_t.device)
            with torch.enable_grad():
                _x = _x.detach().requires_grad_() # 初始化x

                for i, mask in enumerate(_masks.split(1, dim=1)):
                    temp_coef_guid = coef_guid
                    for step in range(self.num_iteration_guid):
                        e_t = get_et(_x=_x, _t=_cur_t)
                        pred_x0 = get_predx0(_x=_x, _t=_cur_t, _et=e_t, interval_num=self.mid_interval_num)

                        if self.use_local_guid:
                            pred_x0_unknown = pred_x0 * (1.0 - mask) + mask # pred_x0.detach() * mask_merged
                        else:
                            pred_x0_unknown = pred_x0

                        guid_y = model_kwargs["class_ids"][:, i]
                        log_prob = cond_fn(pred_x0_unknown, self._scale_timesteps(t_0), guid_y=guid_y)
                        loss_guid = torch.sum(0 - log_prob)
                        logging_info(f"y{i}: {guid_y.item()}, guid_step: {step}, loss_guid: {loss_guid.item():.3e}")

                        gradient = torch.autograd.grad(loss_guid, _x, retain_graph=False, create_graph=False)[0].detach()
                        assert not torch.isnan(gradient).any() and not torch.isinf(gradient).any(), "gradient has NaN/Inf!"
                        # if loss_guid.item() < 0.005:
                        #     break
                        new_x = _x - temp_coef_guid * model_variance * gradient
                        while True:
                            with torch.no_grad():
                                new_e_t = get_et(_x=new_x, _t=_cur_t)
                                new_pred_x0 = get_predx0(_x=new_x, _t=_cur_t, _et=new_e_t, interval_num=self.mid_interval_num)
                                if self.use_local_guid:
                                    new_pred_x0_unknown = new_pred_x0 * (1.0 - mask) + mask # new_pred_x0.detach() * mask_merged
                                else:
                                    new_pred_x0_unknown = new_pred_x0

                                new_log_prob = cond_fn(new_pred_x0_unknown, self._scale_timesteps(t_0), guid_y=guid_y)
                                new_loss_guid = torch.sum(0 - new_log_prob)
                                if new_loss_guid <= loss_guid:
                                    # del new_e_t, new_pred_x0, new_pred_x0_unknown, new_log_prob, new_loss_guid # python 没有块级作用域（由if，for，while等创建的变量），只有函数作用域或全局作用域
                                    break
                                else:
                                    temp_coef_guid *= 0.8
                                    logging_info(
                                        f"new_loss_guid {new_loss_guid.item():.3e} too large, coef_guid decreased to {temp_coef_guid:.5f}"
                                    )
                                    new_x = _x - temp_coef_guid * model_variance * gradient
                        # print(f"new_loss_guid:{new_loss_guid:.5f}")
                        _x = new_x.detach().requires_grad_() # 更新x
                        del gradient, new_x, pred_x0_unknown, log_prob, loss_guid, e_t, pred_x0 # 删除带有计算图的变量，节省内存
            return _x.detach()

        def optimize_xt_with_inp(
                _x0,
                _x,
                _cur_t,
                _mask,
                lr_xt,
                coef_xt_reg,
        ):
            with torch.enable_grad():
                origin_x = _x.clone().detach()
                _x = _x.detach().requires_grad_()

                for step in range(self.num_iteration_inp):
                    e_t = get_et(_x=_x, _t=_cur_t)
                    pred_x0 = get_predx0(
                        _x=_x, _t=_cur_t, _et=e_t, interval_num=self.mid_interval_num
                    )

                    loss = loss_fn(_x0, pred_x0, _mask) + \
                           coef_xt_reg * reg_fn(origin_x, _x)
                    x_grad = torch.autograd.grad(
                        loss, _x, retain_graph=False, create_graph=False
                    )[0].detach()

                    new_x = _x - lr_xt * x_grad

                    logging_info(
                        f"optimize_step: {step}, loss: {loss.item():.3f}, "  # python中f"{}"没有lf，只有f
                        f"grad norm: {torch.norm(x_grad, p=2).item():.3f}"  # torch.norm()计算张量的范数，p=2表示2范数
                    )
                    while self.use_adaptive_lr_xt and True:
                        with torch.no_grad():
                            e_t = get_et(new_x, _t=_cur_t)
                            pred_x0 = get_predx0(
                                new_x, _t=_cur_t, _et=e_t, interval_num=self.mid_interval_num
                            )
                            new_loss = loss_fn(_x0, pred_x0, _mask) + coef_xt_reg * reg_fn(
                                origin_x, new_x
                            )
                            if not torch.isnan(new_loss) and new_loss <= loss:
                                break
                            else:
                                lr_xt *= 0.8
                                logging_info(
                                    "Loss too large (%.3lf->%.3lf)! Learning rate decreased to %.5lf."
                                    % (loss.item(), new_loss.item(), lr_xt)
                                )
                                del new_x, e_t, pred_x0, new_loss
                                new_x = _x - lr_xt * x_grad

                    _x = new_x.detach().requires_grad_()
                    del loss, x_grad, e_t, pred_x0
                    torch.cuda.empty_cache()
            return _x.detach()

        def optimize_predx0_with_guid(
                _x,
                _t,
                _masks,
                coef_guid,
                model_variance,
                cond_fn=None,
                model_kwargs=None,
        ):
            _x = _x.detach()
            t_0 = torch.tensor([0] * B, device=_t.device)
            e_t = get_et(_x=_x, _t=_t)
            pred_x0 = get_predx0(_x=_x, _t=_t, _et=e_t, interval_num=1)
            pred_x0 = pred_x0.detach()

            for i, mask in enumerate(_masks.split(1, dim=1)):
                with torch.enable_grad():
                    delta = torch.zeros_like(_x, requires_grad=True)  # (B, C, H, W)
                    temp_coef_guid = coef_guid
                    for step in range(self.num_iteration_guid):
                        if self.use_local_guid:
                            pred_x0_unknown = (pred_x0 + delta) * (1.0 - mask) + mask # pred_x0.detach() * mask_merged
                        else:
                            pred_x0_unknown = pred_x0 + delta

                        guid_y = model_kwargs["class_ids"][:, i]
                        log_prob = cond_fn(pred_x0_unknown, self._scale_timesteps(t_0), guid_y=guid_y)
                        loss_guid = torch.sum(0 - log_prob)
                        logging_info(f"y{i}: {guid_y.item()}, guid_step: {step}, loss_guid: {loss_guid.item():.3e}")

                        grad = torch.autograd.grad(loss_guid, delta, retain_graph=False, create_graph=False)[0].detach()
                        assert not torch.isnan(grad).any() and not torch.isinf(grad).any(), "gradient has NaN/Inf!"

                        new_delta = delta - coef_guid * model_variance * grad
                        while True:
                            with torch.no_grad():
                                new_pred_x0 = pred_x0 + new_delta
                                if self.use_local_guid:
                                    new_pred_x0_unknown = new_pred_x0 * (1.0 - mask) + mask   # new_pred_x0.detach() * mask_merged
                                else:
                                    new_pred_x0_unknown = new_pred_x0

                                new_log_prob = cond_fn(new_pred_x0_unknown, self._scale_timesteps(t_0), guid_y=guid_y)
                                new_loss_guid = torch.sum(0 - new_log_prob)
                                if new_loss_guid <= loss_guid:
                                    break
                                else:
                                    temp_coef_guid *= 0.8
                                    logging_info(
                                        f"new_loss_guid {new_loss_guid.item():.3e} too large, coef_guid decreased to {temp_coef_guid:.5f}"
                                    )
                                    new_delta = delta - temp_coef_guid * model_variance * grad
                        delta = new_delta.detach().requires_grad_()
                        del loss_guid, log_prob, pred_x0_unknown
                    pred_x0 += delta.detach() # 更新pred_x0

            return pred_x0.detach()

        def get_xt_from_x0(_x0, _t):
            alpha_t = _extract_into_tensor(self.alphas_cumprod, _t, _x0.shape)
            x_t = torch.sqrt(alpha_t) * _x0 + torch.sqrt(1.0 - alpha_t) * torch.randn(_x0.shape, device=_x0.device)
            return x_t.detach()


        with torch.no_grad(): # 节省显存，防止爆显存
            B, C = x.shape[:2]
            assert t.shape == (B,)
            # t_0 = torch.tensor([0] * B, device=t.device)
            x0 = model_kwargs["gt"]
            masks = model_kwargs["masks"]
            mask_merged = masks.prod(dim=1, keepdim=True)  # shape: (B, 1, H, W)

            x = x.detach()
            origin_x = x.clone().detach()
            e_t = get_et(_x=x, _t=t)
            pred_x0 = get_predx0(_x=x, _t=t, _et=e_t, interval_num=self.mid_interval_num)
            before_guid_loss_inp = loss_fn(x0, pred_x0, mask_merged).item()
            logging_info(f"before_guid_loss_inp: {before_guid_loss_inp:.3f}")

        if self.use_smart_lr_xt_decay:
            lr_xt *= math.sqrt(self.alphas_cumprod[t[0].item()])
            # lr_xt *= self.alphas_cumprod[t[0].item()]
            # lr_xt /= get_smart_lr_decay_rate(t, self.mid_interval_num) # 计算太麻烦，效率低

        """class condition guid"""
        if self.use_guidance and t[0].item() >= self.guid_stop_step:
            assert cond_fn is not None, f"cond_fn is None"
            model_fn = self._wrap_model(model_fn)
            assert t.shape == (B,)
            model_output = model_fn(x, self._scale_timesteps(t), **model_kwargs)
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            _, model_var_values = torch.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
            # print(torch.mean(model_variance))
            logging_info(f"classifier_guidance_scale: {coef_guid}")
            x = optimize_xt_with_guid(x, t, masks, coef_guid=coef_guid, model_variance=model_variance, cond_fn=cond_fn, model_kwargs=model_kwargs)

            del model_output, model_var_values, min_log, max_log, frac,

            with torch.no_grad():
                x = x.detach()
                e_t = get_et(_x=x, _t=t)
                pred_x0 = get_predx0(_x=x, _t=t, _et=e_t, interval_num=1)
                loss_inp = loss_fn(x0, pred_x0, mask_merged)
                # logging_info(f"after_guid_loss_inp: {loss_inp.item():.3f}")
                loss_reg = reg_fn(origin_x, x)
                logging_info(f"after_guid_loss_inp: {loss_inp.item():.3f}, after_guid_loss_reg: {loss_reg:.3e}")
                del loss_inp, loss_reg,
                torch.cuda.empty_cache()

        """optimize xt with inpainting constrain"""
        if self.optimize_xt and t[0].item() <= self.inp_start_step:
            logging_info(f"lr_xt_smart_decay: {lr_xt}")
            origin_x = x.clone().detach()  # update origin_x after guid
            x = optimize_xt_with_inp(x0, x, t, mask_merged, lr_xt, coef_xt_reg)
            e_t = get_et(x, _t=t).detach()
            pred_x0 = get_predx0(x, _t=t, _et=e_t, interval_num=self.mid_interval_num).detach()

        """Composition of known regions from GT and unknown regions from pred_x0"""
        if self.use_comb and self.comb_start_step >= t[0].item() >= self.comb_stop_step:
        # if t in [225, 200, 175, 150, 125, 100] and self.use_comb:
            comb_x0 = x0 * mask_merged + pred_x0 * (1.0 - mask_merged)  # combine masked x0 with optimized pred_x0
            x = get_xt_from_x0(comb_x0, t)
        else:
            comb_x0 = None

        with torch.no_grad():
            e_t = get_et(x, _t=t).detach()
            pred_x0 = get_predx0(x, _t=t, _et=e_t, interval_num=self.mid_interval_num).detach() # 关键，必须更新

            new_loss_inp = loss_fn(x0, pred_x0, mask_merged).item()
            logging_info(f"Loss_inp Change: %.3lf -> %.3lf" % (before_guid_loss_inp, new_loss_inp))
            new_reg = reg_fn(origin_x, x).item()
            logging_info("Reg Change: %.3lf -> %.3lf" % (0, new_reg))

            del new_reg, before_guid_loss_inp, origin_x,
            torch.cuda.empty_cache()

            x_prev = get_update(
                x,
                t,
                prev_t,
                e_t,
                _pred_x0=pred_x0 if self.mid_interval_num == 1 else None,
            )
            return {"x": x, "x_prev": x_prev, "pred_x0": pred_x0, "loss": new_loss_inp, "comb_x0": comb_x0,}

    def p_sample_loop(
            self,
            model_fn,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=True,
            return_all=False,
            conf=None,
            sample_dir="",
            **kwargs,
    ):
        if device is None:
            device = next(model_fn.parameters()).device
        assert isinstance(shape, (tuple, list))

        if noise is not None:
            assert not conf["optimize_xt.filter_xT"]
            img = noise
        else:
            xT_shape = (
                shape
                if not conf["optimize_xt.filter_xT"]
                else tuple([20] + list(shape[1:]))
                #shape[]为元组（C，H，W），list()将元组转化为列表，[20]+list 列表合并，tuple()将列表转换为元组(20,C,H,W)
            )
            img = torch.randn(xT_shape, device=device)

        if conf["optimize_xt.filter_xT"]:
            xT_losses = []
            for img_i in img:
                xT_losses.append(
                    self.p_sample(
                        model_fn,
                        x=img_i.unsqueeze(0),
                        t=torch.tensor([self.steps[0]] * 1, device=device),
                        prev_t=torch.tensor([0] * 1, device=device),
                        model_kwargs=model_kwargs,
                        pred_xstart=None,
                        lr_xt=self.lr_xt,
                        coef_xt_reg=self.coef_xt_reg,
                        coef_guid=self.coef_guid,
                        cond_fn=cond_fn,
                    )["loss"]
                )
            img = img[torch.argsort(torch.tensor(xT_losses))[: shape[0]]] # 从20张噪声中选择损失最小的B张

        logging_info(f"time_steps: {self.steps}")
        time_pairs = list(zip(self.steps[:-1], self.steps[1:]))

        # set up hyper paramer for this run
        x_t = img
        loss = None
        status = None

        for cur_t, prev_t in tqdm(time_pairs):
            lr_xt = self.lr_xt * self.lr_xt_decay ** (self.num_inference_steps-1 - cur_t)
            coef_xt_reg = self.coef_xt_reg * self.coef_xt_reg_decay ** (self.num_inference_steps-1 - cur_t)
            coef_guid = self.coef_guid * self.coef_guid_decay ** (self.num_inference_steps-1 - cur_t)
            logging_info(
                f"cur_t: {cur_t}, next_t: {prev_t}, lr_xt: {lr_xt:.8f}, coef_xt_reg: {coef_xt_reg}, coef_guid: {coef_guid}"
            )

            if cur_t > prev_t:  # denoise
                status = "reverse"
                cur_t = torch.tensor([cur_t] * shape[0], device=device)
                prev_t = torch.tensor([prev_t] * shape[0], device=device)
                output = self.p_sample(
                    model_fn,
                    x=x_t,
                    t=cur_t,
                    prev_t=prev_t,
                    model_kwargs=model_kwargs,
                    pred_xstart=None,
                    lr_xt=lr_xt,
                    coef_xt_reg=coef_xt_reg,
                    coef_guid=coef_guid,
                    cond_fn=cond_fn,
                )
                x_t = output["x_prev"]
                loss = output["loss"]

                if cur_t[0].item() >= self.skip_stop_step and self.use_skip_x0:
                    skip_x0 = self.multi_step_skip_x0(
                        model_fn,
                        x=output["x"],
                        t=cur_t,
                        model_kwargs=model_kwargs,
                        cond_fn=cond_fn,
                    )
                else:
                    skip_x0 = None

                if conf["debug"]:
                    from utils import normalize_image, save_grid

                    os.makedirs(os.path.join(sample_dir, "middles"), exist_ok=True)

                    save_grid(
                        normalize_image(output["x"]),  # 保存当前步优化后的 x_t
                        os.path.join(
                            sample_dir, "middles", f"x_{cur_t[0].item()}.png"
                        ),
                    )

                    save_grid(
                        normalize_image(output["pred_x0"]),  # 根据当前步优化后的x_t一步预测的x0
                        os.path.join(
                            sample_dir, "middles", f"pred_x0-{cur_t[0].item()}.png"
                        ),
                    )

                    if skip_x0 is not None:
                        save_grid(
                            normalize_image(skip_x0),
                            os.path.join(
                                sample_dir, "middles",f"skip_x0-{cur_t[0].item()}.png",
                            ),
                        )

                    if output["comb_x0"] is not None:
                        save_grid(
                            normalize_image(output["comb_x0"]),
                            os.path.join(
                                sample_dir, "middles", f"comb_x0-{cur_t[0].item()}.png"
                            )
                        )

            else:  # time travel back
                if status == "reverse" and conf.get(
                    "optimize_xt.optimize_before_time_travel", False
                ):
                    # update xt if previous status is reverse
                    x_t = self.get_updated_xt(
                        model_fn,
                        x=x_t,
                        t=torch.tensor([cur_t] * shape[0], device=device),
                        model_kwargs=model_kwargs,
                        lr_xt=lr_xt,
                        coef_xt_reg=coef_xt_reg,
                        coef_guid=coef_guid,
                        cond_fn=cond_fn,
                    )
                status = "forward"
                # assert prev_t == cur_t + 1, "Only support 1-step time travel back"
                temp_t = cur_t + 1
                while temp_t <= prev_t:
                    temp_t = torch.tensor([temp_t] * shape[0], device=device)
                    with torch.no_grad():
                        x_t = self._undo(x_t, temp_t)
                    temp_t += 1
                # undo lr decay
                logging_info(f"Undo step: {cur_t}")

        x_t = x_t.clamp(-1.0, 1.0)  # normalize
        return {"sample": x_t, "loss": loss}

    def get_updated_xt(
        self,
        model_fn,
        x,
        t,
        model_kwargs,
        lr_xt,
        coef_xt_reg,
        coef_guid,
        cond_fn=None
    ):
        return self.p_sample(
            model_fn,
            x=x,
            t=t,
            prev_t=torch.zeros_like(t, device=t.device),
            model_kwargs=model_kwargs,
            pred_xstart=None,
            lr_xt=lr_xt,
            coef_xt_reg=coef_xt_reg,
            coef_guid=coef_guid,
            cond_fn=cond_fn,
        )["x"]

    def multi_step_skip_x0(
        self,
        model_fn,
        x,
        t,
        model_kwargs,
        cond_fn=None
    ):
        shape, device = x.shape, x.device
        def get_et(_x, _t):
            res = self._get_et(model_fn, _x, _t, model_kwargs)
            return res
        def get_next_x(
            _x,
            _cur_t,
            _prev_t,
            _et=None,
            _pred_x0=None,
        ):
            alpha_t = _extract_into_tensor(self.alphas_cumprod, _cur_t, _x.shape)
            alpha_prev = _extract_into_tensor(
                self.alphas_cumprod, _prev_t, _x.shape)
            sigmas = (
                self.ddim_sigma
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt((1 - alpha_t / alpha_prev))
            )
            mean_pred = (
                _pred_x0 * torch.sqrt(alpha_prev)
                + torch.sqrt(1 - alpha_prev - sigmas**2) * _et  # dir_xt
            )
            noise = noise_like(_x.shape, _x.device, repeat=False)
            nonzero_mask = (_cur_t != 0).float().view(-1,*([1] * (len(_x.shape) - 1)))
            _x_prev = mean_pred + noise * sigmas * nonzero_mask
            return _x_prev
        if t[0].item() > 200:
            skip_steps = [t[0].item(), 200, 150, 100, 50, 0]
        elif t[0].item() > 150:
            skip_steps = [t[0].item(), 150, 100, 50, 0]
        elif t[0].item() > 100:
            skip_steps = [t[0].item(), 100, 50, 0]
        skip_pairs = list(zip(skip_steps[:-1], skip_steps[1:]))
        for cur_t, prev_t in tqdm(skip_pairs):
            lr_xt = self.lr_xt * self.lr_xt_decay ** (self.num_inference_steps - 1 - cur_t)
            coef_xt_reg = self.coef_xt_reg * self.coef_xt_reg_decay ** (self.num_inference_steps - 1 - cur_t)
            coef_guid = self.coef_guid * self.coef_guid_decay ** (self.num_inference_steps - 1 - cur_t)
            logging_info(
                f"cur_t: {cur_t}, next_t: {prev_t}, lr_xt: {lr_xt:.8f}, coef_xt_reg: {coef_xt_reg}, coef_guid: {coef_guid}"
            )
            cur_t = torch.tensor([cur_t] * shape[0], device=device)
            prev_t = torch.tensor([prev_t] * shape[0], device=device)
            if cur_t[0].item() == t[0].item():
                e_t = get_et(x, cur_t)
                pred_x0 = self._predict_xstart_from_eps(x, cur_t, e_t).clamp(-1.0, 1.0)
                next_x = get_next_x(x, cur_t, prev_t, e_t, pred_x0)
                x = next_x # 得到下一步的初始x
                continue
            output = super().p_sample(
                model_fn,
                x=x,
                t=cur_t,
                prev_t=prev_t,
                model_kwargs=model_kwargs,
                pred_xstart=None,
                lr_xt=lr_xt,
                coef_xt_reg=coef_xt_reg,
                cond_fn=cond_fn,
            )
            x = output["x_prev"]
        x0 = x.clamp(-1.0, 1.0)  # normalize
        return x0

