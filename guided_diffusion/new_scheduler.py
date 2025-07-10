# implement DDIM schedulers
import torch
import numpy as np
from .respace import space_timesteps


def ddim_timesteps(
    num_inference_steps,  # ddim step num
    ddpm_num_steps=1000,  # ! notice this should be 250 for celebA model
    schedule_type="linear",
    infer_step_repace=None,
    **kwargs,
):
    if schedule_type == "linear":
        # linear timestep schedule
        step_ratio = ddpm_num_steps / num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int32)
        )
    elif schedule_type == "quad":
        timesteps = (
            (np.linspace(0, np.sqrt(ddpm_num_steps * 0.8), num_inference_steps)) ** 2
        ).astype(int)
    elif schedule_type == "respace":
        timesteps = np.array(list(space_timesteps(num_inference_steps, infer_step_repace))).astype(int)
    timesteps = timesteps.tolist()
    timesteps = sorted(list(set(timesteps)), reverse=True)  # remove duplicates
    # print(f"timesteps: {timesteps}")
    return timesteps

def repaint_step_filter(filter_type, max_T):
    if filter_type == "none":
        return lambda x: False
    elif filter_type.startswith("firstp"):
        percent = float(filter_type.split("-")[1])
        return lambda x: x < max_T * (1.0 - percent / 100.0)  # this isreverse
    elif filter_type.startswith("lastp"):
        percent = float(filter_type.split("-")[1])
        return lambda x: x > max_T * percent / 100.0  # this isreverse
    elif filter_type.startswith("firstn"):
        num = int(filter_type.split("-")[1])
        return lambda x: x < max_T - num
    elif filter_type.startswith("lastn"):
        num = int(filter_type.split("-")[1])
        return lambda x: x > num


def ddim_repaint_timesteps(
    num_inference_steps,  # ddim step num
    ddpm_num_steps=1000,  # ! notice this should be 250 for celebA model
    jump_length=10,
    jump_n_sample=10,
    jump_start_step=230,
    jump_end_step=0,
    device=None,
    time_travel_filter_type="none",
    **kwargs,
):
    num_inference_steps = min(ddpm_num_steps, num_inference_steps)
    timesteps = ddim_timesteps(num_inference_steps, ddpm_num_steps, **kwargs)
    timesteps_resample = []
    jumps = {}
    step_filter = repaint_step_filter(time_travel_filter_type, ddpm_num_steps)
    assert jump_start_step + jump_length <= num_inference_steps-1, f"jump_step out of range {num_inference_steps-1}"
    for j in range(jump_start_step, jump_end_step-1, -jump_length):
        if step_filter(j):  # don't do time travel when t is close to T
            continue
        jumps[j] = jump_n_sample

    t = num_inference_steps
    while t >= 1:
        t = t - 1
        if t in timesteps:
            timesteps_resample.append(t)
        else:
            continue
        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                if t in timesteps:
                    timesteps_resample.append(t)
                else:
                    continue
    timesteps_resample = np.array(timesteps_resample) * (ddpm_num_steps // num_inference_steps)
    timesteps_resample = timesteps_resample.tolist()
    # print(f"resample_timesteps: {timesteps_resample}")
    return timesteps_resample
