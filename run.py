import torch
import yaml
import argparse
import os
from datetime import datetime
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from ctrl_x.utils.sdxl import *
from ctrl_x.utils.utils import *
from ctrl_x.utils.media import *
from ctrl_x.pipelines.pipeline_animatediff import CtrlXAnimateDiffPipeline

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AnimateDiff using external YAML configuration"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML configuration file")
    parser.add_argument('--save_dir', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config, "r") as file:
        exp_config = yaml.safe_load(file)

    structure_video_path = exp_config["structure_video_path"]
    structure_prompt = exp_config["structure_prompt"]
    prompt = exp_config["prompt"]

    # * Option 1 - use an extra appearance video as input
    # appearance_video_path = exp_config["appearance_video_path"]
    # appearance_prompt = exp_config["appearance_prompt"]
    
    # * Option 2
    appearance_video_path = None
    appearance_prompt = prompt

    # According to Option 1 or 2, set appearance_video to None or load_video(path)
    structure_video = load_video(structure_video_path)
    # appearance_video = load_video(appearance_video_path)
    appearance_video = None


    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-2", 
        torch_dtype=torch.float16
    )
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    pipe = CtrlXAnimateDiffPipeline.from_pretrained(
        model_id, 
        motion_adapter=adapter, 
        torch_dtype=torch.float16
    ).to('cuda')

    scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.scheduler = scheduler

    num_inference_steps = 50
    self_recurrence_schedule = get_self_recurrence_schedule([[0.1, 0.6, 2]], num_inference_steps)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    structure_schedule = appearance_schedule = 0.5
    control_config = get_control_config(structure_schedule, appearance_schedule)

    control_params = yaml.safe_load(control_config)
    register_control(
        model=pipe,
        timesteps=timesteps,
        control_schedule=control_params["control_schedule"],
        control_target=control_params["control_target"],
    )

    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            structure_prompt=structure_prompt,
            appearance_prompt=appearance_prompt,
            positive_prompt="8k, high detailed, best quality, masterpiece",
            negative_prompt="bad quality, ugly, blurry, dark, low resolution, unrealistic",
            structure_image=structure_video,
            appearance_image=appearance_video,
            num_frames=16,
            guidance_scale=5.0,
            structure_guidance_scale=5.0,
            appearance_guidance_scale=5.0,
            guidance_rescale=1.0,
            eta=1.0,
            self_recurrence_schedule=self_recurrence_schedule,
            control_schedule=control_params["control_schedule"],
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(422),
            height=512,
            width=512,
        )

    frames = output.frames[0]

    prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.save_dir is not None:
        video_name = exp_config["structure_video_path"].split("/")[4][:-4]
        gif_path = os.path.join(args.save_dir, f"{video_name}.gif")
    else:
        gif_path = f"./results/{prefix}.gif"

    gif_dir = os.path.dirname(gif_path)
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    export_to_gif(frames, gif_path)
    print(f"Animation saved to {gif_path}")

if __name__ == "__main__":
    main()
