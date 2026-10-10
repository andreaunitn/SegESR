import argparse
import gc
import logging
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

# Make `segesr` and the vendored `ram` / `basicsr` packages importable without installation
PROJECT_ROOT = Path(__file__).resolve().parent
for _path in (PROJECT_ROOT, PROJECT_ROOT / "third_party"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from packaging import version
from tqdm.auto import tqdm
from huggingface_hub import create_repo, upload_folder

import numpy as np
import torch
import torch.nn.functional as F

import transformers
from transformers import AutoTokenizer

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers import AutoencoderKL, AutoencoderTiny, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

# Will error if the minimal version of diffusers is not installed
check_min_version("0.21.0.dev0")

from ram.models.ram_lora import ram

from segesr.dataloaders.paired_dataset import PairedCaptionDataset, collate_fn
from segesr.losses import LPIPSLoss, SamPerceptualLoss
from segesr.models.controlnet import ControlNetModel
from segesr.models.unet_2d_condition import UNet2DConditionModel
from segesr.utils import (
    decode_latents_to_rgb,
    get_diffusion_target,
    cast_frozen_params,
    get_sam_kwargs,
    get_tensorboard_writer,
    to_tensorboard_image,
    prepare_validation_conditions,
    import_model_class_from_model_name_or_path,
    init_sam_weights,
    load_sam2,
    parse_args_with_config,
    predict_original_latents,
    register_checkpoint_hooks,
    save_model_card,
    unfreeze_params,
    validation,
    verify_weights,
)

logger = get_logger(__name__)

SAM_IMAGE_ATTENTIONS = "sam2_image_attentions"
SAM_SEGMENTATION_ATTENTIONS = "sam2_segmentation_attentions"

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="SegESR training script.")

    # Models
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="preset/models/stable-diffusion-2-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--seesr_model_path",
        type=str,
        default=None,
        help="SeeSR checkpoint used to initialize the UNet (and the ControlNet) when no self-trained weights are given.",
    )
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="Self-trained checkpoint (with a 'unet/' subfolder) to resume the UNet from.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Checkpoint (with a 'controlnet/' subfolder) to load the ControlNet from. If not specified, it is initialized from the UNet.",
    )
    parser.add_argument(
        "--tiny_vae_path",
        type=str,
        default=None,
        help="Path to the pretrained tiny VAE used to decode predictions for the perceptual losses.",
    )
    parser.add_argument(
        "--ram_path",
        type=str,
        default="preset/models/ram_swin_large_14m.pth",
        help="Path to the pretrained RAM model.",
    )
    parser.add_argument(
        "--ram_ft_path",
        type=str,
        default=None,
        help="Path to the DAPE weights.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    # Output & logging
    parser.add_argument(
        "--output_dir",
        type=str,
        default="preset/train_output/segesr",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="TensorBoard log directory (default: 'tensorboard/<output_dir name>/train', next to the test logs of the run).",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='The integration to report the results and logs to: "tensorboard", "wandb", "comet_ml" or "all".',
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="SegESR",
        help="The `project_name` argument passed to Accelerator.init_trackers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training."
    )

    # Data
    parser.add_argument(
        "--root_folders",
        type=str,
        default="",
        help="Comma-separated list of training data folders (see PairedCaptionDataset for the expected layout).",
    )
    parser.add_argument(
        "--validation_data_dir",
        type=str,
        default=None,
        help="Comma-separated list of folders with the same structure as the training data, used for the validation loss.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution of the training images.",
    )
    parser.add_argument(
        "--null_text_ratio",
        type=float,
        default=0.5,
        help="Probability of replacing the tag prompt with an empty string.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )

    # Training
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store. Older ones are deleted.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Resume from a checkpoint saved by `--checkpointing_steps`, or "latest" for the last one in `output_dir`.',
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate of the pre-existing (SeeSR) modules.",
    )
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=5e-4,
        help="Initial learning rate for newly added modules (fusion convs and SAM 2 attentions). Should typically be higher."
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='One of ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--use_paged_optimizer",
        action="store_true",
        help="With --use_8bit_adam, use the paged 8-bit AdamW, whose states move to CPU memory when the GPU is full.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")

    # Precision & memory
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Whether or not to allow TF32 on Ampere GPUs.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help="Save more memory by setting grads to None instead of zero.",
    )

    # Hub
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )

    # Validation
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help="LR image(s) used to generate a validation sample. Only the first one is used.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help="Optional prompt appended to the RAM tags of the validation image.",
    )
    parser.add_argument(
        "--generate_validation_image",
        action="store_true",
        help="Whether to generate a super-resolved `--validation_image` during validation."
    )

    # Trainable modules
    parser.add_argument(
        "--train_controlnet_tag_attention",
        action="store_true",
        help="Make the ControlNet text (tag) attention modules trainable."
    )
    parser.add_argument(
        "--train_controlnet_dape_attention",
        action="store_true",
        help="Make the ControlNet DAPE image attention modules trainable."
    )
    parser.add_argument(
        "--train_controlnet_sam_image_attention",
        action="store_true",
        help="Make the ControlNet SAM 2 image embedding attention modules (SICA) trainable."
    )
    parser.add_argument(
        "--train_controlnet_sam_segmentation_attention",
        action="store_true",
        help="Make the ControlNet SAM 2 segmentation attention modules (SMCA) trainable."
    )
    parser.add_argument(
        "--train_controlnet_fusion_conv",
        action="store_true",
        help="Make the ControlNet PAFB fusion convolutions trainable."
    )
    parser.add_argument(
        "--train_unet_tag_attention",
        action="store_true",
        help="Make the UNet text (tag) attention modules trainable."
    )
    parser.add_argument(
        "--train_unet_dape_attention",
        action="store_true",
        help="Make the UNet DAPE image attention modules trainable."
    )
    parser.add_argument(
        "--train_unet_sam_image_attention",
        action="store_true",
        help="Make the UNet SAM 2 image embedding attention modules (SICA) trainable."
    )
    parser.add_argument(
        "--train_unet_sam_segmentation_attention",
        action="store_true",
        help="Make the UNet SAM 2 segmentation attention modules (SMCA) trainable."
    )
    parser.add_argument(
        "--train_unet_fusion_conv",
        action="store_true",
        help="Make the UNet PAFB fusion convolutions trainable."
    )
    parser.add_argument(
        "--init_sam_from_dape",
        action="store_true",
        help=(
            "Initialize the trained SAM 2 attention modules by copying the DAPE attention weights. Use it only when"
            " starting from SeeSR weights: it overwrites SAM 2 attentions already trained in the loaded checkpoint."
        ),
    )

    # SegESR architecture (saved in the UNet/ControlNet configs, so test.py rebuilds it from the checkpoint).
    # All off = SeeSR.
    parser.add_argument(
        "--attention_fusion",
        type=str,
        default="sequential",
        choices=["sequential", "parallel"],
        help="'sequential': text then DAPE attention (SeeSR). 'parallel': PAFB, parallel attentions fused by a conv.",
    )
    parser.add_argument(
        "--use_sam2_image_attention",
        action="store_true",
        help="Add the SAM 2 image embedding attention (SICA). Requires `--attention_fusion parallel`.",
    )
    parser.add_argument(
        "--use_sam2_segmentation_attention",
        action="store_true",
        help="Add the SAM 2 segment-token attention (SMCA). Requires `--attention_fusion parallel`.",
    )
    parser.add_argument(
        "--segment_routing",
        action="store_true",
        help="Restrict the SMCA of each latent pixel to the segments covering it. Requires `--use_sam2_segmentation_attention`.",
    )

    # SAM 2
    parser.add_argument(
        "--sam_model_size",
        type=str,
        default="large",
        choices=["tiny", "small", "base_plus", "large"],
        help="SAM 2.1 model used for the perceptual loss and the validation sample. Must match the one used to precompute the embeddings.",
    )
    parser.add_argument(
        "--clean_sam_prob",
        type=float,
        default=0.0,
        help=(
            "Maximum probability of conditioning a sample on the SAM 2 conditions of its GT image instead of its LR"
            " image. It decreases linearly with the timestep (clean_sam_prob at t=0, 0 at t=T), matching the SAM 2"
            " refresh on the predicted clean image at inference. Requires the 'sam_embeds_gt/' and 'seg_embeds_gt/' folders."
        ),
    )
    parser.add_argument(
        "--use_sam_loss",
        action="store_true",
        help="Add the SAM 2 perceptual loss to the diffusion loss."
    )
    parser.add_argument(
        "--sam_loss_weight",
        type=float,
        default=1.0,
        help="Weight of the SAM 2 perceptual loss."
    )
    parser.add_argument(
        "--use_lpips_loss",
        action="store_true",
        help="Add the LPIPS loss to the diffusion loss."
    )
    parser.add_argument(
        "--lpips_loss_weight",
        type=float,
        default=1.0,
        help="Weight of the LPIPS loss."
    )

    args = parse_args_with_config(parser, input_args)

    if not args.root_folders:
        raise ValueError("`--root_folders` must be set.")

    if args.seesr_model_path is None and args.unet_model_name_or_path is None:
        raise ValueError("Specify either `--seesr_model_path` or `--unet_model_name_or_path`.")

    uses_sam2 = args.use_sam2_image_attention or args.use_sam2_segmentation_attention
    if uses_sam2 and args.attention_fusion != "parallel":
        raise ValueError("The SAM 2 attentions require `--attention_fusion parallel`.")
    if args.segment_routing and not args.use_sam2_segmentation_attention:
        raise ValueError("`--segment_routing` requires `--use_sam2_segmentation_attention`.")
    if args.clean_sam_prob > 0 and not uses_sam2:
        raise ValueError("`--clean_sam_prob` requires a SAM 2 attention (`--use_sam2_image_attention` or `--use_sam2_segmentation_attention`).")

    if not 0.0 <= args.clean_sam_prob <= 1.0:
        raise ValueError("`--clean_sam_prob` must be in [0, 1].")

    if (args.use_sam_loss or args.use_lpips_loss) and args.tiny_vae_path is None:
        raise ValueError("`--tiny_vae_path` is required by `--use_sam_loss` and `--use_lpips_loss`.")

    if args.generate_validation_image and not args.validation_image:
        raise ValueError("`--validation_image` must be set when `--generate_validation_image` is used.")

    if args.resolution % 8 != 0:
        raise ValueError("`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder.")

    return args

def prune_checkpoints(output_dir, keep):
    """Deletes the oldest 'checkpoint-*' folders so that at most `keep` remain."""

    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

    for checkpoint in checkpoints[:max(len(checkpoints) - keep, 0)]:
        logger.info(f"Removing old checkpoint {checkpoint}")
        shutil.rmtree(os.path.join(output_dir, checkpoint))

def main(args):
    logging_dir = Path(args.logging_dir) if args.logging_dir else Path("tensorboard", Path(args.output_dir).name, "train")

    # region Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )
    # endregion

    # region Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    logging.getLogger("root").setLevel(logging.WARNING)
    # endregion

    if args.seed is not None:
        set_seed(args.seed)

    repo_id = None
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # region Loading models
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(accelerator.device)
    text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, device_map={"": str(accelerator.device)})
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    vae.enable_tiling()
    vae.enable_slicing()

    tiny_vae = None
    if args.tiny_vae_path:
        tiny_vae = AutoencoderTiny.from_pretrained(args.tiny_vae_path)
        tiny_vae.enable_tiling()
        tiny_vae.enable_slicing()

    # RAM/DAPE is only needed to tag and embed the validation image
    ram_model = None
    if args.generate_validation_image:
        ram_model = ram(pretrained=args.ram_path, pretrained_condition=args.ram_ft_path, image_size=384, vit="swin_l")
        ram_model.eval()
        ram_model.requires_grad_(False)

    # SAM 2 is needed for the perceptual loss and to condition the validation image
    sam_generator = None
    uses_sam2 = args.use_sam2_image_attention or args.use_sam2_segmentation_attention
    if args.use_sam_loss or (uses_sam2 and args.generate_validation_image):
        sam_generator = load_sam2(
            model_size=args.sam_model_size,
            device=accelerator.device,
            apply_postprocessing=False,
            points_per_side=16,
            points_per_batch=128,
            stability_score_thresh=0.9,
        )

    # The validation image is fixed: its RAM tags/embeddings and SAM 2 conditions are computed once here, so
    # the RAM model (and SAM 2, if only needed for this) does not occupy GPU memory during training
    validation_conditions = None
    if args.generate_validation_image:
        validation_conditions = prepare_validation_conditions(
            args.validation_image[0], ram_model, sam_generator if uses_sam2 else None, accelerator.device
        )
        del ram_model
        if not args.use_sam_loss:
            sam_generator = None
        gc.collect()
        torch.cuda.empty_cache()

    # Modules missing from the loaded checkpoint (e.g. the SAM 2 attentions when starting from SeeSR) keep their initialization
    architecture = dict(
        use_image_cross_attention=True,
        attention_fusion=args.attention_fusion,
        use_sam2_image_attention=args.use_sam2_image_attention,
        use_sam2_segmentation_attention=args.use_sam2_segmentation_attention,
        segment_routing=args.segment_routing,
    )
    logger.info(f"SegESR architecture: {architecture}")

    unet_path = args.unet_model_name_or_path or args.seesr_model_path
    logger.info(f"Loading UNet weights from {unet_path}")
    unet = UNet2DConditionModel.from_pretrained(
        unet_path, subfolder="unet", device_map=None, low_cpu_mem_usage=False, revision=args.revision, **architecture
    )

    if args.controlnet_model_name_or_path:
        logger.info(f"Loading ControlNet weights from {args.controlnet_model_name_or_path}")
        controlnet = ControlNetModel.from_pretrained(
            args.controlnet_model_name_or_path, subfolder="controlnet", device_map=None, low_cpu_mem_usage=False, **architecture
        )
    else:
        logger.info("Initializing ControlNet weights from the UNet")
        controlnet = ControlNetModel.from_unet(unet, use_image_cross_attention=True)

    # `accelerator.save_state(...)` serializes the models into diffusers-style 'unet/' and 'controlnet/' subfolders
    register_checkpoint_hooks(accelerator)

    text_encoder.eval()
    text_encoder.requires_grad_(False)
    vae.eval()
    vae.requires_grad_(False)
    if tiny_vae is not None:
        tiny_vae.eval()
        tiny_vae.requires_grad_(False)

    controlnet.train()
    controlnet.requires_grad_(False)
    unet.train()
    unet.requires_grad_(False)

    # Make ControlNet and UNet modules trainable
    trainable_modules = [
        (args.train_controlnet_tag_attention, controlnet, "ControlNet", "attentions", ["image", "sam2"]),
        (args.train_controlnet_dape_attention, controlnet, "ControlNet", "image_attentions", ["sam2"]),
        (args.train_controlnet_sam_image_attention, controlnet, "ControlNet", SAM_IMAGE_ATTENTIONS, None),
        (args.train_controlnet_sam_segmentation_attention, controlnet, "ControlNet", SAM_SEGMENTATION_ATTENTIONS, None),
        (args.train_controlnet_fusion_conv, controlnet, "ControlNet", "fusion_conv", None),
        (args.train_unet_tag_attention, unet, "UNet", "attentions", ["image", "sam2"]),
        (args.train_unet_dape_attention, unet, "UNet", "image_attentions", ["sam2"]),
        (args.train_unet_sam_image_attention, unet, "UNet", SAM_IMAGE_ATTENTIONS, None),
        (args.train_unet_sam_segmentation_attention, unet, "UNet", SAM_SEGMENTATION_ATTENTIONS, None),
        (args.train_unet_fusion_conv, unet, "UNet", "fusion_conv", None),
    ]

    # Modules absent from the chosen architecture (e.g. the fusion convs of SeeSR) are skipped
    for enabled, model, model_name, target_suffix, exclude_keywords in trainable_modules:
        if enabled and any(name.endswith(target_suffix) for name, _ in model.named_modules()):
            unfreeze_params(model, model_name, target_suffix, exclude_keywords=exclude_keywords)

    # SAM 2 attention modules that exist and are being trained (in either model)
    trained_sam_modules = []
    if args.use_sam2_image_attention and (args.train_controlnet_sam_image_attention or args.train_unet_sam_image_attention):
        trained_sam_modules.append(SAM_IMAGE_ATTENTIONS)
    if args.use_sam2_segmentation_attention and (args.train_controlnet_sam_segmentation_attention or args.train_unet_sam_segmentation_attention):
        trained_sam_modules.append(SAM_SEGMENTATION_ATTENTIONS)

    if args.init_sam_from_dape and trained_sam_modules:
        for model in (controlnet, unet):
            init_sam_weights(model, accelerator, trained_sam_modules)
            if verify_weights(model, accelerator, trained_sam_modules) is False:
                raise RuntimeError(f"DAPE -> SAM 2 weight initialization failed for {model.__class__.__name__}.")
    # endregion

    # region Optimizations
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing.")
        unet.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()

    # Trainable weights stay float32 (mixed precision keeps float32 master weights); the frozen ones are stored in
    # the mixed-precision dtype, in which autocast uses them anyway
    frozen_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(accelerator.mixed_precision, torch.float32)
    num_cast = cast_frozen_params([unet, controlnet], frozen_dtype)
    logger.info(f"Frozen UNet/ControlNet parameters stored in {frozen_dtype}: {num_cast:,}")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        # The paged variant moves the optimizer states to CPU memory when the GPU is full
        optimizer_class = bnb.optim.PagedAdamW8bit if args.use_paged_optimizer else bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    # endregion

    # region Optimizer
    # Newly added modules (fusion convs and trained SAM 2 attentions) use `finetune_lr`
    new_module_keywords = ["fusion_conv"] + trained_sam_modules

    new_module_params = []
    base_model_params = []
    for name, param in list(controlnet.named_parameters()) + list(unet.named_parameters()):
        if param.requires_grad:
            if any(keyword in name for keyword in new_module_keywords):
                new_module_params.append(param)
            else:
                base_model_params.append(param)

    num_new_params = sum(p.numel() for p in new_module_params)
    num_base_params = sum(p.numel() for p in base_model_params)
    logger.info(f"Trainable Base Model Parameters: {num_base_params:,} (LR: {args.learning_rate})")
    logger.info(f"Trainable New Module Parameters: {num_new_params:,} (LR: {args.finetune_lr})")
    logger.info(f"Total Trainable Parameters: {num_base_params + num_new_params:,}")

    optimizer_grouped_parameters = [
        {"params": base_model_params, "lr": args.learning_rate},
        {"params": new_module_params, "lr": args.finetune_lr},
    ]
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]

    if not optimizer_grouped_parameters:
        raise ValueError("No trainable parameters found. Check your --train_* flags.")

    optimizer = optimizer_class(
        optimizer_grouped_parameters,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    params_to_optimize = base_model_params + new_module_params
    # endregion

    # region Dataloaders
    logger.info("Creating train dataloader...")
    train_dataset = PairedCaptionDataset(
        root_folders=args.root_folders,
        tokenizer=tokenizer,
        null_text_ratio=args.null_text_ratio,
        load_sam=uses_sam2,
    )

    if args.clean_sam_prob > 0 and not train_dataset.has_gt_sam:
        raise ValueError("`--clean_sam_prob` > 0 requires the 'sam_embeds_gt/' and 'seg_embeds_gt/' training folders.")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=args.dataloader_num_workers,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Validation runs on the main process only, so its dataloader is not sharded by `accelerator.prepare`
    validation_dataloader = None
    if args.validation_data_dir:
        logger.info("Creating validation dataloader...")
        validation_dataset = PairedCaptionDataset(
            root_folders=args.validation_data_dir,
            tokenizer=tokenizer,
            null_text_ratio=0.0,
            load_sam=uses_sam2,
        )

        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            num_workers=args.dataloader_num_workers,
            batch_size=args.train_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
    # endregion

    # region Scheduler
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # endregion

    # region Prepare everything with accelerator
    controlnet, unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, unet, optimizer, train_dataloader, lr_scheduler
    )

    # Frozen models only used for inference are cast to half-precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    if tiny_vae is not None:
        tiny_vae.to(accelerator.device, dtype=weight_dtype)

    sam_loss_fn = None
    if args.use_sam_loss:
        sam_image_encoder = sam_generator.predictor.model.image_encoder
        sam_image_encoder.to(accelerator.device, dtype=weight_dtype)
        sam_loss_fn = SamPerceptualLoss(sam_image_encoder)

    lpips_loss_fn = None
    if args.use_lpips_loss:
        logger.info("Loading LPIPS loss model")
        lpips_loss_fn = LPIPSLoss(device=accelerator.device)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        # Trackers (e.g. tensorboard) only accept scalar config values
        tracker_config = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))}
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

        # The validation input (bicubic x4), to compare with the validation samples logged during training
        writer = get_tensorboard_writer(accelerator)
        if writer is not None and validation_conditions is not None:
            val_input = validation_conditions["image"]
            val_input = val_input.resize((val_input.width * 4, val_input.height * 4), Image.BICUBIC)
            writer.add_image("validation/input_bicubic", to_tensorboard_image(val_input), 0, dataformats="HWC")
    # endregion

    # region Train
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    initial_global_step = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir) if os.path.isdir(args.output_dir) else []
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    run_validation = validation_dataloader is not None or args.generate_validation_image

    # Architecture of the trained models: decides which SAM 2 inputs they receive
    model_config = accelerator.unwrap_model(unet).config

    # Values of the micro-batches of the current optimization step, logged once per step as their mean
    step_values = defaultdict(list)
    grad_norm = None
    step_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet, unet):
                # Ground truth latents
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise to be added to the latents + timestep for each image
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Conditions
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]
                controlnet_image = batch["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                ram_encoder_hidden_states = batch["ram_values"].to(accelerator.device, dtype=weight_dtype)

                # At low noise levels, sometimes condition on the SAM 2 outputs of the clean image
                use_clean_sam = None
                if args.clean_sam_prob > 0:
                    clean_prob = args.clean_sam_prob * (1.0 - timesteps.float() / noise_scheduler.config.num_train_timesteps)
                    use_clean_sam = torch.rand(bsz, device=latents.device) < clean_prob

                sam_kwargs = get_sam_kwargs(batch, accelerator.device, weight_dtype, model_config, mask_size=latents.shape[-2:],
                                            use_clean=use_clean_sam)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                    image_encoder_hidden_states=ram_encoder_hidden_states,
                    **sam_kwargs,
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    image_encoder_hidden_states=ram_encoder_hidden_states,
                    **sam_kwargs,
                ).sample

                del encoder_hidden_states, controlnet_image, ram_encoder_hidden_states, sam_kwargs
                del down_block_res_samples, mid_block_res_sample
                torch.cuda.empty_cache()

                target = get_diffusion_target(noise_scheduler, latents, noise, timesteps)
                diffusion_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = diffusion_loss
                logs = {"loss/train_diffusion": diffusion_loss.detach().item()}

                if sam_loss_fn is not None or lpips_loss_fn is not None:
                    # Decode the predicted x_0 with the (frozen, differentiable) tiny VAE so that
                    # the perceptual losses backpropagate into the UNet and the ControlNet
                    pred_x0_latents = predict_original_latents(noise_scheduler, noisy_latents, model_pred, timesteps)
                    sr_rgb = decode_latents_to_rgb(tiny_vae, pred_x0_latents, weight_dtype)
                    gt_rgb = (pixel_values.to(torch.float32) + 1.0) / 2.0

                    if sam_loss_fn is not None:
                        sam_loss = sam_loss_fn(sr_rgb, gt_rgb)
                        loss = loss + args.sam_loss_weight * sam_loss
                        logs["loss/train_sam"] = sam_loss.detach().item() * args.sam_loss_weight

                    if lpips_loss_fn is not None:
                        lpips_loss = lpips_loss_fn(sr_rgb, gt_rgb)
                        loss = loss + args.lpips_loss_weight * lpips_loss
                        logs["loss/train_lpips"] = lpips_loss.detach().item() * args.lpips_loss_weight

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grads = [torch.norm(p.grad.detach(), 2) for p in params_to_optimize if p.grad is not None]
                    if grads:
                        total_norm = torch.norm(torch.stack(grads), 2)
                        grad_norm = total_norm.item()

                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            logs["loss/train"] = loss.detach().item()
            for key, value in logs.items():
                step_values[key].append(value)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {key: float(np.mean(values)) for key, values in step_values.items()}
                step_values.clear()
                logs["lr"] = lr_scheduler.get_last_lr()[0]
                if grad_norm is not None:
                    logs["grad_norm"] = grad_norm
                logs["time/seconds_per_step"] = time.time() - step_start
                if torch.cuda.is_available():
                    logs["gpu/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            prune_checkpoints(args.output_dir, args.checkpoints_total_limit - 1)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if run_validation and global_step % args.validation_steps == 0:
                        val_logs = validation(
                            accelerator.unwrap_model(unet),
                            accelerator.unwrap_model(controlnet),
                            vae,
                            text_encoder,
                            tokenizer,
                            noise_scheduler,
                            tiny_vae,
                            validation_conditions,
                            sam_loss_fn,
                            lpips_loss_fn,
                            validation_dataloader,
                            global_step,
                            args,
                            accelerator,
                            weight_dtype
                        )

                        for key, value in val_logs.items():
                            logs[f"loss/{key}"] = value

                        if "val_loss" in val_logs:
                            logger.info(f"validation_loss: {val_logs['val_loss']:.4f}")

                progress_bar.set_postfix(**{k: f"{v:.4g}" for k, v in logs.items() if k.startswith("loss/train")})
                accelerator.log(logs, step=global_step)
                step_start = time.time()

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break
    # endregion

    # Save the trained modules in the same layout as the checkpoints ('unet/' and 'controlnet/')
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(controlnet).save_pretrained(os.path.join(args.output_dir, "controlnet"))
        accelerator.unwrap_model(unet).save_pretrained(os.path.join(args.output_dir, "unet"))

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=None,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()

if __name__ == "__main__":
    main(parse_args())
