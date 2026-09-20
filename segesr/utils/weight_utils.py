import torch
from accelerate.logging import get_logger
from segesr.models.unet_2d_blocks import (CrossAttnDownBlock2D, CrossAttnUpBlock2D, UNetMidBlock2DCrossAttn)

logger = get_logger(__name__)

def unfreeze_params(model, model_name, target_suffix, exclude_keywords=None):
    """
    Unfreezes parameters of modules whose names end with target_suffix,
    filtering out any module names that contain any keyword in exclude_keywords.
    """

    if exclude_keywords is None:
        exclude_keywords = []

    trainable_found = False
    for name, module in model.named_modules():
        if name.endswith(target_suffix):
            if any(excluded in name for excluded in exclude_keywords):
                continue

            for param in module.parameters():
                param.requires_grad = True

            if not trainable_found:
                logger.info(f"Found trainable modules in {model_name} matching '{target_suffix}' (excluding {exclude_keywords}):")

            logger.info(f"  - {name}")
            trainable_found = True

def init_sam_weights(model, accelerator, attention_modules):
    """
    Initializes target SAM 2 attention modules (image or segmentation) by copying
    or slicing weights from the pre-trained DAPE attention module ('image_attentions').
    """

    logger.info(f"Attempting to copy DAPE weights to all SAM 2 modules for {model.__class__.__name__}...")
    model_to_copy = accelerator.unwrap_model(model)

    def transfer_weights(source_attention_module, target_attention_module, block_name, target_name):
        source_sd = source_attention_module.state_dict()
        target_sd = target_attention_module.state_dict()
        new_target_sd = {}

        for key, target_param in target_sd.items():
            if key in source_sd:
                source_param = source_sd[key]

                if source_param.shape == target_param.shape:
                    new_target_sd[key] = source_param.clone()
                elif target_param.dim() > 0 and source_param.shape[0] == target_param.shape[0]:
                    if target_param.dim() > 1: # Weight matrix
                        slice_dim = target_param.shape[1]
                        new_target_sd[key] = source_param[:, :slice_dim].clone()
                    else: # Bias vector
                        slice_dim = target_param.shape[0]
                        new_target_sd[key] = source_param[:slice_dim].clone()
                else:
                    logger.error(f"  - UNHANDLED SHAPE MISMATCH for '{key}' in {block_name}.{target_name}. "f"Source: {source_param.shape}, Target: {target_param.shape}. Keeping random init.")
                    new_target_sd[key] = target_param.clone()
            else:
                logger.warning(f"  - Key '{key}' not found in source module. Keeping random init for {block_name}.{target_name}")
                new_target_sd[key] = target_param.clone()

        target_attention_module.load_state_dict(new_target_sd)

    for block_name, block in model_to_copy.named_modules():
        is_relevant_block = isinstance(block, (CrossAttnDownBlock2D, CrossAttnUpBlock2D, UNetMidBlock2DCrossAttn))

        if is_relevant_block and hasattr(block, "use_sam") and block.use_sam:
            if not hasattr(block, "image_attentions"):
                logger.warning(f"  - Block {block_name} is SAM2 enabled but has no 'image_attentions' to copy from. Skipping.")
                continue

            source_attns = block.image_attentions
            for target_attr_name in attention_modules:
                if hasattr(block, target_attr_name):
                    target_attns = getattr(block, target_attr_name)

                    num_modules_to_copy = min(len(source_attns), len(target_attns))
                    if len(source_attns) != len(target_attns):
                        logger.warning(f"     - Mismatch in number of attention modules between source ('image_attentions') "f"and target ('{target_attr_name}'). Will copy {num_modules_to_copy} modules.")

                    for i in range(num_modules_to_copy):
                        source_module = source_attns[i]
                        target_module = target_attns[i]
                        full_target_name = f"{target_attr_name}.{i}"
                        transfer_weights(source_module, target_module, block_name, full_target_name)

    logger.info(f"Finished copying weights for {model.__class__.__name__}.")

def verify_weights(model, accelerator, attention_modules):
    """
    Verifies that target SAM 2 attention modules have been correctly
    initialized from the DAPE attention modules.
    """

    if not accelerator.is_main_process:
        return

    logger.info(f"--- STARTING WEIGHT VERIFICATION FOR {model.__class__.__name__} ---")
    verification_passed = True

    unwrapped_model = accelerator.unwrap_model(model)
    for block in unwrapped_model.named_modules():
        is_relevant_block = isinstance(block, CrossAttnDownBlock2D, CrossAttnUpBlock2D, UNetMidBlock2DCrossAttn)

        if is_relevant_block and hasattr(block, "image_attentions"):
            source_attns = block.image_attentions

            for target_attr_name in attention_modules:
                if hasattr(block, target_attr_name):
                    target_attns = getattr(block, target_attr_name)
                    num_modules_to_verify = min(len(source_attns), len(target_attns))

                    for i in range(num_modules_to_verify):
                        source_transformer = source_attns[i]
                        target_transformer = target_attns[i]
                        full_target_name = f"{target_attr_name}.{i}"

                        source_params = dict(source_transformer.named_parameters())
                        target_params = dict(target_transformer.named_parameters())

                        for param_name, target_param in target_params.items():
                            if param_name not in source_params:
                                logger.error(f"  - FAILED: Parameter '{param_name}' not found in source module "f"for '{full_target_name}'. Verification incomplete.")
                                verification_passed = False
                                continue

                            source_param = source_params[param_name].detach()
                            target_param = target_param.detach()

                            if source_param.shape != target_param.shape:
                                if target_param.dim() > 0 and source_param.shape[0] == target_param.shape[0]:
                                    if target_param.dim() > 1:  # Linear weights
                                        slice_dim = target_param.shape[1]
                                        source_slice = source_param[:, :slice_dim]
                                    else: # Bias terms
                                        slice_dim = target_param.shape[0]
                                        source_slice = source_param[:slice_dim]
                                else:
                                    logger.error(f"  - FAILED: Unhandled shape mismatch for '{param_name}' in '{full_target_name}'. "f"Source: {source_param.shape}, Target: {target_param.shape}")
                                    verification_passed = False
                                    continue

                            else:
                                source_slice = source_param

                            are_equal = torch.allclose(source_slice, target_param, atol=1e-6)
                            if not are_equal:
                                diff = torch.abs(source_slice - target_param).max().item()
                                logger.error(f"  - FAILED: Verification for '{param_name}' in '{full_target_name}'. "f"Max absolute difference: {diff:.6f}")
                                verification_passed = False


    if verification_passed:
        logger.info("--- OVERALL VERIFICATION RESULT: PASSED ---")
    else:
        logger.error("--- OVERALL VERIFICATION RESULT: FAILED ---")

    logger.info(f"--- WEIGHT VERIFICATION COMPLETE FOR {model.__class__.__name__} ---")