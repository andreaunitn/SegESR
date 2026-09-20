import os
from PIL import Image
from transformers import PretrainedConfig
from segesr.models.controlnet import ControlNetModel
from segesr.models.unet_2d_condition import UNet2DConditionModel

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision=None):
    """
    Dynamically loads the appropriate text encoder model class
    based on the pretrained model's configuration.
    """

    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported as a text encoder.")

def register_checkpoint_hooks(accelerator):
    """
    Registers custom Accelerate save and load state hooks so that UNet and ControlNet
    are cleanly serialized into diffusers-compatible subdirectories ('unet/' and 'controlnet/').
    """

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 2 and len(weights) == 2, f"Expected 2 models and weights, got {len(models)} and {len(weights)}"

        for model in models:
            sub_dir = "unet" if isinstance(model, UNet2DConditionModel) else "controlnet"
            model.save_pretrained(os.path.join(output_dir, sub_dir))
            weights.pop()

    def load_model_hook(models, input_dir):
        assert len(models) == 2, f"Expected 2 models to load, got {len(models)}"

        for _ in range(len(models)):
            model = models.pop()

            if not isinstance(model, UNet2DConditionModel):
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
            else:
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")

            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

def save_model_card(repo_id, image_logs=None, base_model="", repo_folder=""):
    """
    Generates a README.md model card for the Hugging Face Hub repository.
    """

    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"

        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images

            w, h = images[0].size
            grid = Image.new("RGB", size=(len(images) * w, h))
            for idx, img in enumerate(images):
                grid.paste(img, box=(idx * w, 0))

            grid.save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i}](./images_{i}.png)\n"

    yaml = f"""---
            license: creativeml-openrail-m
            base_model: {base_model}
            tags:
            - stable-diffusion
            - stable-diffusion-diffusers
            - text-to-image
            - diffusers
            - controlnet
            - segesr
            inference: true
            ---
            """

    model_card = f"""
                # controlnet-{repo_id}

                These are SegESR ControlNet weights trained on top of {base_model}.
                {img_str}
                """

    os.makedirs(repo_folder, exist_ok=True)
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)