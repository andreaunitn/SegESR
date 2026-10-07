import glob
import os
from pathlib import Path
from PIL import Image
import random

import torch
from torchvision import transforms
from torch.utils import data as data

from segesr.utils.sam_utils import pad_seg_logits

class PairedCaptionDataset(data.Dataset):
    """
    Paired training dataset. Every root folder must contain, for each GT image stem:

        gt/<stem>.png            ground-truth image
        sr_bicubic/<stem>.png    bicubic-upsampled LR image (ControlNet condition)
        tag/<stem>.txt           tag prompt
        dape_embeds/<stem>.pt    DAPE image embeddings
        sam_embeds/<stem>.pt     SAM 2 image embeddings (1, 256, 64, 64)
        seg_embeds/<stem>.pt     SAM 2 mask decoder logits (N, 1, 256, 256)
        sam_embeds_gt/<stem>.pt  (optional) SAM 2 image embeddings of the GT image
        seg_embeds_gt/<stem>.pt  (optional) SAM 2 mask decoder logits of the GT image
        gt_seg/<stem>.pt         (optional) SAM 2 binary masks of the GT image

    Files are matched by stem, so the per-folder listing order does not matter.
    Use `collate_fn` as the DataLoader collate function: the number of masks varies per image.
    """

    def __init__(
            self,
            root_folders=None,
            tokenizer=None,
            null_text_ratio=0.5,
            validation=False,
    ):
        super(PairedCaptionDataset, self).__init__()

        self.null_text_ratio = null_text_ratio
        self.lr_list = []
        self.gt_list = []
        self.tag_path_list = []
        self.sam_img_embeds_list = []
        self.sam_seg_embeds_list = []
        self.dape_img_embeds_list = []
        self.gt_seg_list = []
        self.sam_img_embeds_gt_list = []
        self.sam_seg_embeds_gt_list = []

        self.validation = validation

        if self.validation:
            self.val_list = []

        if isinstance(root_folders, str):
            root_folders = [folder for folder in root_folders.split(',') if folder]

        for root_folder in root_folders:
            gt_paths = sorted(glob.glob(os.path.join(root_folder, 'gt', '*.png')))
            gt_seg_dir = os.path.join(root_folder, 'gt_seg')
            has_gt_seg = os.path.isdir(gt_seg_dir)
            has_gt_sam = all(os.path.isdir(os.path.join(root_folder, d)) for d in ('sam_embeds_gt', 'seg_embeds_gt'))

            for gt_path in gt_paths:
                stem = Path(gt_path).stem
                self.gt_list.append(gt_path)
                self.lr_list.append(self._require(root_folder, 'sr_bicubic', stem, '.png'))
                self.tag_path_list.append(self._require(root_folder, 'tag', stem, '.txt'))
                self.dape_img_embeds_list.append(self._require(root_folder, 'dape_embeds', stem, '.pt'))
                self.sam_img_embeds_list.append(self._require(root_folder, 'sam_embeds', stem, '.pt'))
                self.sam_seg_embeds_list.append(self._require(root_folder, 'seg_embeds', stem, '.pt'))

                if has_gt_seg:
                    self.gt_seg_list.append(self._require(root_folder, 'gt_seg', stem, '.pt'))

                if has_gt_sam:
                    self.sam_img_embeds_gt_list.append(self._require(root_folder, 'sam_embeds_gt', stem, '.pt'))
                    self.sam_seg_embeds_gt_list.append(self._require(root_folder, 'seg_embeds_gt', stem, '.pt'))

            if self.validation:
                self.val_list += sorted(glob.glob(os.path.join(root_folder, 'validation', 'HR', 'val', '*.png')))

        if len(self.gt_list) == 0:
            raise ValueError(f"No GT images found in {[os.path.join(f, 'gt') for f in root_folders]}")

        if self.gt_seg_list and len(self.gt_seg_list) != len(self.gt_list):
            raise ValueError("'gt_seg' exists only for some of the root folders. Provide it for all of them or for none.")

        if self.sam_seg_embeds_gt_list and len(self.sam_seg_embeds_gt_list) != len(self.gt_list):
            raise ValueError("'sam_embeds_gt'/'seg_embeds_gt' exist only for some of the root folders. Provide them for all of them or for none.")

        self.img_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])

        ram_mean = [0.485, 0.456, 0.406]
        ram_std = [0.229, 0.224, 0.225]
        self.ram_normalize = transforms.Normalize(mean=ram_mean, std=ram_std)

        self.tokenizer = tokenizer

    @property
    def has_gt_sam(self):
        """Whether the SAM 2 conditions of the GT images are available."""
        return len(self.sam_seg_embeds_gt_list) > 0

    @staticmethod
    def _require(root_folder, sub_dir, stem, ext):
        path = os.path.join(root_folder, sub_dir, stem + ext)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing '{path}' (expected one '{sub_dir}/*{ext}' file for every GT image).")
        return path

    def tokenize_caption(self, caption=""):
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    def __getitem__(self, index):

        gt_path = self.gt_list[index]
        gt_img = Image.open(gt_path).convert('RGB')
        gt_img = self.img_preproc(gt_img)

        lq_path = self.lr_list[index]
        lq_img = Image.open(lq_path).convert('RGB')
        lq_img = self.img_preproc(lq_img)

        if self.validation:
            val_path = self.val_list[index]
            val_img = Image.open(val_path).convert('RGB')
            val_img = self.img_preproc(val_img)

        if random.random() < self.null_text_ratio:
            tag = ''
        else:
            with open(self.tag_path_list[index], 'r') as file:
                tag = file.read()

        example = dict()
        example["conditioning_pixel_values"] = lq_img.squeeze(0)
        example["pixel_values"] = gt_img.squeeze(0) * 2.0 - 1.0
        example["input_ids"] = self.tokenize_caption(caption=tag).squeeze(0)

        example["ram_values"] = torch.load(self.dape_img_embeds_list[index], map_location="cpu").squeeze(0)
        example["sam_img_embeds"] = torch.load(self.sam_img_embeds_list[index], map_location="cpu").squeeze(0)
        example["sam_seg_embeds"] = torch.load(self.sam_seg_embeds_list[index], map_location="cpu").squeeze(1)

        if self.has_gt_sam:
            example["sam_img_embeds_gt"] = torch.load(self.sam_img_embeds_gt_list[index], map_location="cpu").squeeze(0)
            example["sam_seg_embeds_gt"] = torch.load(self.sam_seg_embeds_gt_list[index], map_location="cpu").squeeze(1)

        if self.gt_seg_list:
            example["sam_gt_seg"] = torch.load(self.gt_seg_list[index], map_location="cpu")

        if self.validation:
            example["val_pixel_values"] = val_img.squeeze(0) * 2.0 - 1.0

        return example

    def __len__(self):
        if self.validation:
            return len(self.val_list)
        else:
            return len(self.gt_list)

def collate_fn(examples):
    """Stacks a list of examples, padding the variable number of SAM 2 masks with empty (all-zero) maps."""

    batch = {}
    for key in examples[0]:
        values = [example[key] for example in examples]
        if key in ("sam_seg_embeds", "sam_seg_embeds_gt"):
            batch[key] = pad_seg_logits(values)
        elif key == "sam_gt_seg":
            batch[key] = values
        else:
            batch[key] = torch.stack(values)
    return batch
