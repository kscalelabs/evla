"""
test.py

"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
import draccus
import torch
import torch.distributed as dist
import yaml
from PIL import Image
from prismatic.conf import VLAConfig, VLARegistry
from prismatic.models import load, load_vla
from prismatic.overwatch import initialize_overwatch
from prismatic.util import set_global_seed
from prismatic.vla import get_vla_dataset_and_collator
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from transformers import AutoProcessor
import numpy as np
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig (`prismatic/conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS.vla_id)
    )

    # Directory Paths
    data_root_dir: Path = Path(                                     # Path to Open-X dataset directory
        "datasets/open-x-embodiment"
    )
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints

    # Resume Run Parameters
    pretrained_checkpoint: str = "/nfs/scratch/pawel/inference_vla/26082024_qwen_multi/checkpoints/latest-checkpoint.pt"                    # Absolute Path to Checkpoint

    is_resume: bool = True                                          # Whether we are continuing a prior training run
                                                                    #   (only applicable given pretrained checkpoint)
    resume_step: Optional[int] = None                               # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None                              # Epoch to Resume (should match checkpoint)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    save_interval: int = 10000                                       # Interval for saving checkpoints (in steps)
    image_aug: bool = False                                         # Whether to enable image augmentations
    seed: int = 7                                                   # Random seed (for reproducibility)
    
    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)

    def __post_init__(self) -> None:
        """Lift optimization parameters from `self.vla` for ease of use =>> validate on `expected_world_size`"""
        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy

        # [Validate] Assert on `expected_world_size`
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

    # fmt: on


@draccus.wrap()
def test(cfg: TrainConfig) -> None:
    torch.cuda.set_device(device_id := 0)
    torch.cuda.empty_cache()

    # Configure Unique Run Name & Save Directory
    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        cfg.run_id += "--image_aug"

    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)

    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)

    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    vlm = load_vla(
        cfg.pretrained_checkpoint, hf_token=hf_token, load_for_training=True
    ).to("cuda:0")

    # [Validate] Model should be in Full Precision!
    for param in vlm.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    stage = "vla-train"  # Frozen vision encoder

    # Get VLA Dataset & Collator
    overwatch.info(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")
    vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=vlm.vision_backbone.get_image_transform(),
        tokenizer=vlm.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,
        default_image_resolution=vlm.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        image_aug=cfg.image_aug
    )

    # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
    dataloader = DataLoader(
        vla_dataset,
        batch_size=1,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
        worker_init_fn=worker_init_fn,
    )

    num_patches = 256
    mixed_precision_dtype = torch.bfloat16
    enable_mixed_precision_training = True
    
    # we can copy from batch dataload-  we have to verify its the same data
    # original version
    for batch in dataloader:
        with torch.autocast(
            "cuda", dtype=mixed_precision_dtype, enabled=enable_mixed_precision_training
        ):

            batch["pixel_values"]["siglip"] = batch["pixel_values"]["siglip"].to("cuda:0")
            batch["pixel_values"]["dino"] = batch["pixel_values"]["dino"].to("cuda:0")
            # [Contract] self.vlm.forward() must automatically compute `loss` and return!
            output: CausalLMOutputWithPast = vlm(
                input_ids=batch["input_ids"].to("cuda:0"),
                attention_mask=batch["attention_mask"].to("cuda:0"),
                pixel_values=batch["pixel_values"],
                labels=batch["labels"].to("cuda:0"),
            )
            action_preds = output.logits[:, num_patches : -1].argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            correct_preds = (action_preds == action_gt) & mask
            print(f"Action Accuracy original: {correct_preds.sum().float() / mask.sum().float()}")

            # first naive
            import time
            start_time = time.time()
            action = vlm.predict_action(batch["input_ids"].to("cuda:0"), batch["attention_mask"].to("cuda:0"), batch["pixel_values"], batch["labels"].to("cuda:0"))
            end_time = time.time()
            print(f"Time taken: {end_time - start_time}")
            # # second version
            # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            # INSTRUCTION = "put spoon on towel"
            # prompt = f"In: What action should the robot take to {INSTRUCTION.lower()}?\nOut:"
            # MODEL_PATH = "openvla/openvla-7b"
            # processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
            # image = Image.fromarray(np.asarray(np.random.rand(256, 256, 3) * 255, dtype=np.uint8))
            # inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

            # action = vlm.predict_action2(image, INSTRUCTION, unnorm_key="bridge_orig", do_sample=False)


if __name__ == "__main__":
    test()
