# flake8: noqa

__version__ = "0.7.8.dev0"

from .core import set_seed
from .environment import TextEnvironment, TextHistory
from .extras import BestOfNSampler
from .import_utils import (
    is_diffusers_available,
    is_npu_available,
    is_peft_available,
    is_wandb_available,
    is_xpu_available,
)
from .models import (
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
    AutoModelForDialogueActLMWithValueHead,
    PreTrainedModelWrapper,
    create_reference_model,
)
from .trainer import (
    DataCollatorForCompletionOnlyLM,
    DPOTrainer,
    IterativeSFTTrainer,
    DialogueActPPOTrainer,
    PPOConfig,
    PPOTrainer,
    CustomPPOTrainer,
    RewardConfig,
    RewardTrainer,
    SFTTrainer,
)


if is_diffusers_available():
    from .models import (
        DDPOPipelineOutput,
        DDPOSchedulerOutput,
        DDPOStableDiffusionPipeline,
        DefaultDDPOStableDiffusionPipeline,
    )
    from .trainer import DDPOConfig, DDPOTrainer