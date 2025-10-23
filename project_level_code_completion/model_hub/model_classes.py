import torch
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig


class ModelBuilderBase:
    @classmethod
    def build_model(cls, **kwargs):
        raise NotImplementedError


class HFModelBuilder(ModelBuilderBase):
    SEND_TO_DEVICE = True
    @classmethod
    def build_model(cls, checkpoint, **kwargs):
        kwargs = cls._update_kwargs(checkpoint, kwargs)
        device = cls._get_device()
        config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "eager"

        # ✅ Force FP16 explicitly
        config.torch_dtype = torch.float16
        kwargs['torch_dtype'] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            config=config,
            torch_dtype=torch.float16,  # hard-coded safety net
            **{k: v for k, v in kwargs.items() if k != 'torch_dtype'}  # avoid duplicate key
        )

        # ✅ Double-check all params converted
        model = model.to(dtype=torch.float16)
        if cls.SEND_TO_DEVICE:
            model = model.to(device)
        model.eval()
        # model = model.to_bettertransformer()
        print('model is ready')
        return model, device

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    @staticmethod
    def _update_kwargs(checkpoint, kwargs):
        if 'attn_implementation' not in kwargs:
            if 'starcoder' not in checkpoint:  # Quick fix for Flash-attention 2 and starcoder
                kwargs['attn_implementation'] = 'eager' #'flash_attention_2'
        if 'torch_dtype' not in kwargs:
            if torch.cuda.is_bf16_supported():
                kwargs['torch_dtype'] = torch.bfloat16
            else:
                kwargs['torch_dtype'] = torch.float16
        return kwargs


class HFModelBuilder4bit(HFModelBuilder):
    SEND_TO_DEVICE = False

    @classmethod
    def _update_kwargs(cls, checkpoint, kwargs):
        if 'attn_implementation' not in kwargs:
            if 'starcoder' not in checkpoint:  # Quick fix for Flash-attention 2 and starcoder
                kwargs['attn_implementation'] = 'flash_attention_2'
        if 'quantization_config' not in kwargs:
            kwargs['quantization_config'] = cls._get_q_config()

        return kwargs

    @staticmethod
    def _get_q_config():
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return q_config
