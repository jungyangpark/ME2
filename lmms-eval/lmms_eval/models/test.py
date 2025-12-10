import gemma_hf

from transformers import (
    AutoConfig,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)


config = AutoConfig.from_pretrained("google/gemma-3-12b-it")
max_frames_num = 32
model_type = getattr(config, "model_type", "gemma")
model_type = Gemma3ForConditionalGeneration
model = model_type.from_pretrained("google/gemma-3-12b-it",
                                    revision="main", torch_dtype="auto", 
                                    device_map="auto", trust_remote_code=False,
                                    attn_implementation=None)
