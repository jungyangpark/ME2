import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

DEFAULT_IMAGE_TOKEN = "<start_of_image>"

@register_model("gemma_hf")
class GemmaHf(lmms):
    """
    Gemma: A multimodal model for video and text generation.
    """

    def __init__(
        self,
        pretrained: str = "google/gemma-3-12b-it",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = False,
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device

        # Load the model
        self._processor = AutoProcessor.from_pretrained(
            pretrained,
            trust_remote_code=trust_remote_code,
            device_map=self._device,
            torch_dtype=torch.bfloat16,
            use_fast=False
        )

        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            pretrained, 
            torch_dtype=torch.bfloat16,
            device_map=self._device, 
            trust_remote_code=trust_remote_code
        )
        
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)


        # Configuration
        config = AutoConfig.from_pretrained(pretrained)
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1


        self._device = self._model.device
        self.accelerator = accelerator

        # 모델 초기화 후 device map 로깅
        if hasattr(self._model, 'hf_device_map'):
            eval_logger.info(f"Model device map: {self._model.hf_device_map}")
        
        # 각 레이어별 상세 device 정보 로깅
        for name, module in self._model.named_modules():
            if hasattr(module, 'device'):
                eval_logger.info(f"Layer {name} is on device: {module.device}")

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, add_special_tokens=False) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests):
        res = []
        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]
        
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)

            task = task[0]
            split = split[0]

            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]            
            
            until = [self.tok_decode(self.eot_token_id)]

            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            
            context = contexts[0]

            # Prepare input text
            if DEFAULT_IMAGE_TOKEN not in context:
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                context = f"{image_tokens}\n{context}"

            # Apply chat template
            messages = [{"role": "user", "content": context}]
            # 이미지 처리
            inputs = self._processor(images=visuals, text=context, return_tensors="pt").to(self._device)

            # Set generation parameters
            gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            try:
                generated_tokens = self._model.generate(
                    **inputs,
                    do_sample=gen_kwargs.get("temperature", 0) > 0,
                    max_new_tokens=gen_kwargs.get("max_new_tokens", 1024),
                    temperature=gen_kwargs.get("temperature", 0),
                    top_p=gen_kwargs.get("top_p", 1.0),
                    num_beams=gen_kwargs.get("num_beams", 1),
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=self.use_cache,
                )
                if isinstance(generated_tokens, str):
                    text_outputs = generated_tokens
                else:
                    generated_tokens = generated_tokens[:, inputs["input_ids"].shape[-1]:]
                    text_outputs = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

            except Exception as e:
                eval_logger.error(f"Error in generation: {str(e)}")
                print("Error in generation: ", str(e))
                text_outputs = ""

            res.append(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)

        res = re_ords.get_original(res)

        pbar.close()
        return res

    # def generate_until(self, requests):
    #     res = []

    #     # 요청을 정렬 및 그룹화하여 배치 처리를 준비합니다.
    #     def _collate(x):
    #         toks = self.tok_encode(x[0])
    #         return -len(toks), x[0]
    #     re_ords = utils.Collator([req.args for req in requests], _collate, grouping=True)
    #     chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
    #     pbar = tqdm(total=len(chunks), disable=(self.rank != 0), desc="Model Responding")

    #     for chunk in chunks:
    #         # 각 배치에서 요청 정보를 언패킹합니다.
    #         contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
    #         task = task[0]
    #         split = split[0]
    #         gen_kwargs = all_gen_kwargs[0]  # 배치 내 요청들은 동일한 gen_kwargs를 가정

    #         # 각 요청에 대해 이미지를 로드하고 필요한 전처리를 수행합니다.
    #         batched_visuals = []
    #         for ids in doc_id:
    #             try:
    #                 img = doc_to_visual[0](self.task_dict[task][split][ids])
    #                 if isinstance(img, PIL.Image.Image):
    #                     if img.mode in ['RGBA', 'LA', 'P']:
    #                         background = PIL.Image.new('RGB', img.size, (255, 255, 255))
    #                         if img.mode == 'RGBA':
    #                             background.paste(img, mask=img.split()[3])
    #                             img = background
    #                         else:
    #                             img = img.convert('RGB')
    #                     elif img.mode != 'RGB':
    #                         img = img.convert('RGB')
    #                 elif isinstance(img, (np.ndarray, torch.Tensor)):
    #                     if isinstance(img, torch.Tensor):
    #                         img = img.cpu().numpy()
    #                     if len(img.shape) == 2:
    #                         img = np.stack([img] * 3, axis=-1)
    #                     elif len(img.shape) == 3 and img.shape[-1] == 1:
    #                         img = np.concatenate([img] * 3, axis=-1)
    #                     img = PIL.Image.fromarray(img)
    #                 batched_visuals.append(img)
    #             except Exception as e:
    #                 eval_logger.warning(f"Error processing image for doc_id {ids}: {str(e)}")
    #                 batched_visuals.append(None)

    #         # 각 요청에 대해 텍스트 입력을 준비합니다.
    #         batch_texts = []
    #         for context in contexts:
    #             if DEFAULT_IMAGE_TOKEN not in context:
    #                 context = f"{DEFAULT_IMAGE_TOKEN}\n" + context
    #             batch_texts.append(context)

    #         # 배치 입력을 한 번에 처리합니다.
    #         inputs = self._processor(images=batched_visuals, text=batch_texts, return_tensors="pt").to(self._device)

    #         # 기본 생성 파라미터 설정
    #         gen_kwargs.setdefault("max_new_tokens", 1024)
    #         gen_kwargs.setdefault("temperature", 0)
    #         gen_kwargs.setdefault("top_p", 1.0)
    #         gen_kwargs.setdefault("num_beams", 1)
    #         gen_kwargs["image_sizes"] = [img.size for img in batched_visuals]

    #         try:
    #             generated_tokens = self._model.generate(
    #                 **inputs,
    #                 do_sample=gen_kwargs["temperature"] > 0,
    #                 max_new_tokens=gen_kwargs["max_new_tokens"],
    #                 temperature=gen_kwargs["temperature"],
    #                 top_p=gen_kwargs["top_p"],
    #                 num_beams=gen_kwargs["num_beams"],
    #                 pad_token_id=self.tokenizer.eos_token_id,
    #                 eos_token_id=self.tokenizer.eos_token_id,
    #                 use_cache=self.use_cache,
    #             )
    #             # 입력 프롬프트 부분은 제거합니다.
    #             prompt_length = inputs["input_ids"].shape[1]
    #             generated_tokens = generated_tokens[:, prompt_length:]
    #         except Exception as e:
    #             eval_logger.error(f"Error in generation: {str(e)}")
    #             generated_tokens = None

    #         if generated_tokens is not None:
    #             batch_outputs = [
    #                 self.tokenizer.decode(tokens, skip_special_tokens=True)
    #                 for tokens in generated_tokens
    #             ]
    #         else:
    #             batch_outputs = ["" for _ in chunk]

    #         res.extend(batch_outputs)
    #         if self.accelerator.is_main_process:
    #             for did, output in zip(doc_id, batch_outputs):
    #                 if did % 100 == 0:
    #                     eval_logger.debug(f"Generated text for doc ID {did}:\n\n{output}\n")
    #         self.cache_hook.add_partial("generate_until", (batch_texts, gen_kwargs), batch_outputs)
    #         pbar.update(1)

    #     res = re_ords.get_original(res)
    #     pbar.close()
    #     return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for GemmaHf")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("If your eval needs loglikelihood, implement it here.")
