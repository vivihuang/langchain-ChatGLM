import json
import os

from langchain.llms.base import LLM
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from configs.model_config import *
from utils import torch_gc
from uuid import uuid1
import datetime
import time
from wudao.api_request import getToken, executeEngineV2, queryTaskResult
from dotenv import load_dotenv

load_dotenv()

DEVICE_ = LLM_DEVICE
DEVICE_ID = "0" if torch.cuda.is_available() else None
DEVICE = f"{DEVICE_}:{DEVICE_ID}" if DEVICE_ID else DEVICE_

# 接口API KEY
API_KEY = os.getenv("API_KEY")
# 公钥
PUBLIC_KEY = os.getenv("PUBLIC_KEY")


def auto_configure_device_map(num_gpus: int, use_lora: bool) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: PEFT加载lora模型出现的层命名不同
    if LLM_LORA_PATH and use_lora:
        layer_prefix = 'base_model.model.transformer'
    else:
        layer_prefix = 'transformer'

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    device_map = {f'{layer_prefix}.word_embeddings': 0,
                  f'{layer_prefix}.final_layernorm': 0, 'lm_head': 0,
                  f'base_model.model.lm_head': 0, }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'{layer_prefix}.layers.{i}'] = gpu_target
        used += 1

    return device_map


class ChatGLM(LLM):
    max_token: int = 10000
    temperature: float = 0.8
    top_p = 0.9
    history = []
    tokenizer: object = None
    model: object = None
    history_len: int = 10

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self,
              prompt: str,
              history: List[List[str]] = [],
              streaming: bool = STREAMING):  # -> Tuple[str, List[List[str]]]:
        answer = self.chat(
            prompt,
            history=history[-self.history_len:] if self.history_len > 0 else [],
            # max_length=self.max_token,
            # temperature=self.temperature,
            # top_p=self.top_p,
        )
        torch_gc()
        if answer:
            # history += [{"query": prompt, "answer": answer}]
            history += [[prompt, answer]]
        yield answer, history
        torch_gc()

    def chat(self,
             prompt: str,
             history: List[List[str]] = []) -> str:

        if not API_KEY or not PUBLIC_KEY:
            print("请检查API_KEY和PUBLIC_KEY是否存在")
            raise Exception("请检查API_KEY和PUBLIC_KEY是否存在")

        ability_type = "chatglm_qa_6b"
        # 引擎类型
        engine_type = "chatglm_6b"
        # 请求参数样例
        uuid = uuid1()
        request_task_no = str(uuid).replace("-", "")
        data = {
            "requestTaskNo": request_task_no,
            "prompt": prompt,
            "history": history
        }

        '''
          注意这里仅为了简化编码每一次请求都去获取token， 线上环境token有过期时间， 客户端可自行缓存过期后重新获取。
        '''
        token_result = getToken(API_KEY, PUBLIC_KEY)

        if token_result and token_result["code"] == 200:
            token = token_result["data"]
            resp = executeEngineV2(ability_type, engine_type, token, data)
            '''
                注意一下逻辑是为方便调试阶段获取结果样例，实际需用户按业务场景自行组装
            '''
            beforeTime = datetime.datetime.now()
            print(f"结果查询开始时间{beforeTime}")
            answer = ''
            while resp["code"] == 200 and resp['data']['taskStatus'] == 'PROCESSING':
                taskOrderNo = resp['data']['taskOrderNo']
                time.sleep(10)
                resp = queryTaskResult(token, taskOrderNo)
                print(resp)
                answer = resp['data']['outputText']
            print("----------FINISHED-------------")
            print(f"回复：{answer}")
            afterTime = datetime.datetime.now()
            print(f"结果响应结束时间{afterTime}")
            print(f'总耗费时间：{afterTime - beforeTime}秒')
            return answer
        else:
            print("获取token失败，请检查 API_KEY 和 PUBLIC_KEY")

    # def load_model(self,
    #                model_name_or_path: str = "THUDM/chatglm-6b",
    #                llm_device=LLM_DEVICE,
    #                use_ptuning_v2=False,
    #                use_lora=False,
    #                device_map: Optional[Dict[str, int]] = None,
    #                **kwargs):
    #     self.tokenizer = AutoTokenizer.from_pretrained(
    #         model_name_or_path,
    #         trust_remote_code=True
    #     )
    #
    #     model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    #
    #     if use_ptuning_v2:
    #         try:
    #             prefix_encoder_file = open('ptuning-v2/config.json', 'r')
    #             prefix_encoder_config = json.loads(prefix_encoder_file.read())
    #             prefix_encoder_file.close()
    #             model_config.pre_seq_len = prefix_encoder_config['pre_seq_len']
    #             model_config.prefix_projection = prefix_encoder_config['prefix_projection']
    #         except Exception as e:
    #             logger.error(f"加载PrefixEncoder config.json失败: {e}")
    #     self.model = AutoModel.from_pretrained(model_name_or_path, config=model_config, trust_remote_code=True,
    #                                            **kwargs)
    #     if LLM_LORA_PATH and use_lora:
    #         from peft import PeftModel
    #         self.model = PeftModel.from_pretrained(self.model, LLM_LORA_PATH)
    #
    #     if torch.cuda.is_available() and llm_device.lower().startswith("cuda"):
    #         # 根据当前设备GPU数量决定是否进行多卡部署
    #         num_gpus = torch.cuda.device_count()
    #         if num_gpus < 2 and device_map is None:
    #             self.model = self.model.half().cuda()
    #         else:
    #             from accelerate import dispatch_model
    #
    #             # model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True,
    #             #         config=model_config, **kwargs)
    #             if LLM_LORA_PATH and use_lora:
    #                 from peft import PeftModel
    #                 model = PeftModel.from_pretrained(self.model, LLM_LORA_PATH)
    #             # 可传入device_map自定义每张卡的部署情况
    #             if device_map is None:
    #                 device_map = auto_configure_device_map(num_gpus, use_lora)
    #
    #             self.model = dispatch_model(self.model.half(), device_map=device_map)
    #     else:
    #         self.model = self.model.float().to(llm_device)
    #
    #     if use_ptuning_v2:
    #         try:
    #             prefix_state_dict = torch.load('ptuning-v2/pytorch_model.bin')
    #             new_prefix_state_dict = {}
    #             for k, v in prefix_state_dict.items():
    #                 if k.startswith("transformer.prefix_encoder."):
    #                     new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    #             self.model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    #             self.model.transformer.prefix_encoder.float()
    #         except Exception as e:
    #             logger.error(f"加载PrefixEncoder模型参数失败:{e}")
    #
    #     self.model = self.model.eval()


if __name__ == "__main__":
    llm = ChatGLM()
    # llm.load_model(model_name_or_path=llm_model_dict[LLM_MODEL],
    #                llm_device=LLM_DEVICE, )
    last_print_len = 0
    # for resp, history in llm._call("你好", streaming=True):
    #     logger.info(resp[last_print_len:], end="", flush=True)
    #     last_print_len = len(resp)
    for resp, history in llm._call("你好", streaming=False):
        logger.info(resp)
    pass
