from llm_cfg_extraction.model import CFGModel
from llm_cfg_extraction.cfg import CFG
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


json_grammar = r"""
    ?start: value

    ?value: syptoms
    syptoms : "["  [syptm ("," syptm)*]  "]" 
    syptm : "{" qt zhengzhuang qt ":" qt zhengzhuang_txt+  qt  "," qt buwei qt ":" qt buwei_txt+ qt "}"
    qt : /"/
    zhengzhuang : /症状/
    buwei : /部位/
    zhengzhuang_txt : /(?<="症状":")[^"]+/
    buwei_txt : /(?<="部位":")[^"]+/
    string : ESCAPED_STRING
    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS

    %ignore WS
"""

llm_path = "/mypath/Baichuan2-13B-Chat"


device = 'cuda' if torch.cuda.is_avaibale() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(
    llm_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    llm_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(llm_path)
model.to(device)

cfg = CFG(tokenizer, json_grammar)
cfg_model = CFGModel(model, tokenizer, cfg)

messages = []
messages.append(
    {"role": "user", "content": '解析文本中的症状，每个症状需要提取症状描述与症状部位，并输出json格式。例: 文本:该患者胃部疼痛。结果：[{"症状":"疼痛","部位":"胃"}]。文本:该患者头部疼痛'})
inputs = cfg_model.model._build_chat_input(tokenizer, messages)
inputs.to(device)  # input_ids

print(cfg_model.generate(inputs))
