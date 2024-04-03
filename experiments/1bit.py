import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("NousResearch/OLMo-Bitnet-1B")
model = AutoModelForCausalLM.from_pretrained("NousResearch/OLMo-Bitnet-1B",
    torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

streamer = TextStreamer(tokenizer)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id,
    temperature=0.8, repetition_penalty=1.1, do_sample=True,streamer=streamer)
pipe("The capitol of Paris is",  max_new_tokens=256)
