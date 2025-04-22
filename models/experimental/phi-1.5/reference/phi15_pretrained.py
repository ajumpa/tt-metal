import torch
import torchinfo
from transformers import PhiForCausalLM, AutoTokenizer

torch.set_default_device("cpu")

# define the model and tokenizer.
model = PhiForCausalLM.from_pretrained("microsoft/phi-1_5")

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

torchinfo.summary(model)

# feel free to change the prompt to your liking.
prompt = '''def print_prime(n):
          """
          Print all primes between 1 and n
          """'''
# apply the tokenizer.
tokens = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

# use the model to generate new tokens.
generated_output = model.generate(**tokens, use_cache=True, max_length=200)

text = tokenizer.batch_decode(generated_output)[0]

print(text)
