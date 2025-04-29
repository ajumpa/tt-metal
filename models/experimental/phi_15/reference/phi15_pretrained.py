import torch
from transformers import PhiForCausalLM, AutoTokenizer


def phi_prompt(model, tokenizer, prompt):
    if prompt == None:
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


torch.set_default_device("cpu")

# define the model and tokenizer.
model = PhiForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

config = model.config
state_dict = model.state_dict()

# Extract all weights as a dictionary
weights = {name: param.data for name, param in model.named_parameters()}


print(config.layer_norm_eps)
