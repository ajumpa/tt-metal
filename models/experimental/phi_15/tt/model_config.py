"""
PhiForCausalLM(
  (model): PhiModel(
    (embed_tokens): Embedding(51200, 2048)
    (embed_dropout): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0-23): 24 x PhiDecoderLayer(
        (self_attn): PhiAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (dense): Linear(in_features=2048, out_features=2048, bias=True)
          (rotary_emb): PhiRotaryEmbedding()
        )
        (mlp): PhiMLP(
          (activation_fn): NewGELUActivation()
          (fc1): Linear(in_features=2048, out_features=8192, bias=True)
          (fc2): Linear(in_features=8192, out_features=2048, bias=True)
        )
        (input_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (resid_dropout): Dropout(p=0.0, inplace=False)
      )
    )
    (final_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=2048, out_features=51200, bias=True)
)

model.embed_tokens.weight: torch.Size([51200, 2048])
for i in range(0,24)
  model.layers.0.self_attn.q_proj.weight: torch.Size([2048, 2048])
  model.layers.0.self_attn.q_proj.bias: torch.Size([2048])
  model.layers.0.self_attn.k_proj.weight: torch.Size([2048, 2048])
  model.layers.0.self_attn.k_proj.bias: torch.Size([2048])
  model.layers.0.self_attn.v_proj.weight: torch.Size([2048, 2048])
  model.layers.0.self_attn.v_proj.bias: torch.Size([2048])
  model.layers.0.self_attn.dense.weight: torch.Size([2048, 2048])
  model.layers.0.self_attn.dense.bias: torch.Size([2048])
  model.layers.0.mlp.fc1.weight: torch.Size([8192, 2048])
  model.layers.0.mlp.fc1.bias: torch.Size([8192])
  model.layers.0.mlp.fc2.weight: torch.Size([2048, 8192])
  model.layers.0.mlp.fc2.bias: torch.Size([2048])
  model.layers.0.input_layernorm.weight: torch.Size([2048])
  model.layers.0.input_layernorm.bias: torch.Size([2048])
model.final_layernorm.weight: torch.Size([2048])
model.final_layernorm.bias: torch.Size([2048])
lm_head.weight: torch.Size([51200, 2048])
lm_head.bias: torch.Size([51200])


PhiConfig {
  "_name_or_path": "microsoft/phi-1_5",
  "architectures": [
    "PhiForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": null,
  "embd_pdrop": 0.0,
  "eos_token_id": null,
  "hidden_act": "gelu_new",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 2048,
  "model_type": "phi",
  "num_attention_heads": 32,
  "num_hidden_layers": 24,
  "num_key_value_heads": 32,
  "partial_rotary_factor": 0.5,
  "qk_layernorm": false,
  "resid_pdrop": 0.0,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.38.0",
  "use_cache": true,
  "vocab_size": 51200
}

"""
