"""
Layer (type:depth-idx)                                  Param #
================================================================================
PhiForCausalLM                                          --
├─PhiModel: 1-1                                         --
│    └─Embedding: 2-1                                   104,857,600
│    └─Dropout: 2-2                                     --
│    └─ModuleList: 2-3                                  --
│    │    └─PhiDecoderLayer: 3-1                        50,354,176
│    │    └─PhiDecoderLayer: 3-2                        50,354,176
│    │    └─PhiDecoderLayer: 3-3                        50,354,176
│    │    └─PhiDecoderLayer: 3-4                        50,354,176
│    │    └─PhiDecoderLayer: 3-5                        50,354,176
│    │    └─PhiDecoderLayer: 3-6                        50,354,176
│    │    └─PhiDecoderLayer: 3-7                        50,354,176
│    │    └─PhiDecoderLayer: 3-8                        50,354,176
│    │    └─PhiDecoderLayer: 3-9                        50,354,176
│    │    └─PhiDecoderLayer: 3-10                       50,354,176
│    │    └─PhiDecoderLayer: 3-11                       50,354,176
│    │    └─PhiDecoderLayer: 3-12                       50,354,176
│    │    └─PhiDecoderLayer: 3-13                       50,354,176
│    │    └─PhiDecoderLayer: 3-14                       50,354,176
│    │    └─PhiDecoderLayer: 3-15                       50,354,176
│    │    └─PhiDecoderLayer: 3-16                       50,354,176
│    │    └─PhiDecoderLayer: 3-17                       50,354,176
│    │    └─PhiDecoderLayer: 3-18                       50,354,176
│    │    └─PhiDecoderLayer: 3-19                       50,354,176
│    │    └─PhiDecoderLayer: 3-20                       50,354,176
│    │    └─PhiDecoderLayer: 3-21                       50,354,176
│    │    └─PhiDecoderLayer: 3-22                       50,354,176
│    │    └─PhiDecoderLayer: 3-23                       50,354,176
│    │    └─PhiDecoderLayer: 3-24                       50,354,176
│    └─LayerNorm: 2-4                                   4,096
├─Linear: 1-2                                           104,908,800
================================================================================
Total params: 1,418,270,720
Trainable params: 1,418,270,720
Non-trainable params: 0
================================================================================



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
"""
