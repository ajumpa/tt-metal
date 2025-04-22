import ttnn


"""
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
"""


def get_freqs(seq_len, dim, device):
    theta = 1.0 / (10000 ** (ttnn.arange(0, dim, 2, device=device).float() / dim))
    t = ttnn.arange(seq_len, device=device).float()
    freqs = ttnn.outer(t, theta)
    return freqs.repeat(1, 2)


class Phi15Model:
    def __init__(self, device, vocab_size=51200, n_layers=24, dim=2048, heads=32):
        self.device = device
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dim = dim
        self.heads = heads

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        B, T = x.shape
        freqs = get_freqs(T, self.weights.shape[1] // 2, self.device)
        # x = ttnn.embedding(x, )

    def phi_embedding():
        return None

    def phi_attention():
        return None
