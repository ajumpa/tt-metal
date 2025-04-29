import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.rope import RotarySetup


class PhiAttention(LightweightModule):
    def __init__(
        self,
        device,
        q_proj_weight,
        q_proj_bias,
        k_proj_weight,
        k_proj_bias,
        v_proj_weight,
        v_proj_bias,
        dense_weight,
        dense_bias,
        dim=2048,
        heads=32,
        rope_theta=10000,
    ):
        super().__init__()

        self.q_w = q_proj_weight
        self.q_b = q_proj_bias
        self.k_w = k_proj_weight
        self.k_b = k_proj_bias
        self.v_w = v_proj_weight
        self.v_b = v_proj_bias
        self.dense_weight = dense_weight
        self.dense_bias = dense_bias

        self.heads = heads
        self.dim = dim
        self.rope_theta = rope_theta
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5

        rope_setup_decode = RotarySetup(device, self.head_dim, dim, rope_theta, use_scaled_rope=None)

        self.transformation_mats_decode = rope_setup_decode.get_trans_mats()

        # Projections
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        batch_sz, _, seq_len = x.shape
        qkv = self.qkv(x).reshape(batch_sz, seq_len, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, T, head_dim)

        position_ids = ttnn.arange(batch_sz)
        cos, sin = self.rope_setup_decode.get_rot_mats(position_ids)
        q = ttnn.experimental.rotary_embedding_llama(q, cos, sin, self.transformation_mats_decode, is_decode_mode=True)

        k = ttnn.experimental.rotary_embedding_llama(
            k, self.cos_matrix, self.sin_matrix, self.transformation_mats_decode, is_decode_mode=True
        )

        # Flash Attention (use PyTorch 2.0+ optimized attention)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.permute(0, 2, 1, 3).reshape(batch_sz, seq_len, -1)
        attn = ttnn.linear(attn, weight=self.dense_weight, bias=self.dense_bias)
        return attn
