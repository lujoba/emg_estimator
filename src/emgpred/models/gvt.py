"""GlobalViewTransformer (GVT): encoder temporal seguido de encoder espacial."""
import keras
from keras import layers

from emgpred.models.layers import POSITIONAL_EMBEDDINGS, ClassToken, TransformerEncoderBlock


def gvt_backbone(inputs, time_steps, space_dim, output_dim, channels=1, d_model=128, n_heads=4,
                 n_layers_time=4, n_layers_space=4, mlp_ratio=4.0, dropout=0.4,
                 pooling_method="cls", pos_embedding_method="learned",
                 use_causal_mask_time=True, use_causal_mask_space=False):
    """Aplica o GVT a um tensor simbólico `inputs` (B, T, C, channels) e devolve (B, output_dim).

    Separado de `build_gvt` para poder ser composto com outras camadas (ex.: o seletor
    de eletrodos do autopruning).
    """
    assert pooling_method in ("gap", "max", "cls"), "pooling_method deve ser 'gap', 'max' ou 'cls'"
    assert pos_embedding_method in POSITIONAL_EMBEDDINGS, f"opções: {list(POSITIONAL_EMBEDDINGS)}"
    PosEmb = POSITIONAL_EMBEDDINGS[pos_embedding_method]

    # --- Bloco temporal: cada instante é um token com todos os eletrodos ---
    x = layers.Reshape((time_steps, space_dim * channels))(inputs)
    x = layers.Dense(d_model, name="feature_projection")(x)
    if pooling_method == "cls":
        x = ClassToken(d_model=d_model, name="time_cls_token")(x)
        effective_time_steps = time_steps + 1
    else:
        effective_time_steps = time_steps
    x = PosEmb(max_length=effective_time_steps, d_model=d_model, name="time_pos_emb")(x)
    for i in range(n_layers_time):
        x = TransformerEncoderBlock(n_heads, mlp_ratio, dropout, use_causal_mask=use_causal_mask_time,
                                    name=f"time_encoder_{i + 1}")(x)

    # --- Bloco espacial: transposto, cada feature é um token ---
    x = layers.Permute((2, 1))(x)                      # (B, d_model, effective_time_steps)
    x = layers.Dense(d_model, name="space_feature_projection")(x)
    x = PosEmb(max_length=d_model, d_model=d_model, name="space_pos_emb")(x)
    for i in range(n_layers_space):
        x = TransformerEncoderBlock(n_heads, mlp_ratio, dropout, use_causal_mask=use_causal_mask_space,
                                    name=f"space_encoder_{i + 1}")(x)

    # --- Pooling e cabeça ---
    if pooling_method == "cls":
        x = x[:, 0, :]
    elif pooling_method == "max":
        x = layers.GlobalMaxPooling1D()(x)
    else:
        x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(int(d_model * mlp_ratio), activation="gelu")(x)
    x = layers.Dropout(dropout)(x)
    return layers.Dense(output_dim)(x)


def build_gvt(time_steps=100, space_dim=128, channels=1, output_dim=5, **kwargs):
    """Modelo GVT completo com entrada (time_steps, space_dim, channels)."""
    inputs = keras.Input(shape=(time_steps, space_dim, channels))
    outputs = gvt_backbone(inputs, time_steps, space_dim, output_dim, channels=channels, **kwargs)
    return keras.Model(inputs=inputs, outputs=outputs, name="GlobalViewTransformer")
