import torch
import torch.nn as nn

from Model_PMA import CharBertModel, Model


def _load_encoder_checkpoint(module, ckpt_path, label, allowed_unexpected_prefixes=()):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = module.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.startswith("cbam.")]
    unexpected = [
        key for key in unexpected
        if not any(key.startswith(prefix) for prefix in allowed_unexpected_prefixes)
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"{label} checkpoint load mismatch. Missing keys: {missing}. Unexpected keys: {unexpected}."
        )


def load_fusion_checkpoint(module, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    current_state = module.state_dict()

    filtered_state = {}
    skipped_shape = []
    skipped_unknown = []

    for key, value in state.items():
        if key not in current_state:
            skipped_unknown.append(key)
            continue
        if current_state[key].shape != value.shape:
            skipped_shape.append((key, tuple(value.shape), tuple(current_state[key].shape)))
            continue
        filtered_state[key] = value

    missing, unexpected = module.load_state_dict(filtered_state, strict=False)
    required_prefixes = ("fusion_fc1.", "fusion_fc2.")
    required_missing = [key for key in missing if key.startswith(required_prefixes)]

    if required_missing:
        raise RuntimeError(
            "Fusion checkpoint is missing required fusion-head weights. "
            f"Missing keys: {required_missing}."
        )
    if unexpected:
        raise RuntimeError(
            "Fusion checkpoint contains unexpected keys after compatibility filtering. "
            f"Unexpected keys: {unexpected}."
        )
    if skipped_shape:
        mismatch_text = ", ".join(
            f"{key} saved{saved_shape} current{current_shape}"
            for key, saved_shape, current_shape in skipped_shape
        )
        raise RuntimeError(
            "Fusion checkpoint has incompatible tensor shapes. "
            f"Mismatched keys: {mismatch_text}."
        )

    loaded_keys = sorted(filtered_state.keys())
    loaded_fusion_keys = [key for key in loaded_keys if key.startswith(required_prefixes)]
    skipped_encoder_keys = [
        key for key in skipped_unknown
        if key.startswith("url_encoder.") or key.startswith("html_encoder.")
    ]
    if skipped_encoder_keys:
        print(
            "Loaded fusion checkpoint with legacy encoder compatibility. "
            f"Skipped {len(skipped_encoder_keys)} encoder keys and loaded "
            f"{len(loaded_fusion_keys)} fusion-head tensors from {ckpt_path}."
        )
    else:
        print(f"Loaded fusion checkpoint from {ckpt_path}.")


class FusionModel(nn.Module):

    def __init__(
        self,
        url_ckpt="urlmodel.pth",
        html_ckpt="htmlmodel.pth",
        num_classes=2,
        hidden_dim=768,
        dropout=0.1,
        freeze_encoders=True,
    ):
        super(FusionModel, self).__init__()
        self.url_encoder = CharBertModel()
        self.html_encoder = Model()

        if url_ckpt:
            _load_encoder_checkpoint(self.url_encoder, url_ckpt, "URL")
        if html_ckpt:
            _load_encoder_checkpoint(self.html_encoder, html_ckpt, "HTML")

        if freeze_encoders:
            for p in self.url_encoder.parameters():
                p.requires_grad = False
            for p in self.html_encoder.parameters():
                p.requires_grad = False

        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_fc2 = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.GELU()

    def forward(self, url_inputs, html_inputs):
        _, url_pooled, url_logits = self.url_encoder(url_inputs)
        _, html_pooled, html_logits = self.html_encoder(html_inputs)

        fused_embedding = torch.cat([url_pooled, html_pooled], dim=1)
        x = self.fusion_dropout(fused_embedding)
        x = self.activation(self.fusion_fc1(x))
        x = self.fusion_dropout(x)
        fusion_logits = self.fusion_fc2(x)

        return {
            "url_logits": url_logits,
            "html_logits": html_logits,
            "fusion_logits": fusion_logits,
            "url_pooled": url_pooled,
            "html_pooled": html_pooled,
            "fused_embedding": fused_embedding,
        }
