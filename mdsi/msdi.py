class ModularSDIModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        proj_dim: int = 16,
        num_classes: int = 2,
        max_pairs: int = 128,
        memory_momentum: float = 0.95,
        use_sums: bool = True,
        use_mlp: bool = True,
        sdi_only_logits: bool = False,
        normalize_memory: bool = True,
        normalize_diffs: bool = True,
        agg_method: Literal["mean", "sum", "max"] = "mean",
        deterministic_pairs: bool = False,
        diff_dropout_rate: float = 0.0,
        memory_freeze_epoch: Optional[int] = None,
        memory_update_strategy: Literal["moving_average", "batchwise_reset"] = "moving_average",
        memory_attention: bool = False
    ):
        super().__init__()
        assert agg_method in {"mean", "sum", "max"}
        assert memory_update_strategy in {"moving_average", "batchwise_reset"}

        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.max_pairs = max_pairs
        self.memory_momentum = memory_momentum
        self.use_sums = use_sums
        self.use_mlp = use_mlp
        self.sdi_only_logits = sdi_only_logits
        self.normalize_memory = normalize_memory
        self.normalize_diffs = normalize_diffs
        self.agg_method = agg_method
        self.deterministic_pairs = deterministic_pairs
        self.diff_dropout_rate = diff_dropout_rate
        self.memory_freeze_epoch = memory_freeze_epoch
        self.memory_update_strategy = memory_update_strategy
        self.memory_attention = memory_attention
        self.epoch = 0

        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(diff_dropout_rate) if diff_dropout_rate > 0 else nn.Identity()
        self.register_buffer("sdi_memory", F.normalize(torch.randn(num_classes, proj_dim), dim=1))

        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, num_classes)
            )

        if self.memory_attention:
            self.memory_attn = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=1, batch_first=True)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def compute_sdi_vector(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)

        if z.shape[0] < 2:
            return torch.zeros(1, self.proj_dim, device=z.device)

        if self.deterministic_pairs:
            idx = torch.arange(len(z) - 1)
            pairs = torch.stack([idx, idx + 1], dim=1)
        else:
            pairs = torch.combinations(torch.arange(z.shape[0], device=z.device), r=2)
            if len(pairs) > self.max_pairs:
                idx = torch.randperm(len(pairs))[:self.max_pairs]
                pairs = pairs[idx]

        diffs = z[pairs[:, 0]] - z[pairs[:, 1]]
        parts = [diffs]
        if self.use_sums:
            parts.append(z[pairs[:, 0]] + z[pairs[:, 1]])

        z = torch.cat(parts, dim=0)

        if self.normalize_diffs:
            z = F.normalize(z, dim=1)

        z = self.dropout(z)

        if self.agg_method == "mean":
            return z.mean(dim=0, keepdim=True)
        elif self.agg_method == "sum":
            return z.sum(dim=0, keepdim=True)
        elif self.agg_method == "max":
            z, _ = z.max(dim=0, keepdim=True)
            return z

    def update_memory(self, sdi_vecs: torch.Tensor, y: torch.Tensor):
        if self.memory_freeze_epoch is not None and self.epoch >= self.memory_freeze_epoch:
            return
        with torch.no_grad():
            for c in range(self.num_classes):
                mask = (y == c)
                if mask.any():
                    new_mean = sdi_vecs[mask].mean(dim=0)
                    updated = (
                        self.memory_momentum * self.sdi_memory[c] +
                        (1 - self.memory_momentum) * new_mean
                        if self.memory_update_strategy == "moving_average"
                        else new_mean
                    )
                    self.sdi_memory[c] = F.normalize(updated, dim=0) if self.normalize_memory else updated

    def compute_logits(self, sdi_vecs: torch.Tensor, B: int) -> torch.Tensor:
        mem = F.normalize(self.sdi_memory, dim=1) if self.normalize_memory else self.sdi_memory
        logits = torch.matmul(sdi_vecs, mem.T)

        if self.memory_attention:
            mem_exp = mem.unsqueeze(0).expand(B, -1, -1)
            query = sdi_vecs.unsqueeze(1)
            attended, _ = self.memory_attn(query=query, key=mem_exp, value=mem_exp)
            logits = logits + attended.squeeze(1) @ mem.T

        if self.use_mlp and not self.sdi_only_logits:
            mlp_logits = self.mlp(sdi_vecs)
            logits = logits + mlp_logits

        return logits

    def forward(self, X: torch.Tensor, y: Optional[torch.Tensor] = None, update_memory: bool = False):
        if X.dim() == 2:
            X = X.unsqueeze(0)
        B = X.shape[0]
        sdi_vecs = torch.cat([self.compute_sdi_vector(X[b]) for b in range(B)], dim=0)

        if update_memory and y is not None:
            self.update_memory(sdi_vecs, y)

        logits = self.compute_logits(sdi_vecs, B)
        return logits, sdi_vecs
