import torch
from torch import nn
import torch.nn.functional as F


class EmbeddingModel(nn.Module):
    def __init__(self, num_items, item_dims, **embedding_kwargs):
        super().__init__()
        self.embedding = nn.Embedding(
            num_items, item_dims, **embedding_kwargs
        )

    def forward(self, x):
        return self.embedding(x)

    def lookup(self, zs):
        """Find nearest neighbor in embedding for each z"""
        w = self.embedding.weight
        # zs = (b, emb_dims, steps), w = (num_items, emb_dims)
        results = []
        for i, z in enumerate(zs):
            weights = torch.mm(w, z)
            indexes = torch.argmax(weights, dim=0)
            results.append(indexes)
        return results


class SkipGram(nn.Module):
    def __init__(self, num_items, item_dims, **embedding_kwargs):
        super().__init__()
        self.embedding_u = nn.Embedding(
            num_items, item_dims, **embedding_kwargs
        )
        self.embedding_v = nn.Embedding(
            num_items, item_dims, **embedding_kwargs
        )
        self.num_items = num_items

    def forward(self, x):
        return self.embedding_u(x)

    def loss(self, words, context):
        emb_u = self.embedding_u(words)
        emb_v = self.embedding_v(context)
        scores = torch.bmm(emb_v, emb_u.unsqueeze(-1)).squeeze(-1)
        pos_loss = F.logsigmoid(scores).sum(1)
        # negative samples
        batch_size = words.shape[0]
        context_size = context.shape[1]
        negative_context = torch.randint_like(context, 0, self.num_items)
        emb_v_negative = self.embedding_v(negative_context)
        negative_scores = torch.bmm(emb_v_negative, emb_u.unsqueeze(-1)).squeeze(-1)
        neg_loss = F.logsigmoid(-negative_scores).sum(1)

        return -(pos_loss + neg_loss).mean()

    def lookup(self, zs):
        """Find nearest neighbor in embedding for each z

        zs.shape ~ (batch, embedding_dims, steps)
        """
        w = self.embedding_u.weight
        w_norm = w.pow(2).sum(1).sqrt()[..., None]
        results = []
        for i, z in enumerate(zs):
            weights = torch.mm(w, z) / w_norm
            indexes = torch.argmax(weights[1:], dim=0)
            results.append(indexes)
        return results
