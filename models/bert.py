import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import BertPretrainingHeads, BertEncoder, BertEmbedding, BertPooler

class Bert(nn.Module):

    def __init__(
        self,
        num_blocks,
        num_heads,
        d_model,
        vocab_size,
        d_ff,
        max_len=512,
        pd=0.1,
        norm_eps=1e-12,
        initialize=True,
        add_pooler=True,
    ) -> None:
        super().__init__()
        self.bert_embedding = BertEmbedding(vocab_size, d_model, max_len, pd, norm_eps)
        self.bert_encoder = BertEncoder(
            num_blocks, d_model, num_heads, d_ff, pd, F.gelu, norm_eps, True, True
        )
        self.pooler = BertPooler(d_model) if add_pooler else None

        if initialize:
            self.init_parameters()

    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, attention_mask=None, token_type_ids=None):
        x = self.bert_embedding(x, token_type_ids=token_type_ids)
        x = self.bert_encoder(
            x, attention_mask=attention_mask
        )

        pooler_output = self.pooler(x) if self.pooler is not None else None
        return x, pooler_output


class BertForPretraining(nn.Module):

    def __init__(
        self,
        num_blocks,
        num_heads,
        d_model,
        vocab_size,
        d_ff,
        max_len=512,
        pd=0,
        norm_eps=1e-12,
        initialize=True,
        add_pooler=True,
        device=None,
    ) -> None:
        super().__init__()
        self.bert = Bert(
            num_blocks,
            num_heads,
            d_model,
            vocab_size,
            d_ff,
            max_len,
            pd,
            norm_eps,
            initialize,
            add_pooler,
        )
        self.cls = BertPretrainingHeads(d_model, vocab_size)
        self.cls.mlm_head.decoder.weight = (
            self.bert.bert_embedding.word_embedding.weight
        )
        self.loss = nn.CrossEntropyLoss(
            ignore_index=-100, reduction="mean")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if initialize:
            self.init_parameters()

    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch):
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        next_sentence_label = batch["next_sentence_label"].to(self.device)
        sequence_output, pooled_output = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )
        loss_mlm = self.loss(
            prediction_scores.view(-1, prediction_scores.size(-1)),
            labels.view(-1),
        )
        loss_nsp = self.loss(
            seq_relationship_score.view(-1, seq_relationship_score.size(-1)),
            next_sentence_label.view(-1),
        )
        total_loss = loss_mlm + loss_nsp
        return {'loss': total_loss,
                'prediction_scores': prediction_scores,
                'seq_relationship_score': seq_relationship_score}
