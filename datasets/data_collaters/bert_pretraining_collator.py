import torch
from torch.nn.utils.rnn import pad_sequence

class BertPretrainingCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [example["input_ids"] for example in batch]
        token_type_ids = [example["token_type_ids"] for example in batch]
        attention_mask = [torch.ones(len(x), dtype=torch.long) for x in input_ids]
        labels = [example["labels"] for example in batch]
        special_tokens_mask = [example["special_tokens_mask"] for example in batch]
        is_next = torch.stack([example["is_next"] for example in batch])

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        special_tokens_mask = pad_sequence(
            special_tokens_mask, batch_first=True, padding_value=1
        )

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "special_tokens_mask": special_tokens_mask,
            "next_sentence_label": is_next,
        }

