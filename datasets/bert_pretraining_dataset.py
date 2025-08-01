import torch
from torch.utils.data import Dataset
import random

class BertPretrainingDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        max_seq_length=512,
        mlm_prob=0.15,
        mask_prob=0.8,
        random_prob=0.1,
        same_prob=0.1,
        batched=True,
        num_proc=1,
        device="cpu",
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.chunk_size = max_seq_length // 2

        dataset = dataset.map(
            self.tokenize_function,
            batched=batched,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
        )
        self.dataset = dataset.sort("sample_length")
        assert (
            mask_prob + random_prob + same_prob == 1.0
        ), "mask_prob + random_prob + same_prob should be 1.0"
        self.mlm_prob = mlm_prob
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self.same_prob = same_prob
        self.device = device

    def tokenize_function(self, examples):
        results = {
            "input_ids": [],
            "is_next": [],
            "token_type_ids": [],
            "special_tokens_mask": [],
            "sample_length": [],
        }

        toks = self.tokenizer(
            examples["text"], return_special_tokens_mask=True, add_special_tokens=False
        )

        chunk_main = self.chunk_size - 2
        chunk_random = self.chunk_size - 1
        input_list = toks["input_ids"]
        total = len(input_list)

        for i, tokens in enumerate(input_list):
            num_chunks = len(tokens) // self.chunk_size
            if num_chunks < 2:
                continue

            for j in range(num_chunks - 2):
                a_ids = tokens[j * chunk_main : (j + 1) * chunk_main]

                if random.random() > 0.5:
                    b_ids = tokens[(j + 1) * chunk_random : (j + 2) * chunk_random]
                    is_next = 1
                else:
                    # pick a random different example with at least 2 chunks
                    while True:
                        rand_idx = random.randint(0, total - 1)
                        if (
                            rand_idx != i
                            and len(input_list[rand_idx]) // self.chunk_size >= 2
                        ):
                            break
                    rand_tokens = input_list[rand_idx]
                    rand_chunk = random.randint(
                        0, (len(rand_tokens) // self.chunk_size) - 2
                    )
                    b_ids = rand_tokens[
                        rand_chunk * chunk_random : (rand_chunk + 1) * chunk_random
                    ]
                    is_next = 0

                input_ids = (
                    [self.tokenizer.cls_token_id]
                    + a_ids
                    + [self.tokenizer.sep_token_id]
                    + b_ids
                    + [self.tokenizer.sep_token_id]
                )
                token_type_ids = [0] * (len(a_ids) + 2) + [1] * (len(b_ids) + 1)
                special_tokens_mask = (
                    [1] + [0] * len(a_ids) + [1] + [0] * len(b_ids) + [1]
                )

                results["input_ids"].append(input_ids)
                results["is_next"].append(is_next)
                results["token_type_ids"].append(token_type_ids)
                results["special_tokens_mask"].append(special_tokens_mask)
                results["sample_length"].append(len(input_ids))

        return results

    def mask_example(self, example):
        input_ids = torch.tensor(example["input_ids"])
        special_tokens_mask = torch.tensor(
            example["special_tokens_mask"], dtype=torch.bool
        )
        token_type_ids = torch.tensor(example["token_type_ids"])
        is_next = torch.tensor(example["is_next"])

        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        probability_matrix[special_tokens_mask] = 0.0

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% [MASK]
        replace_prob = torch.bernoulli(torch.full(labels.shape, self.mask_prob)).bool()
        input_ids[masked_indices & replace_prob] = self.tokenizer.mask_token_id

        # 10% random token
        random_prob = self.random_prob / (1 - self.mask_prob)
        rand_prob = torch.bernoulli(torch.full(labels.shape, random_prob)).bool()
        random_tokens = torch.randint(
            len(self.tokenizer), input_ids.shape, dtype=torch.long
        )
        input_ids[masked_indices & ~replace_prob & rand_prob] = random_tokens[
            masked_indices & ~replace_prob & rand_prob
        ]

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "labels": labels,
            "special_tokens_mask": special_tokens_mask,
            "is_next": is_next,
        }

    def __getitem__(self, idx):
        return self.mask_example(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)
