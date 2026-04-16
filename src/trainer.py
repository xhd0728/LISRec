import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from openmatch.dataset.train_dataset import (
    MappingTrainDatasetMixin,
    StreamTrainDatasetMixin,
    TrainDatasetBase,
)
from openmatch.trainer import DRTrainer
from transformers import BatchEncoding, DataCollatorWithPadding


@dataclass
class TasteCollator(DataCollatorWithPadding):
    max_q_len: int = 32
    max_p_len: int = 128
    len_seq: int = 2

    def _pad_query_sequence(self, sequence_features):
        collated = self.tokenizer.pad(
            sequence_features,
            padding="max_length",
            max_length=self.max_q_len,
            return_tensors="pt",
        )

        input_ids = collated["input_ids"]
        attention_mask = collated["attention_mask"]
        if input_ids.size(0) < self.len_seq:
            pad_rows = self.len_seq - input_ids.size(0)
            seq_length = input_ids.size(1)
            pad_tensor = torch.zeros((pad_rows, seq_length), dtype=input_ids.dtype)
            input_ids = torch.cat((input_ids, pad_tensor), dim=0)
            attention_mask = torch.cat((attention_mask, pad_tensor), dim=0)

        return input_ids.unsqueeze(0), attention_mask.unsqueeze(0)

    def __call__(self, features):
        query_features = [feature["query_"] for feature in features]
        passage_features = [feature["passages"] for feature in features]

        if isinstance(passage_features[0], list):
            passage_features = [
                passage for passages in passage_features for passage in passages
            ]

        query_tensors = [
            self._pad_query_sequence(sequence_features)
            for sequence_features in query_features
        ]
        query = (
            torch.cat([input_ids for input_ids, _ in query_tensors], dim=0),
            torch.cat([attention_mask for _, attention_mask in query_tensors], dim=0),
        )

        d_collated = self.tokenizer.pad(
            passage_features,
            padding="max_length",
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        item_input_ids = d_collated["input_ids"].unsqueeze(1)
        item_attention_mask = d_collated["attention_mask"].unsqueeze(1)

        return query, (item_input_ids, item_attention_mask)


class TasteTrainer(DRTrainer):
    def _prepare_inputs(
        self, inputs: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [
            (input_ids.to(self.args.device), attention_mask.to(self.args.device))
            for input_ids, attention_mask in inputs
        ]


class TasteTrainDataset(TrainDatasetBase):
    def create_one_example(
        self, text_encoding: List[int], is_query=False
    ) -> BatchEncoding:
        return self.tokenizer.encode_plus(
            text_encoding,
            truncation="only_first",
            max_length=(
                self.data_args.q_max_len if is_query else self.data_args.p_max_len
            ),
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            encoded_query = [
                self.create_one_example(item, True) for item in example["query"]
            ]
            group_positives = example["positives"]
            group_negatives = example["negatives"]
            has_hashed_seed = hashed_seed is not None

            positive_passage = (
                group_positives[0]
                if self.data_args.positive_passage_no_shuffle or not has_hashed_seed
                else group_positives[(hashed_seed + epoch) % len(group_positives)]
            )
            encoded_passages = [self.create_one_example(positive_passage)]

            negative_size = self.data_args.train_n_passages - 1
            negs = self._get_negatives(
                group_negatives, negative_size, epoch, hashed_seed
            )
            encoded_passages.extend(
                self.create_one_example(neg_psg) for neg_psg in negs
            )

            assert len(encoded_passages) == self.data_args.train_n_passages

            return {"query_": encoded_query, "passages": encoded_passages}

        return process_fn

    def _get_negatives(self, group_negatives, negative_size, epoch, hashed_seed):
        if negative_size <= 0:
            return []

        has_hashed_seed = hashed_seed is not None
        if len(group_negatives) < negative_size:
            return (
                random.choices(group_negatives, k=negative_size)
                if has_hashed_seed
                else (group_negatives * 2)[:negative_size]
            )
        if self.data_args.negative_passage_no_shuffle:
            return group_negatives[:negative_size]

        offset = epoch * negative_size % len(group_negatives)
        negatives = group_negatives.copy()
        if has_hashed_seed:
            random.Random(hashed_seed).shuffle(negatives)

        return (negatives * 2)[offset : offset + negative_size]


class StreamDRTrainDataset(StreamTrainDatasetMixin, TasteTrainDataset):
    pass


class MappingDRTrainDataset(MappingTrainDatasetMixin, TasteTrainDataset):
    pass
