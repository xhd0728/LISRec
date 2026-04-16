import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, data, tokenizer, args):
        self.data = data
        self.tokenizer = tokenizer
        self.args = args
        self.template_tokens = self._build_template_tokens()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _build_template_tokens(self):
        templates = (
            "Here is the visit history list of user: ",
            " recommend next item ",
        )
        return [
            self.tokenizer.encode(
                template, add_special_tokens=False, truncation=False
            )
            for template in templates
        ]

    def collect_fn(self, batch):
        sequence_ids = []
        sequence_masks = []
        batch_target = []

        for seq_text, target_item in batch:
            batch_target.append(target_item)
            encoded_sequence = self.tokenizer.encode(
                seq_text, add_special_tokens=False, truncation=False
            )
            sequence_chunks = list_split(encoded_sequence, self.args.split_num)
            sequence_chunks[0] = (
                self.template_tokens[0]
                + sequence_chunks[0]
                + self.template_tokens[1]
            )

            chunk_ids = []
            chunk_masks = []
            for chunk in sequence_chunks:
                outputs = self.tokenizer.encode_plus(
                    chunk,
                    max_length=self.args.seq_size,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                )
                chunk_ids.append(outputs["input_ids"])
                chunk_masks.append(outputs["attention_mask"])

            chunk_ids = torch.cat(chunk_ids, dim=0)
            chunk_masks = torch.cat(chunk_masks, dim=0)
            num_chunks = chunk_ids.size(0)
            if num_chunks < self.args.num_passage:
                pad_rows = self.args.num_passage - num_chunks
                seq_length = chunk_ids.size(1)
                padding = torch.zeros((pad_rows, seq_length), dtype=chunk_ids.dtype)
                chunk_ids = torch.cat((chunk_ids, padding), dim=0)
                chunk_masks = torch.cat((chunk_masks, padding), dim=0)

            sequence_ids.append(chunk_ids.unsqueeze(0))
            sequence_masks.append(chunk_masks.unsqueeze(0))

        sequence_ids = torch.cat(sequence_ids, dim=0)
        sequence_masks = torch.cat(sequence_masks, dim=0)

        return {
            "seq_ids": sequence_ids,
            "seq_masks": sequence_masks,
            "target_list": batch_target,
        }


class ItemDataset(Dataset):
    def __init__(self, data, tokenizer, args):
        self.data = data
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collect_fn(self, batch):
        item_ids, item_masks = encode_batch(batch, self.tokenizer, self.args.item_size)

        return {
            "item_ids": item_ids,
            "item_masks": item_masks,
        }


def list_split(array, n):
    first_chunk = array[:n]
    second_chunk = array[n:]
    return [first_chunk] + ([second_chunk] if second_chunk else [])


def encode_batch(batch_text, tokenizer, max_length):
    outputs = tokenizer.batch_encode_plus(
        batch_text,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
    )
    input_ids = outputs["input_ids"].unsqueeze(1)
    attention_mask = outputs["attention_mask"].unsqueeze(1)

    return input_ids, attention_mask


def load_item_name(filename):
    item_desc = {}
    title_prefix = "title:"

    with open(filename, "r", encoding="utf-8") as file:
        next(file, None)
        for line_number, raw_line in enumerate(file, start=2):
            try:
                fields = raw_line.strip().split("\t")
                if len(fields) < 2:
                    raise ValueError(
                        f"Line {line_number} does not have enough elements: {fields}"
                    )

                item_id = int(fields[0])
                item_name = fields[1].replace("&amp;", "")
                item_desc[item_id] = f"{title_prefix} {item_name}"
            except Exception as exc:
                print(f"Error processing line {line_number}: {raw_line.strip()}")
                print(f"Exception: {exc}")

    return item_desc


def load_item_address(filename):
    item_desc = {}
    title_prefix = "title:"
    passage_prefix = "address:"

    with open(filename, "r", encoding="utf-8") as file:
        next(file, None)
        for raw_line in file:
            fields = raw_line.strip().split("\t")
            item_id = int(fields[0])
            name = fields[1]
            address = fields[3]
            city = fields[4]
            state = fields[5]
            item_desc[item_id] = (
                f"{title_prefix} {name} {passage_prefix} {address} {city} {state}"
            )

    return item_desc


def load_data(filename, item_desc):
    data = []

    with open(filename, "r", encoding="utf-8") as file:
        next(file, None)
        for raw_line in file:
            fields = raw_line.strip().split("\t")
            target_item = int(fields[-1])
            sequence_ids = fields[1:-1]

            text_list = []
            for item_id in sequence_ids:
                item_id = int(item_id)
                if item_id == 0:
                    break
                text_list.append(item_desc[item_id])

            text_list.reverse()
            data.append([", ".join(text_list), target_item])

    return data


def load_item_data(item_desc):
    return list(item_desc.values())
