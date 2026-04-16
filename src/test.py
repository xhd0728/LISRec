import os

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import T5Tokenizer
from src.data_loader import (
    ItemDataset,
    SequenceDataset,
    load_data,
    load_item_address,
    load_item_data,
    load_item_name,
)
from src.metrics import get_metrics_dict
from src.model import TASTEModel
from src.option import Options
from src.utils import init_logger, set_randomseed


def evaluate(model, test_seq_dataloader, test_item_dataloader, device, ks, logger):
    logger.info("***** Running testing *****")
    model.eval()
    model = model.module if hasattr(model, "module") else model
    item_embeddings, seq_embeddings, target_item_list = [], [], []
    top_k = max(ks)

    with torch.no_grad():
        for batch in test_item_dataloader:
            item_inputs, item_masks = batch["item_ids"].to(device), batch[
                "item_masks"
            ].to(device)
            _, item_emb = model(item_inputs, item_masks)
            item_embeddings.append(item_emb.cpu().numpy())

        item_embeddings = np.concatenate(item_embeddings, axis=0)

        for batch in test_seq_dataloader:
            seq_inputs, seq_masks = batch["seq_ids"].to(device), batch["seq_masks"].to(
                device
            )
            batch_target = batch["target_list"]
            _, seq_emb = model(seq_inputs, seq_masks)
            seq_embeddings.append(seq_emb.cpu().numpy())
            target_item_list.extend(batch_target)

        seq_embeddings = np.concatenate(seq_embeddings, axis=0)

        faiss.omp_set_num_threads(16)
        cpu_index = faiss.IndexFlatIP(item_embeddings.shape[1])
        cpu_index.add(np.asarray(item_embeddings, dtype=np.float32))
        query_embeds = np.asarray(seq_embeddings, dtype=np.float32)
        _, ranked_indices = cpu_index.search(query_embeds, top_k)

        n_item, n_seq = item_embeddings.shape[0], seq_embeddings.shape[0]
        metrics_dict = get_metrics_dict(
            ranked_indices, n_seq, n_item, ks, target_item_list
        )

        logger.info(
            f"Test: Recall@10: {metrics_dict[10]['recall']:.4f}, Recall@20: {metrics_dict[20]['recall']:.4f}, NDCG@10: {metrics_dict[10]['ndcg']:.4f}, NDCG@20: {metrics_dict[20]['ndcg']:.4f}"
        )
    logger.info("***** Finish test *****")


def main():
    opt = Options().parse()
    set_randomseed(opt.seed)
    logger = init_logger(os.path.join(opt.logging_dir, "test", "test.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5Tokenizer.from_pretrained(opt.best_model_path)
    model = TASTEModel.from_pretrained(opt.best_model_path).to(device)

    item_desc = (
        load_item_name(os.path.join(opt.data_dir, "item.txt"))
        if opt.data_name in ["beauty", "sports", "toys"]
        else load_item_address(os.path.join(opt.data_dir, "item.txt"))
    )

    logger.info(f"item len: {len(item_desc)}")

    test_data = load_data(os.path.join(opt.data_dir, "test.txt"), item_desc)
    logger.info(f"test len: {len(test_data)}")

    item_data = load_item_data(item_desc)
    test_seq_dataset = SequenceDataset(test_data, tokenizer, opt)
    test_item_dataset = ItemDataset(item_data, tokenizer, opt)

    test_seq_dataloader = DataLoader(
        test_seq_dataset,
        sampler=SequentialSampler(test_seq_dataset),
        batch_size=opt.eval_batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=test_seq_dataset.collect_fn,
    )

    test_item_dataloader = DataLoader(
        test_item_dataset,
        sampler=SequentialSampler(test_item_dataset),
        batch_size=opt.eval_batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=test_item_dataset.collect_fn,
    )

    evaluate(
        model,
        test_seq_dataloader,
        test_item_dataloader,
        device,
        opt.Ks,
        logger,
    )


if __name__ == "__main__":
    main()
