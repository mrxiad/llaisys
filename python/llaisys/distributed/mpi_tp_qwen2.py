import argparse
from dataclasses import dataclass
from typing import List, Tuple

from transformers import AutoTokenizer

import llaisys

try:
    from mpi4py import MPI
except Exception as exc:  # pylint: disable=broad-except
    raise RuntimeError(
        "mpi4py is required for Project #5 CPU MPI path. "
        "Install with: ./.venv310/bin/pip install mpi4py"
    ) from exc


@dataclass
class ShardRange:
    start: int
    end: int


def vocab_shard(vocab_size: int, rank: int, world_size: int) -> ShardRange:
    start = (vocab_size * rank) // world_size
    end = (vocab_size * (rank + 1)) // world_size
    return ShardRange(start=start, end=end)


def distributed_argmax_step(
    model: llaisys.models.Qwen2,
    tokens: List[int],
    shard: ShardRange,
    comm: MPI.Comm,
    rank: int,
) -> int:
    local_idx, local_val = model.infer_shard_argmax(tokens, shard.start, shard.end)
    gathered: List[Tuple[float, int]] = comm.gather((float(local_val), int(local_idx)), root=0)

    if rank == 0:
        best_val, best_idx = max(gathered, key=lambda x: x[0])
        _ = best_val
        next_token = int(best_idx)
    else:
        next_token = 0
    next_token = comm.bcast(next_token, root=0)
    return int(next_token)


def run_distributed_infer(
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    rank: int,
    world_size: int,
    comm: MPI.Comm,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = llaisys.models.Qwen2(model_path, device=llaisys.DeviceType.CPU)
    shard = vocab_shard(model._meta.voc, rank, world_size)  # noqa: SLF001

    if rank == 0:
        prompt_text = tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        tokens = tokenizer.encode(prompt_text)
    else:
        tokens = None
    tokens = comm.bcast(tokens, root=0)
    prompt_len = len(tokens)

    for _ in range(int(max_new_tokens)):
        if len(tokens) >= int(model._meta.maxseq):  # noqa: SLF001
            break
        next_token = distributed_argmax_step(model, tokens, shard, comm, rank)
        tokens.append(next_token)
        if next_token == int(model._end_token):  # noqa: SLF001
            break

    if rank == 0:
        out_text = tokenizer.decode(tokens[prompt_len:], skip_special_tokens=True)
        print("=== Distributed Tensor Parallel Result ===")
        print(f"world_size={world_size}")
        print(f"generated_tokens={len(tokens) - prompt_len}")
        print(out_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max-new-tokens", default=32, type=int)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    run_distributed_infer(
        model_path=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        rank=rank,
        world_size=world_size,
        comm=comm,
    )


if __name__ == "__main__":
    main()
