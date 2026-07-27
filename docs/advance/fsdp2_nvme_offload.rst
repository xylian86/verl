FSDP2 NVMe Optimizer Offload
============================

verl can keep AdamW gradients, moments, and FP32 master weights in rank-local
NVMe files. The optimizer reads and updates one host-memory chunk at a time,
which avoids materializing complete optimizer states in CPU DRAM.

Enable the feature on the actor FSDP config::

    actor_rollout_ref:
      actor:
        strategy: fsdp2
        fsdp_config:
          strategy: fsdp2
          optimizer_offload: false
          offload_policy: false
          nvme_offload:
            enabled: true
            path: /mnt/raid0/verl_nvme
            offload_gradients: true
            offload_optimizer: true
            chunk_size_mb: 256
            state_dtype: fp32
            master_weights: true
            fsync: false

This initial implementation supports transformer FSDP2 engines and
``torch.optim.AdamW``. It is incompatible with ``optimizer_offload`` and
``offload_policy``. Keep ``trainer.use_legacy_worker_impl=disable``.

Storage and memory behavior
---------------------------

Each worker creates a unique ``run_<id>/rank_<rank>`` directory under
``path``. With FP32 moments, gradients, and master weights, allow 16 bytes of
NVMe space per local trainable parameter element. The optimizer stages at most
five chunks in host memory, or approximately ``5 * chunk_size_mb``.

Gradients are spilled after distributed norm clipping, so this version does
not reduce peak memory during backward. It does release gradients before the
chunked optimizer update. Old scratch run directories are not removed
automatically.

Checkpoint behavior
-------------------

Optimizer checkpoints copy the persistent moment and master-weight files
without constructing a full optimizer state dict. Checkpoints currently
require a local filesystem and must resume with the same world size and shard
layout. Set ``trainer.save_freq=-1`` when optimizer checkpoints are not needed.
