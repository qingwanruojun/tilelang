# AICPU transfer executor for Ascend 950

The framework ships one device executor for all Python programs defined with
`@T.aicpu.kernel`. The host specializes a program into a table of instructions
for each team rank. The device executor submits local copies, channel writes
and notifications on one AICPU_TS thread. User programs share the executor.

Build the framework device package on a machine with the CANN 9.2 toolkit:

```sh
cmake -S tilelang/ascend/distributed/aicpu_backend -B build/aicpu-transfer \
    -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"
cmake --build build/aicpu-transfer --target aicpu_package
```

The output is `build/aicpu-transfer/aicpu_tilelang_transfer.tar.gz`, containing
the AArch64 ELF and its SHA-256 manifest. Platform packaging should sign that
archive with the platform's CANN package signing service using
`aicpu_tilelang_check_cfg.xml`. Deploy the signed archive in the toolkit's
`opp/vendors/cust/aicpu/kernel` and the supplied `libtilelang_aicpu_kernel.json`
in `opp/vendors/cust/aicpu/config`. Register the package with the platform's
AICPU package loader using the same policy as other custom communication ops
(package path `opp/vendors/cust/aicpu/kernel`, install path 2). The package and
HCOMM device library must be available on every participating NPU.

The runtime loads the JSON with the standard ACL AICPU kernel loader and
builds the small host binding in the user cache on first use. The team
owns the stream, communication resources, instruction buffers and ACL binary
handle. A team requires a direct UB_CTP link to each participating peer.

Run the two-rank numerical example from the repository root:

```sh
torchrun --standalone --nproc-per-node=2 examples/ascend/aicpu_collectives.py
```

It checks AllGather and a ring exchange against independent tensor references
on a non-default stream, including a nonzero source offset and untouched output
guards.
