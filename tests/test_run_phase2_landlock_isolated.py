from __future__ import annotations

import errno

from scripts.run_phase2_landlock_isolated import (
    AF_UNIX,
    BPF_JMP_JEQ_K,
    BPF_LD_W_ABS,
    BPF_RET_K,
    SECCOMP_DATA_ARG0_OFFSET,
    SECCOMP_RET_ALLOW,
    SECCOMP_RET_ERRNO,
    X86_64_SOCKET_CREATION_SYSCALLS,
    _network_filter_instructions,
)


def test_seccomp_allows_only_unix_domain_socket_creation() -> None:
    instructions = _network_filter_instructions()
    assert instructions[0].code == BPF_LD_W_ABS
    assert instructions[0].k == 0
    assert len(instructions) == 2 + 5 * len(X86_64_SOCKET_CREATION_SYSCALLS)

    for index, syscall_number in enumerate(X86_64_SOCKET_CREATION_SYSCALLS):
        start = 1 + index * 5
        syscall_check, domain_load, unix_check, deny, allow = instructions[start : start + 5]
        assert (syscall_check.code, syscall_check.jt, syscall_check.jf, syscall_check.k) == (
            BPF_JMP_JEQ_K,
            0,
            4,
            syscall_number,
        )
        assert (domain_load.code, domain_load.k) == (
            BPF_LD_W_ABS,
            SECCOMP_DATA_ARG0_OFFSET,
        )
        assert (unix_check.code, unix_check.jt, unix_check.jf, unix_check.k) == (
            BPF_JMP_JEQ_K,
            1,
            0,
            AF_UNIX,
        )
        assert (deny.code, deny.k) == (
            BPF_RET_K,
            SECCOMP_RET_ERRNO | errno.EPERM,
        )
        assert (allow.code, allow.k) == (BPF_RET_K, SECCOMP_RET_ALLOW)

    assert (instructions[-1].code, instructions[-1].k) == (
        BPF_RET_K,
        SECCOMP_RET_ALLOW,
    )
