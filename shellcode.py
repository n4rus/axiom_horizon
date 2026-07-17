#!/usr/bin/env python3
"""x86-64 shellcode execution: code executes because you READ it.

Memory is allocated RW (read+write), shellcode is written, then mprotect
removes write permission (PROT_READ | PROT_EXEC). The code can only be READ
and EXECUTED — not modified. The CPU reads the bytes as instructions because
the page is marked executable: execute-on-read.

Usage:
    from shellcode import SelfShellCode
    ssc = SelfShellCode()
    addr, add_fn = ssc.make_function(shellcode_add())
    print(add_fn(2, 3))  # 5
"""

import ctypes, os, sys

PAGE_SIZE = 4096
PROT_READ = 0x01
PROT_WRITE = 0x02
PROT_EXEC = 0x04

libc = ctypes.CDLL('libc.so.6', use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mprotect.restype = ctypes.c_int
libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.munmap.restype = ctypes.c_int
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
MAP_PRIVATE = 0x02
MAP_ANONYMOUS = 0x20


def shellcode_add() -> bytes:
    """int add(int a, int b) — x86-64 SysV ABI: edi=1st, esi=2nd, eax=ret"""
    return bytes([0x89, 0xf8, 0x01, 0xf0, 0xc3])


def shellcode_mul() -> bytes:
    """int mul(int a, int b)"""
    return bytes([0x89, 0xf8, 0x0f, 0xaf, 0xc6, 0xc3])


def shellcode_identity() -> bytes:
    """int identity(int a) — returns first arg unchanged"""
    return bytes([0x89, 0xf8, 0xc3])


class SelfShellCode:
    """Allocate RX memory, load shellcode, wrap as a ctypes callable."""

    def __init__(self):
        self._pages: list[tuple[int, int]] = []
        self._fns: list = []

    def _ensure_page(self, size: int) -> tuple[int, int]:
        size = (size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)
        addr = libc.mmap(0, size,
                         PROT_READ | PROT_WRITE | PROT_EXEC,
                         MAP_PRIVATE | MAP_ANONYMOUS,
                         -1, 0)
        if addr == ctypes.c_void_p(-1).value:
            err = ctypes.get_errno()
            raise OSError(err, f'mmap: {os.strerror(err)}')
        self._pages.append((addr, size))
        return addr, size

    def load(self, code: bytes) -> int:
        """Load shellcode into a fresh RX page. Returns address."""
        size = max(len(code), PAGE_SIZE)
        addr, actual = self._ensure_page(size)
        ctypes.memmove(addr, code, len(code))
        ret = libc.mprotect(addr, actual, PROT_READ | PROT_EXEC)
        if ret != 0:
            err = ctypes.get_errno()
            raise OSError(err, f'mprotect: {os.strerror(err)}')
        return addr

    def make_function(self, code: bytes,
                      restype=ctypes.c_int,
                      argtypes=None) -> tuple[int, ctypes.CFUNCTYPE]:
        """Load shellcode and return (address, callable)."""
        if argtypes is None:
            argtypes = [ctypes.c_int, ctypes.c_int]
        addr = self.load(code)
        fn = ctypes.CFUNCTYPE(restype, *argtypes)(addr)
        self._fns.append(fn)
        return addr, fn

    def __del__(self):
        for addr, size in self._pages:
            try:
                libc.munmap(addr, size)
            except Exception:
                pass

    def __len__(self):
        return len(self._pages)


# ─── Tests ─────────────────────────────────

def test():
    ssc = SelfShellCode()
    passed = 0

    addr, add_fn = ssc.make_function(shellcode_add())
    assert add_fn(2, 3) == 5
    assert add_fn(100, 200) == 300
    assert add_fn(-1, 1) == 0
    passed += 3
    print(f'  ✓ add() — 3 assertions')

    addr2, mul_fn = ssc.make_function(shellcode_mul())
    assert mul_fn(6, 7) == 42
    assert mul_fn(0, 100) == 0
    assert mul_fn(-2, 3) == -6
    passed += 3
    print(f'  ✓ mul() — 3 assertions')

    addr3, id_fn = ssc.make_function(shellcode_identity(),
                                      argtypes=[ctypes.c_int])
    assert id_fn(42) == 42
    assert id_fn(-7) == -7
    passed += 2
    print(f'  ✓ identity() — 2 assertions')

    assert addr not in (addr2, addr3)
    assert len(ssc) == 3
    passed += 2
    print(f'  ✓ 3 separate pages')

    print(f'\n  {passed} assertions passed.')
    return passed


if __name__ == '__main__':
    print(f'  Platform: {sys.platform} ({os.uname().machine})')
    test()
