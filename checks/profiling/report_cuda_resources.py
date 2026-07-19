#!/usr/bin/env python3
"""Report CUDA kernel resource usage from compiled extension modules."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess


FUNCTION_RE = re.compile(r"^ Function (?P<signature>.+):$")
RESOURCE_RE = re.compile(
    r"REG:(?P<registers>\d+)\s+STACK:(?P<stack>\d+)\s+"
    r"SHARED:(?P<shared>\d+)\s+LOCAL:(?P<local>\d+)"
)

KERNEL_KINDS = {
    "m2l": ("CountInteractionsAndM2L",),
    "m2m": ("SummarizeMultipoles",),
    "l2l": ("TranslateLocalToLocal",),
}


@dataclass(frozen=True)
class KernelResources:
    module: str
    kernel: str
    kind: str
    dtype: str | None
    registers: int
    stack: int
    local: int
    shared: int

    @property
    def spilled(self) -> bool:
        # CUDA stack frames and explicit local allocations both reside in local
        # memory. Either is therefore relevant when looking for register spills.
        return self.stack > 0 or self.local > 0


def default_modules(repo_root: Path) -> list[Path]:
    modules = sorted((repo_root / "build").glob("*.so"))
    if not modules:
        raise FileNotFoundError(
            "No build/*.so files found. Build the extensions before running this report."
        )
    return modules


def tool_output(tool: str, *args: str, stdin: str | None = None) -> str:
    executable = shutil.which(tool)
    if executable is None:
        raise FileNotFoundError(f"Required tool not found on PATH: {tool}")
    return subprocess.run(
        [executable, *args],
        input=stdin,
        check=True,
        text=True,
        capture_output=True,
    ).stdout


def short_kernel_name(signature: str) -> str:
    name = signature.split("(", 1)[0]
    return name.removeprefix("void ")


def kernel_kind(kernel: str) -> str:
    for kind, prefixes in KERNEL_KINDS.items():
        if kernel.startswith(prefixes):
            return kind
    return "other"


def kernel_dtype(kernel: str) -> str | None:
    matches = re.findall(r"(?:^|[<, ])(float|double)(?=[>, ])", kernel)
    return matches[-1] if matches else None


def read_resources(module: Path) -> tuple[list[KernelResources], set[str]]:
    dumped = tool_output("cuobjdump", "--dump-resource-usage", str(module))
    demangled = tool_output("c++filt", stdin=dumped)
    architectures = set(re.findall(r"^arch = (\S+)$", dumped, re.MULTILINE))

    rows: list[KernelResources] = []
    lines = demangled.splitlines()
    for index, line in enumerate(lines):
        function = FUNCTION_RE.match(line)
        if function is None:
            continue
        if index + 1 == len(lines):
            raise RuntimeError(f"No resource record after: {line}")
        usage = RESOURCE_RE.search(lines[index + 1])
        if usage is None:
            raise RuntimeError(f"No resource record after: {line}")

        kernel = short_kernel_name(function.group("signature"))
        rows.append(
            KernelResources(
                module=module.name.split(".cpython-", 1)[0],
                kernel=kernel,
                kind=kernel_kind(kernel),
                dtype=kernel_dtype(kernel),
                registers=int(usage.group("registers")),
                stack=int(usage.group("stack")),
                local=int(usage.group("local")),
                shared=int(usage.group("shared")),
            )
        )
    return rows, architectures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "modules",
        nargs="*",
        type=Path,
        help="shared libraries to inspect (default: all build/*.so)",
    )
    kinds = parser.add_argument_group("kernel filters")
    kinds.add_argument("--m2l", action="store_true", help="show M2L kernels")
    kinds.add_argument("--m2m", action="store_true", help="show M2M kernels")
    kinds.add_argument("--l2l", action="store_true", help="show L2L kernels")
    precisions = parser.add_argument_group("precision filters")
    precisions.add_argument(
        "--float", action="store_true", dest="show_float", help="show float kernels"
    )
    precisions.add_argument(
        "--double", action="store_true", dest="show_double", help="show double kernels"
    )
    parser.add_argument(
        "--spilled",
        action="store_true",
        help="only show kernels with nonzero stack or local memory",
    )
    parser.add_argument(
        "--top",
        type=int,
        metavar="N",
        help="show the N kernels with the largest register count",
    )
    args = parser.parse_args()
    if args.top is not None and args.top < 1:
        parser.error("--top must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    modules = [module.resolve() for module in args.modules] or default_modules(repo_root)

    rows: list[KernelResources] = []
    architectures: set[str] = set()
    for module in modules:
        module_rows, module_architectures = read_resources(module)
        rows.extend(module_rows)
        architectures.update(module_architectures)

    selected_kinds = {
        kind for kind in KERNEL_KINDS if getattr(args, kind)
    }
    if selected_kinds:
        rows = [row for row in rows if row.kind in selected_kinds]
    selected_dtypes = {
        dtype
        for dtype, selected in (
            ("float", args.show_float),
            ("double", args.show_double),
        )
        if selected
    }
    if selected_dtypes:
        rows = [row for row in rows if row.dtype in selected_dtypes]
    if args.spilled:
        rows = [row for row in rows if row.spilled]

    if args.top is not None:
        rows.sort(
            key=lambda row: (row.registers, row.stack, row.local, row.kernel),
            reverse=True,
        )
        rows = rows[: args.top]
    else:
        rows.sort(key=lambda row: (row.kind, row.kernel, row.module))

    if not rows:
        raise RuntimeError("No kernels matched the requested filters")

    print(f"modules: {', '.join(str(module) for module in modules)}")
    print(tool_output("cuobjdump", "--version").splitlines()[-1])
    print(f"architectures: {', '.join(sorted(architectures)) or 'unknown'}")
    print()
    print(
        f"{'module':<19} {'kind':<5} {'dtype':<6} {'regs':>4} "
        f"{'stack_B':>8} {'local_B':>8} {'shared_B':>9}  kernel"
    )
    for row in rows:
        print(
            f"{row.module:<19} {row.kind:<5} {(row.dtype or '-'):6} "
            f"{row.registers:4d} {row.stack:8d} {row.local:8d} "
            f"{row.shared:9d}  {row.kernel}"
        )


if __name__ == "__main__":
    main()
