def read_golden(path: str):
    with open(path, "r", encoding="utf-8") as f:
        nums = [float(x) for x in f.read().strip().split()]
    if len(nums) != 16:
        raise ValueError(f"golden size is {len(nums)}, expected 16")
    return nums


def read_riscv_output(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("riscv_out:"):
                parts = line.split(":", 1)[1].strip().split()
                nums = [float(x) for x in parts]
                if len(nums) != 16:
                    raise ValueError(f"riscv_out size is {len(nums)}, expected 16")
                return nums
    raise ValueError("riscv_out line not found")


def main():
    golden = read_golden("golden_ref.txt")
    riscv = read_riscv_output("riscv_run.txt")
    ok = all(abs(a - b) < 1e-4 for a, b in zip(golden, riscv))
    print("compare golden vs riscv:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
