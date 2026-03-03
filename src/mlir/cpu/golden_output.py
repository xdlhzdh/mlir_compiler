import torch


def main():
    a = torch.eye(4, dtype=torch.float32)
    b = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=torch.float32,
    )
    c = torch.matmul(a, b)
    ok = torch.allclose(c, b, atol=1e-4, rtol=0.0)
    flat = c.reshape(-1).tolist()
    with open("golden_ref.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(f"{x:.6f}" for x in flat) + "\n")
    print("python first row:", " ".join(f"{x:.1f}" for x in c[0].tolist()))
    print("python matmul check:", "PASS" if ok else "FAIL")
    print("golden file: golden_ref.txt")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
