import itertools
import subprocess
import sys


def main():
    chunk_targets = [20, 40, 60, 80, 100]
    overlaps = [10, 20, 30, 40, 50]

    combos = [(t, o) for t in chunk_targets for o in overlaps if o < t]
    print(f"Running {len(combos)} configurations (llama3.2:3b, temp=0.5)...")

    for t, o in combos:
        out = f"results_llama_chunk{t}_ov{o}.jsonl"
        cmd = [
            sys.executable,
            "llama_ner_multi_strategy.py",
            "--develop_jsonl",
            "./datasets/ncbi_develop.jsonl",
            "--out_pred",
            out,
            "--limit",
            "5",
            "--confidence_threshold",
            "0.3",
            "--strategies",
            "llama32_optimized",
            "--s1_target",
            str(t),
            "--s1_overlap",
            str(o),
            "--s1_temp",
            "0.5",
        ]
        print("\n>>>", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed for chunk={t}, overlap={o}: {e}")

    print("\nAll runs finished.")


if __name__ == "__main__":
    main()
