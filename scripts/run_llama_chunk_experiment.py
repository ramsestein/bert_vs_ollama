import itertools
import subprocess
import sys


def main():
    chunk_targets = [20, 40, 60, 80, 100]
    overlaps = [10, 20, 30, 40, 50]

    combos = [(t, o) for t in chunk_targets for o in overlaps if o < t]
    print(f"Running {len(combos)} chunk configurations for llama3.2:3b (temp=0.5)...")
    print("Testing chunk_target vs chunk_overlap combinations...")

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
            "10",
            "--confidence_threshold",
            "0.3",
            "--strategies",
            "llama32_chunk_experiment",
            "--s1_target",
            str(t),
            "--s1_overlap",
            str(o),
            "--s1_temp",
            "0.5",
        ]
        print(f"\n>>> Running: chunk_target={t}, chunk_overlap={o}")
        print(f"Command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ SUCCESS: {out}")
        except subprocess.CalledProcessError as e:
            print(f"❌ FAILED for chunk={t}, overlap={o}: {e}")

    print("\n🎯 All chunk experiments completed!")
    print("Next: Run 'python analyze_chunk_performance.py' to analyze results")


if __name__ == "__main__":
    main()
