import sys
import pandas as pd

d = sys.argv[1].rstrip("/")
mb = pd.read_csv(f"{d}/maxima_baseline.csv"); mc = pd.read_csv(f"{d}/maxima_cp.csv")
sb = pd.read_csv(f"{d}/saddles_baseline.csv"); sc = pd.read_csv(f"{d}/saddles_cp.csv")

ok = True
def report(name, cond):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

print(d)
report("maxima identical", set(mb.max_id) == set(mc.max_id))

m = sb.merge(sc, on="saddle_id", how="left", suffixes=("_b", "_c"))
missing = int(m.multiplicity_c.isna().sum())
report(f"every baseline saddle found by cp (missing: {missing})", missing == 0)
lower = int((m.multiplicity_c < m.multiplicity_b).sum())
report(f"cp multiplicity >= baseline (violations: {lower})", lower == 0)

extra = len(set(sc.saddle_id) - set(sb.saddle_id))
mb_tot, mc_tot = int(sb.multiplicity.sum()), int(sc.multiplicity.sum())
print(f"  baseline: {len(sb)} saddles, total mult {mb_tot}")
print(f"  cp:       {len(sc)} saddles, total mult {mc_tot}")
print(f"  extra:    {extra} saddles ({100 * extra / max(len(sc), 1):.2f}% of cp), "
      f"extra mult {mc_tot - mb_tot}")
sys.exit(0 if ok else 1)