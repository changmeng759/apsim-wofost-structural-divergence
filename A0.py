import pandas as pd

IN_CSV = "/Users/mengchang/Desktop/重要文稿/meta_model_input_44.csv"
OUT_CSV = "/Users/mengchang/Desktop/scenario_table_44_A0.csv"

df = pd.read_csv(IN_CSV)

out = pd.DataFrame({
    "year": df["Year"],
    "sowing": df["sowing"],
    "N_label": df["fertilizer"],
    "yield_APSIM": df["APSIM_Yield"],
    "yield_WOFOST": df["WOFOST_Yield"],
})

out.to_csv(OUT_CSV, index=False)

print(f"Saved to: {OUT_CSV}")
print(out.head())