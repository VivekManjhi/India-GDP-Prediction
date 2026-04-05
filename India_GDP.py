import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# 1) Load Data
df = pd.read_csv("India_GDP_1960-2025.csv")

df = df.rename(columns={
    "GDP in (Billion) $": "GDP",
    "Growth %": "Growth_pct"
})

# Clean GDP
df["GDP"] = df["GDP"].astype(str).str.replace(",", "")
df["GDP"] = pd.to_numeric(df["GDP"], errors="coerce") * 1e9

# Convert Year
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Remove invalid rows
df = df.dropna(subset=["Year", "GDP"])

# 🔥 ================== DATA UPDATE ==================

# Sort data
df = df.sort_values(by="Year")

# Fix 1960 growth
df.loc[df["Year"] == 1960, "Growth_pct"] = 3.72

# Add new rows (2022–2025)
new_data = pd.DataFrame({
    "Year": [2022, 2023, 2024, 2025],
    "GDP": [3385e9, 3500e9, 3700e9, 3900e9],
    "Growth_pct": [7.0, 7.2, 6.8, 6.5]
})

# Remove duplicates
df = df[~df["Year"].isin(new_data["Year"])]

# Merge
df = pd.concat([df, new_data], ignore_index=True)

# Sort again
df = df.sort_values(by="Year")

# 🔥 ==================================================

# 2) Prepare data
X = df[["Year"]].values
y = df["GDP"].values

# 3) Model
model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

model.fit(X, y)

# ✅ 2026–2040 → Prediction
future_pred = np.arange(2026, 2041).reshape(-1,1)
gdp_pred = model.predict(future_pred)

# 🔮 2041–2050 → Scenario
future_scn = np.arange(2041, 2051).reshape(-1,1)

last_gdp = gdp_pred[-1]
growth_rate = 0.05

scenario_gdp = []
current = last_gdp

for _ in future_scn:
    current = current * (1 + growth_rate)
    scenario_gdp.append(current)

scenario_gdp = np.array(scenario_gdp)

# Smooth curve till 2040
all_years = np.linspace(df["Year"].min(), 2040, 200).reshape(-1,1)
predicted_curve = model.predict(all_years)

# 4) Plot
plt.figure(figsize=(7,4))

# Actual
plt.scatter(df["Year"], df["GDP"]/1e12, label="Actual GDP")

# Prediction line
plt.plot(all_years, predicted_curve/1e12, label="Future Prediction (2026–2040)")

# Prediction points
plt.scatter(future_pred, gdp_pred/1e12, label="Predicted Points")

# Scenario
plt.plot(future_scn, scenario_gdp/1e12, linestyle='dashed',
         label="Scenario (2041–2050)")

plt.xlabel("Year")
plt.ylabel("GDP (Trillion USD)")
plt.title("India GDP: Actual vs Prediction vs Scenario")
plt.legend()
plt.grid(True)

# 🔥 IMPORTANT: Save Graph
plt.savefig("GDP_Output.png", dpi=300)

plt.show()

# 5) Print
print("\nPrediction (2026–2040):")
for y, g in zip(future_pred.flatten(), gdp_pred):
    print(f"{y} : {g/1e12:.2f} Trillion USD")

print("\nScenario (2041–2050):")
for y, g in zip(future_scn.flatten(), scenario_gdp):
    print(f"{y} : {g/1e12:.2f} Trillion USD")