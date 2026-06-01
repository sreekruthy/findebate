import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# folder containing outputs
OUTPUT_FOLDER = "outputs"

results = []

# loop through files
for file in os.listdir(OUTPUT_FOLDER):

    if file.endswith(".json"):

        path = os.path.join(OUTPUT_FOLDER, file)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        try:
            debate = data["debate_result"]

            stance = debate["overall_stance"]
            conviction = debate["overall_conviction"]

            debate_log = debate["_debate_log"]["steps"]

            safety_passed = True

            trust_length = 0
            skeptic_length = 0
            leader_length = 0

            for step in debate_log:

                if step["step"] == "trust_phase":
                    trust_length = step.get("length", 0)

                if step["step"] == "skeptic_phase":
                    skeptic_length = step.get("length", 0)

                if step["step"] == "leader_phase":
                    leader_length = step.get("length", 0)

                if "FAILED" in str(step.get("result", "")):
                    safety_passed = False

            results.append({
                "file": file,
                "stance": stance,
                "conviction": conviction,
                "safety_passed": safety_passed,
                "trust_length": trust_length,
                "skeptic_length": skeptic_length,
                "leader_length": leader_length
            })

        except Exception as e:
            print(f"Error in {file}: {e}")

# dataframe
df = pd.DataFrame(results)

# save csv
df.to_csv("evaluation_results.csv", index=False)

print("\nEvaluation Summary\n")

print(df.head())

# STANCE COUNTS
stance_counts = df["stance"].value_counts()

plt.figure(figsize=(6,6))
stance_counts.plot(kind="pie", autopct="%1.1f%%")
plt.title("Stance Distribution")
plt.ylabel("")
plt.savefig("charts/stance_distribution.png")

# SAFETY
safety_counts = df["safety_passed"].value_counts()

plt.figure(figsize=(6,4))
safety_counts.plot(kind="bar")
plt.title("Debate Safety Results")
plt.savefig("charts/safety_results.png")

# PHASE LENGTHS
avg_lengths = [
    df["trust_length"].mean(),
    df["skeptic_length"].mean(),
    df["leader_length"].mean()
]

labels = ["Trust", "Skeptic", "Leader"]

plt.figure(figsize=(6,4))
plt.bar(labels, avg_lengths)
plt.title("Average Debate Phase Lengths")
plt.savefig("charts/phase_lengths.png")

print("\nCharts saved in charts/ folder")
print("CSV saved as evaluation_results.csv")