import os
from utils import parse_params

filename = input("enter the filename of the log: ").strip()


with open(os.path.join("optuna_logs", filename), "r") as file:
    raw = file.read()


runs_dict = {}
for run in raw.split("#####################################################"):
    if run == "\n":
        continue
    loss = float(run.split(",")[0].split(": ")[1])
    params = parse_params(run.split("Best value: ")[0])
    runs_dict[loss] = params

for i, key in enumerate(sorted(runs_dict)):
    print(f"{i+1} lowest loss ({key}) achieved with params:\n{runs_dict[key]}\n")
