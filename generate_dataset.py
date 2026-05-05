import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# Number of samples
N = 5000

# Feature Categories
stages_list = ["NEW", "CONTACTED", "QUALIFIED", "PROPOSITION", "NEGOCIATION", "WON", "LOST"]
industries_list = ["TECHNOLOGY", "FINANCE", "HEALTHCARE", "EDUCATION", "OTHER"]
sizes_list = ["SMALL", "MEDIUM", "LARGE"]

# Generate independent features
industries = np.random.choice(industries_list, N, p=[0.3, 0.2, 0.15, 0.1, 0.25])
sizes = np.random.choice(sizes_list, N, p=[0.5, 0.35, 0.15])

deal_values = []
for size in sizes:
    if size == "SMALL":
        deal_values.append(np.random.lognormal(mean=7.5, sigma=0.5))  # ~ $1800
    elif size == "MEDIUM":
        deal_values.append(np.random.lognormal(mean=8.5, sigma=0.6))  # ~ $5000
    else:
        deal_values.append(np.random.lognormal(mean=10.0, sigma=0.7)) # ~ $22000

deal_values = np.round(deal_values, 2)

# Base probability of winning
base_prob = np.zeros(N) + 0.1

# Adjust probability based on industry and size
for i in range(N):
    if industries[i] == "TECHNOLOGY":
        base_prob[i] += 0.05
    elif industries[i] == "FINANCE":
        base_prob[i] += 0.03
    
    if sizes[i] == "LARGE":
        base_prob[i] -= 0.05 # Harder to close
    elif sizes[i] == "SMALL":
        base_prob[i] += 0.05

# Generate engagement features based on target (to create correlation)
won = np.zeros(N, dtype=int)
calls = np.zeros(N, dtype=int)
emails = np.zeros(N, dtype=int)
meetings = np.zeros(N, dtype=int)
stages = []

for i in range(N):
    # Determine win status first based on adjusted probability
    is_won = np.random.rand() < base_prob[i]
    
    if is_won:
        won[i] = 1
        calls[i] = int(np.random.normal(loc=12, scale=4))
        emails[i] = int(np.random.normal(loc=20, scale=5))
        meetings[i] = int(np.random.normal(loc=4, scale=2))
        # If won, stage is likely further along (or WON itself, but let's say this is historical data at time of scoring)
        # We'll set the stage to a late stage just before winning, or WON.
        stages.append(np.random.choice(["PROPOSITION", "NEGOCIATION", "WON"], p=[0.2, 0.3, 0.5]))
    else:
        won[i] = 0
        calls[i] = int(np.random.normal(loc=3, scale=3))
        emails[i] = int(np.random.normal(loc=5, scale=4))
        meetings[i] = int(np.random.normal(loc=0.5, scale=1))
        # If lost, stage could be early or LOST
        stages.append(np.random.choice(["NEW", "CONTACTED", "QUALIFIED", "LOST"], p=[0.4, 0.3, 0.1, 0.2]))
        
# Ensure non-negative engagement
calls = np.clip(calls, 0, 50)
emails = np.clip(emails, 0, 100)
meetings = np.clip(meetings, 0, 20)

# Build DataFrame
df = pd.DataFrame({
    'deal_value': deal_values,
    'calls': calls,
    'emails': emails,
    'meetings': meetings,
    'stage': stages,
    'industry': industries,
    'company_size': sizes,
    'won': won
})

# Save to CSV
df.to_csv("lead_scoring_dataset.csv", index=False)
print("Dataset generated successfully with 5000 rows: lead_scoring_dataset.csv")
