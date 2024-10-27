import pandas as pd

# Example dataset: math problems with step-by-step solutions
data = [
    {"question": "Solve for x: 2x + 3 = 7", "answer": "Step 1: Subtract 3 from both sides. 2x = 4.\nStep 2: Divide both sides by 2. x = 2."},
    {"question": "Find the derivative of x^2", "answer": "Step 1: Apply the power rule.\nStep 2: The derivative is 2x."},
    {"question": "Solve for y: 3y - 5 = 10", "answer": "Step 1: Add 5 to both sides. 3y = 15.\nStep 2: Divide both sides by 3. y = 5."},
    {"question": "Find the derivative of x^3", "answer": "Step 1: Apply the power rule.\nStep 2: The derivative is 3x^2."},
]

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("math_dataset.csv", index=False)

print("Dataset created and saved as 'math_dataset.csv'.")
