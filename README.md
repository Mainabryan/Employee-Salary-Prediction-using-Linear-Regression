# Employee-Salary-Prediction-using-Linear-Regression
Practiseing linear regression (ml)
📘 Linear Regression – Notes (Bryan Style 😊)
🔹 1. What is Linear Regression?

Linear Regression is like teaching a computer:

    "If someone has X years of experience, how much Y salary should they earn?"

It's just like drawing the best straight line through your data points to predict future values.
🔹 2. Why Do We Learn It?

    It’s the simplest machine learning model.

    Helps us predict real-life outcomes like salary, house price, etc.

    It teaches the computer to learn a pattern from past data and make future guesses.

🔹 3. Independent vs Dependent Variable

    Independent (X): What we already know or give the machine (e.g., YearsExperience)

    Dependent (y): What we want to predict (e.g., Salary)

X = df.drop('Salary', axis=1)  # remove 'Salary', keep input
y = df['Salary']               # the output/target

🔹 4. Train-Test Split

We split the data into two parts:

    Training set: What the model will learn from.

    Test set: What we’ll use to check how good the model is.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

🔹 5. Model Fitting (Training)

Fitting means:

    “Let the machine study the pattern between years and salary.”

from sklearn.linear_model import LinearRegression

LR = LinearRegression()
LR.fit(X_train, y_train)

🔹 6. Prediction

After training, we ask the model:

    “Okay, you’ve learned. Now predict salaries for new people.”

y_pred = LR.predict(X_test)

🔹 7. Comparing y_test vs y_pred
Real Salary (y_test)	Model's Prediction (y_pred)	Error
What really happened	What the model guessed	How far it was off
🔹 8. Evaluation (How good was the model?)

We use 3 key metrics:

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)  # small = good
mse = mean_squared_error(y_test, y_pred)  # small = good
r2 = r2_score(y_test, y_pred)              # close to 1 = good

🔹 9. Visualization

This helps you see how close your model is to the real world.

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Prediction Line')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary')
plt.legend()
plt.grid(True)
plt.show()

    🔵 Blue dots = model guesses

    🔴 Red line = perfect guesses

    The closer the dots to the line, the better the model!

✅ Summary

    We use Linear Regression to predict numbers using past data.

    We train the model, then test how good it is.

    We look at errors and visuals to understand how well it learned.

    It’s like showing the model examples and asking it to guess new ones.

Would you like me to format this into a neat .md file for GitHub or add next steps like polynomial regression or saving the model?
