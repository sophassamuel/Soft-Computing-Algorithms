import numpy as np
import skfuzzy as fuzz

# Define the universe of discourse
x_height = np.arange(0, 201, 1)
x_age = np.arange(0, 201, 1)
x_health = np.arange(0, 101, 1)

# Define the fuzzy sets
height_poor = fuzz.trimf(x_height, [0, 0, 100])
height_average = fuzz.trimf(x_height, [0, 100, 200])
height_good = fuzz.trimf(x_height, [100, 200, 200])

age_poor = fuzz.trimf(x_age, [0, 0, 100])
age_average = fuzz.trimf(x_age, [0, 100, 200])
age_good = fuzz.trimf(x_age, [100, 200, 200])

# Input specific values for height and age
test_height = 199
test_age = 45

# Define the rules (Sugeno-style)
def rule1(height, age):
    return 0.2 * height + 0.8 * age

def rule2(height, age):
    return 0.5 * height + 0.5 * age

def rule3(height, age):
    return 0.8 * height + 0.2 * age

# Calculate the rule activations
activation_rule1 = np.fmin(fuzz.interp_membership(x_height, height_poor, test_height),
                            fuzz.interp_membership(x_age, age_poor, test_age))
activation_rule2 = np.fmin(fuzz.interp_membership(x_height, height_average, test_height),
                            fuzz.interp_membership(x_age, age_average, test_age))
activation_rule3 = np.fmin(fuzz.interp_membership(x_height, height_good, test_height),
                            fuzz.interp_membership(x_age, age_good, test_age))

# Calculate the weighted output
output = (rule1(test_height, test_age) * activation_rule1 +
          rule2(test_height, test_age) * activation_rule2 +
          rule3(test_height, test_age) * activation_rule3)

# Normalize the output
total_activation = activation_rule1 + activation_rule2 + activation_rule3
output = np.sum(output) / np.sum(total_activation)

# Print the result
print(f"For Height: {test_height} cm and Age: {test_age} years, the Predicted Health is: {output}")

if output > 80:
  print("Health : Good")
elif output < 80 or output > 40:
  print("Health : Normal")
else:
  print("Health : Low")