#!/usr/bin/env python
# coding: utf-8

# In[112]:


import random
import math

# Define the problem variables
areas = ['X1', 'X2', 'X3', 'X4']
crops = ['corn', 'wheat', 'soybeans', 'sunflowers']
soil_types = {
    'X1': ['sandy', ['sunflowers', 'soybeans']],
    'X2': ['loamy', ['corn', 'wheat', 'soybeans']],
    'X3': ['clay', ['wheat', 'soybeans']],
    'X4': ['mixture', ['corn', 'wheat', 'soybeans']]
}
min_percentages = {'corn': 0.2, 'wheat': 0.3, 'soybeans': 0.1, 'sunflowers': 0.1}
max_percentages = {'corn': 1, 'wheat': 1, 'soybeans': 0.4, 'sunflowers': 1}
total_percentage = 1


# In[113]:


def evaluate_solution(solution):
    
    # Calculate the percentage of each crop planted
    percentages = {crop: 0 for crop in crops}
    for area in areas:
        crop = solution[area]
        percentages[crop] += 1 / len(areas)
        
    # Calculate the yield for each crop
    yields = {crop: 0 for crop in crops}
    for area in areas:
        crop = solution[area]
        soil_type = soil_types[area][0]
        if crop in soil_types[area][1]:
            yields[crop] += 1 / len(areas)
        if soil_type == 'sandy' and crop in ['sunflowers', 'soybeans']:
            yields[crop] += 1 / len(areas)
        if soil_type == 'loamy' and crop in ['corn', 'wheat', 'soybeans']:
            yields[crop] += 1 / len(areas)
        if soil_type == 'clay' and crop in ['wheat', 'soybeans']:
            yields[crop] += 1 / len(areas)
        if soil_type == 'mixture' and crop in ['corn', 'wheat', 'soybeans']:
            yields[crop] += 1 / len(areas)
            
             # Check for unique assignment constraint
        if len(set(solution.values())) != len(areas):
        # If the constraint is violated, return a very low cost
             return -float('inf')
            
            
    # Calculate the total yield
    total_yield = sum(yields.values())
    # Calculate the total percentage of all crops planted
    total_planted = sum(percentages.values())
    # Calculate the penalty for not meeting the minimum percentage requirements
    min_penalty = sum([max(0, min_percentages[crop] - percentages[crop]) for crop in crops])
    # Calculate the penalty for exceeding the maximum percentage requirements
    max_penalty = sum([max(0, percentages[crop] - max_percentages[crop]) for crop in crops])
    # Calculate the penalty for not planting exactly 100% of the land
    land_penalty = abs(total_planted - total_percentage)
    # Return the total yield minus the penalties
    return total_yield - min_penalty - max_penalty - land_penalty


# In[114]:


def generate_neighbor(solution):
    # Select two random areas to swap
    area1, area2 = random.sample(areas, 2)
    # Swap the crop types assigned to the two areas
    new_solution = solution.copy()
    new_solution[area1], new_solution[area2] = solution[area2], solution[area1]
    return new_solution


# In[115]:



def simulated_annealing(initial_solution, temperature, cooling_rate):
    current_solution = initial_solution
    current_cost = evaluate_solution(current_solution)
    best_solution = current_solution
    best_cost = current_cost
    while temperature > 1e-5:
        # Generate a random neighbor solution
        neighbor_solution = generate_neighbor(current_solution)
        neighbor_cost = evaluate_solution(neighbor_solution)
        # Accept the neighbor solution with a certain probability
        delta_cost = neighbor_cost - current_cost
        try:
            acceptance_probability = math.exp(delta_cost / temperature)
        except OverflowError:
            acceptance_probability = float('inf')
        if delta_cost < 0 or random.random() < acceptance_probability:
            current_solution = neighbor_solution
            current_cost = neighbor_cost
        # Update the best solution if necessary
        if current_cost > best_cost:
            best_solution = current_solution
            best_cost = current_cost
        # Cool down the temperature
        temperature *= cooling_rate
    return best_solution, best_cost


# In[116]:


initial_solution = {area: random.choice(crops) for area in areas}
temperature = 1000
cooling_rate = 0.95

best_solution, best_cost = simulated_annealing(initial_solution, temperature, cooling_rate)

print("Best solution: ", best_solution)
print("Best cost: ", best_cost)


# In[ ]:




