import numpy as np
import random

# Set up the possible states
states = ('Happy', 'Sad')

# Set the probabilities for the initial states
initialStateProb = np.zeros(2)
initialStateProb = [0.6, 0.4]

# Observation probability table
# Happy: 0.9 to stay happy, 0.1 to become sad
# Sad: 0.8 to stay sad, 0.2 to become happy
observationProb = np.zeros((2, 2))
observationProb = [0.9, 0.1], [0.2, 0.8]
observationProb = np.asarray(observationProb)

# Generate float from 0.0 - 1.0
randFloat = random.uniform(0.0, 1.0)

# 0.0 - 0.6 is happy
if (randFloat <= initialStateProb[0]):
    initialState = states[0]

# 0.6 - 1.0 is sad
else:
    initialState = states[1]

print("Initial state:", initialState)

observations = ('Cooking', 'Crying', 'Sleeping', 'Socializing', 'Watching TV')
