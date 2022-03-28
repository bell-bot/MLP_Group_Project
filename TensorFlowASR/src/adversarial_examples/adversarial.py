import numpy as np

def adversarial(model, input, goal_output, actual_output, keyphrase):
    """
    model: a trained model for keyword search in audio
    input: an audio file in mfcc format
    goal_output: 1 or 0 (recognized or not)
    actual_output: 1 or 0 (recognized or not)
    keyphrase: the phrase we want to detect in the audio (or not)
    """
    input_dim = input.size()

    # Create a random input vector sampled from a normal distribution
    x = np.random.normal(.5,.3,(784,1))

    # Gradient descent on the input
    for i in range(steps):
        # Calculate the derivative        
        d = input_derivative(net,x,goal)
        # The GD update on x        
        x -= eta * d