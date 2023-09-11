class ModelParameters:
    def __init__(self, loss_function, loss_function_derivative, initial_learning_rate=0.1,
                 final_learning_rate=0.01, learning_rate_steps=10, epochs=100):
        self.epochs = epochs
        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative
        self.learning_rate_update_number = learning_rate_steps
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
