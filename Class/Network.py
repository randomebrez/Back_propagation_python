import numpy as np


class Network:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_layer_index = 0
        self.layers = []

    # Training
    def train(self, input_batches, target_batches, train_model):
        # results
        batch_mean_cost_fct = []

        # setup learning_rate evolution
        current_learning_rate = train_model.initial_learning_rate
        decrease_rate = 0
        learning_rate_decrease_threshold = len(input_batches)
        if train_model.learning_rate_update_number > 0:
            learning_rate_decrease_threshold = len(input_batches) / train_model.learning_rate_update_number
            decrease_rate = (train_model.initial_learning_rate - train_model.final_learning_rate) / train_model.learning_rate_update_number

        # use backprop on input_batches
        for input_batch, target_batch in zip(input_batches, target_batches):
            if len(batch_mean_cost_fct) > learning_rate_decrease_threshold:
                current_learning_rate -= decrease_rate
                learning_rate_decrease_threshold += learning_rate_decrease_threshold

            # Feed forward
            self.feed_forward(input_batch, True)

            # Compute error
            back_prop_inputs, mean_cost_fct = self.compute_error(target_batch, train_model.loss_function, train_model.loss_function_derivative)
            batch_mean_cost_fct.append(mean_cost_fct)

            # Back propagate & update layers parameters
            self.backprop(back_prop_inputs, current_learning_rate)

        return batch_mean_cost_fct

    # Testing
    def test(self, input_batches, target_batches, test_model, with_details=True):
        result = {'accuracy': [], 'cost_function': [], 'values': [], 'good_answers': []}
        for input_batch, target_batch in zip(input_batches, target_batches):
            # Feed forward
            self.feed_forward(input_batch, False)
            # Compute error
            _, mean_cost_fct = self.compute_error(target_batch, test_model.loss_function, test_model.loss_function_derivative)
            result['cost_function'].append(mean_cost_fct)

            # return percentage of good answer
            if with_details:
                accuracy, mean_output_values, good_answer_mean = self.compute_accuracy(target_batch)
                result['accuracy'].append(accuracy)
                result['values'].append(mean_output_values)
                result['good_answers'].append(good_answer_mean)
        return result

    # Feed forward
    def feed_forward(self, inputs, store=True):
        next_activations = inputs
        for layer in self.layers:
            next_activations = layer.compute(next_activations, store)

    # Backpropagation
    def backprop(self, back_prop_inputs, learning_rate):
        # Compute partial derivative (Error gradient)
        dE_da_i = back_prop_inputs

        for i in range(len(self.layers) - 1, 0, -1):
            dE_da_i = self.layers[i].compute_backward_and_update_weights(dE_da_i, learning_rate)

    # Tools
    def get_outputs(self):
        return self.layers[self.output_layer_index].get_activation_values()

    def compute_error(self, targets, loss_function, loss_function_derivative):
        outputs = self.get_outputs()
        back_prop_inputs = loss_function_derivative(outputs, targets)
        mean_cost_function = loss_function(outputs, targets)
        return back_prop_inputs, mean_cost_function

    def compute_accuracy(self, targets):
        outputs = self.layers[self.output_layer_index].get_activation_values()
        output_indices = np.argmax(outputs, axis=1)
        labels = np.argmax(targets, axis=1)
        good_answer_indices = np.where(output_indices == labels)
        ga = outputs[good_answer_indices]
        gam = np.max(ga, axis=1)
        good_answer_mean = 0
        if len(gam) > 0:
            good_answer_mean = np.mean(gam)
        return 100 * np.mean(output_indices == labels), np.sum(outputs * targets) / targets.shape[0], good_answer_mean

    def clean_layers_cache(self):
        for layer in self.layers:
            layer.clean_cache()
