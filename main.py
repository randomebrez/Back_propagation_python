import time
import numpy as np
import Manager as manager
import Tools.Computations as computer
# 1 - Program
# 2 - Tests
# 0 - ExecConsole


launch = 1
# Program
if launch == 1:
    start = time.time()
    # Get dataset
    ds_inputs, ds_targets = manager.get_dataset(40996, 'class', 255, batch_size=100)

    # Train & Test network (input size & output size are calculated from dataset shape
    hidden_layers = [800, 500]
    epochs = 50
    initial_learning_rate = 0.1
    final_learning_rate = 0.01
    learning_rate_steps = 20
    manager.back_prop_batch_on_dataset(ds_inputs, ds_targets, hidden_layers,
                                       computer.cross_entropy, computer.distance_get,
                                       initial_learning_rate, final_learning_rate, learning_rate_steps, epochs=epochs)
    tick = time.time()
    print('Execution time batch_size = 100 : {0}'.format(tick - start))


# test
elif launch == 2:
    print("")
    x = np.arange(1, 5)
    print(x)
    print(x ** 2)


# Exec console
else:
    print("")
