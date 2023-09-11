import time
import TestLibraryFunction as tests

start = time.time()

# tests.test_dense_1_hidden('relu', 'norm_2')
tests.test_auto_encoder('relu', 'norm_2')

tick = time.time()
print('Execution time : {0}'.format(tick - start))