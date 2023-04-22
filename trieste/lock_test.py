from functions.lockwood.runlock import runlock
import tensorflow as tf

if __name__ == "__main__":
    a = tf.Variable([[1.5, 1, 1, 1, 1, 1], [1.5, 1.289843, 1.111, 1.2343, 0.173, 0.154]], dtype=tf.float64)
    print(runlock.lockwood_constraint_observer(a))