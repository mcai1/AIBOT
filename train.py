# Arda Mavi
import os
import numpy
from get_dataset import get_dataset
from get_model import get_model, save_model
from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

epochs = 50
batch_size = 5

def train_model(model, X, X_test, Y, Y_test):
    checkpoints = []
    if not os.path.exists('Data/Checkpoints/'):
        os.makedirs('Data/Checkpoints/')

    checkpoints.append(ModelCheckpoint('Data/Checkpoints/best_weights.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
    checkpoints.append(TensorBoard(log_dir='Data/Checkpoints/./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))

    model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test), shuffle=True, callbacks=checkpoints)

    return model

def main():
    config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    # config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
    X, X_test, Y, Y_test = get_dataset()
    model = get_model()
    print("got model")
    model = train_model(model, X, X_test, Y, Y_test)
    print("saving model")
    save_model(model)
    return model

if __name__ == '__main__':
    main()
