# %% [code] {"_kg_hide-input":false}
# %% [code] {"_kg_hide-output":false}

import datetime as calender
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import tensorflow as tf

SCRIPT_RUN_DATETIME = calender.datetime.now().strftime("%Y%m%d-%H%M%S")

# Logging Output Location
LOGS_DIR = "./logs/"
TENSORBOARD_LOGGING_DIR = LOGS_DIR + "tboard/"
METRICS_CSV_SAVE_DIR = LOGS_DIR + "./metrics_data/"

# Input Data Location
INPUT_DIR = "../input/"
FASHIONMNIST_INPUT_DIR = INPUT_DIR + "fashionmnist/"

# Saved Model Snapshots Output Location
CHECKPOINTS_DIR = "./checkpoints/"
CHECKPOINTS_FILE = CHECKPOINTS_DIR + "model-accuracy-{val_accuracy:.2f}.hdf5"

# Input CSV Files
TRAINING_DATA_CSV_FILE = FASHIONMNIST_INPUT_DIR + "fashion-mnist_train.csv"
TESTING_DATA_CSV_FILE = FASHIONMNIST_INPUT_DIR + "fashion-mnist_test.csv"

# Output CSV File
METRICS_CSV_SAVE_FILE = METRICS_CSV_SAVE_DIR + SCRIPT_RUN_DATETIME + ".csv"

# Model Training/Fitting Configuration
BATCH_SIZE_MULTIPLIER_GPU = 256
BATCH_SIZE_DIVISOR_TPU = 1024
STEPS_PER_EXECUTION = 128
EPOCHS = 2

# Input Labelled Classes
CLASSES = ('Top/T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# Input Image Metadata
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNELS = 1

IMAGE_MIN_PIXEL_VALUE = 0
IMAGE_MAX_PIXEL_VALUE = 255
IMAGE_PIXEL_VALUE_SCALAR = IMAGE_MAX_PIXEL_VALUE - IMAGE_MIN_PIXEL_VALUE

DEFAULT_VERBOSITY = 1

PLOT_FONT = {'family': 'monospace', 'weight': 'bold', 'size': '14'}


def setup_tensorboard_callback():
    os.makedirs(TENSORBOARD_LOGGING_DIR, exist_ok=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOGGING_DIR, histogram_freq=1)
    return tensorboard_callback


def setup_checkpoint_callback(verbosity=DEFAULT_VERBOSITY):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINTS_FILE, verbose=verbosity,
                                                             save_freq='epoch',
                                                             save_best_only=True, monitor="val_accuracy")
    return checkpoint_callback


def build_callbacks():
    # tensorboard_callback = setup_tensorboard_callback()
    checkpoint_callback = setup_checkpoint_callback()

    callbacks = (
        # tensorboard_callback,
        checkpoint_callback
    )

    return callbacks


def attempt_restore_model(allow_restore=True):
    if not allow_restore:
        return False

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    checkpoints = [CHECKPOINTS_DIR + name for name in os.listdir(CHECKPOINTS_DIR)]

    if checkpoints:
        latest_checkpoint = max(checkpoints)
        print("Restoring from", latest_checkpoint)
        return tf.keras.models.load_model(latest_checkpoint)
    else:
        return False


def load_mnist_data():
    # Usage based upon the following code snippet, but we rolled our own
    # (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_csv = pd.read_csv(TRAINING_DATA_CSV_FILE)
    test_csv = pd.read_csv(TESTING_DATA_CSV_FILE)

    # Load CSV values
    (train_images, train_labels) = (train_csv.drop(['label'], axis=1).values, train_csv.label.values)
    (test_images, test_labels) = (test_csv.drop(['label'], axis=1).values, test_csv.label.values)

    # Reshape images to by 28x28, with a single channel, instead of flattened
    train_images = np.divide(train_images, IMAGE_PIXEL_VALUE_SCALAR).astype("float32", copy=False)
    train_images = train_images.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

    test_images = np.divide(test_images, IMAGE_PIXEL_VALUE_SCALAR).astype("float32", copy=False)
    test_images = test_images.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

    train_labels = tf.keras.utils.to_categorical(y=train_labels, num_classes=len(CLASSES))
    test_labels = tf.keras.utils.to_categorical(y=test_labels, num_classes=len(CLASSES))

    return (train_images, train_labels), (test_images, test_labels)


def data_augmentation_training(image, label):
    augmented_image = image

    # convert image to color
    augmented_image = tf.image.grayscale_to_rgb(augmented_image)

    augmented_image = tf.image.random_flip_left_right(augmented_image)
    augmented_image = tf.image.random_flip_up_down(augmented_image)
    augmented_image = tf.image.random_contrast(augmented_image, 0.05, 0.25)
    augmented_image = tf.image.random_hue(augmented_image, 0.15)

    augmented_image = tf.image.rgb_to_grayscale(augmented_image)

    # convert image back to grayscale

    return augmented_image, label


def data_augmentation_testing(image, label):
    augmented_image = image

    # convert image to color
    augmented_image = tf.image.grayscale_to_rgb(augmented_image)
    augmented_image = tf.image.rgb_to_grayscale(augmented_image)

    return augmented_image, label


def generate_dataset(images, labels, augmentation_func, batch_size):
    # image_data_pyfunc = np.frompyfunc(image_data_augmentation, 1, 1)
    # augmented_images = image_data_pyfunc(images)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    dataset = dataset.map(augmentation_func, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    # This will print every image to console!
    # dataset = dataset.map(print_images)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    dataset = dataset.with_options(options)

    return dataset


def setup_strategy(tpu_resolver=tf.distribute.cluster_resolver.TPUClusterResolver,
                   tpu_strategy=tf.distribute.TPUStrategy,
                   gpu_strategy=tf.distribute.MirroredStrategy):
    try:
        tpu = tpu_resolver()
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tpu_strategy()
    else:
        strategy = gpu_strategy()

    print("Using", type(strategy).__name__, "...")

    return tpu, strategy


def get_max_gpu_compute_capability():
    gpu_devices = tf.config.get_visible_devices('GPU')
    device_details = [tf.config.experimental.get_device_details(device) for device in gpu_devices]

    max_compute_version = 0

    for device_detail in device_details:
        print(device_detail)
        device_compute_capability = device_detail.get("compute_capability", (0, 0))
        max_device_compute_version = device_compute_capability[0]
        max_compute_version = max(max_device_compute_version, max_compute_version)
        print(max_compute_version)

    gpu_exists: bool = len(device_details) >= 1

    return max_compute_version, gpu_exists


def enable_accelerator_specific_optimizations(tpu: tf.distribute.cluster_resolver.TPUClusterResolver):
    max_compute_capability, gpu_exists = get_max_gpu_compute_capability()

    print("Accelerator optimization supports GPU compute version", max_compute_capability)

    if tpu and not gpu_exists:
        print("Exclusive TPU detected: enabling bfloat16 variables")
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    elif gpu_exists and max_compute_capability >= 7:
        print("GPU with TensorCores detected: enabling float16 variables")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    if not tpu:
        print("No TPU detected, enabling global XLA JIT compilation")
        tf.config.optimizer.set_jit(True)
        tf.config.optimizer.set_experimental_options({
            "scoped_allocator_optimization": True,
            "constant_folding": True,
            "layout_optimizer": True,
            "shape_optimization": True,
            "arithmetic_optimization": True,
            "dependency_optimization": True,
            "loop_optimization": True,
            "function_optimization": True,
            "implementation_selector": True
        })


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer='he_uniform', activation='tanh'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(name="flatten", data_format="channels_last"),
        tf.keras.layers.Dropout(0.2, name="dropout"),
        tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform', name="input_discriminator"),
        tf.keras.layers.Dense(len(CLASSES), name="prediction", dtype='float32'),

        # Convolutional Processing
        # tf.keras.Sequential([
        # tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='tanh'),
        # tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
        # tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='tanh'),
        # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # tf.keras.layers.Flatten(name="flatten"),
        # tf.keras.layers.Dense(64, activation="tanh"),
        # tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="prediction")
        # ])
    ])

    return model


def get_strategy_batch_size(tpu, strategy):
    if tpu:
        return BATCH_SIZE_DIVISOR_TPU // strategy.num_replicas_in_sync
    else:
        return BATCH_SIZE_MULTIPLIER_GPU * strategy.num_replicas_in_sync


def plot_confusion_matrix(model, images, labels, title="Confusion Matrix", class_names=CLASSES):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        class_names (array, shape = [n]): String names of the integer classes
        :param title:
        :param class_names:
        :param labels:
        :param images:
        :param model:
    """

    # Use the model to predict the values from the validation dataset.
    label_predicted = np.argmax(model.predict(images), axis=1)
    label_true = np.argmax(labels, axis=1)

    # Calculate the confusion matrix.
    # cm (array, shape = [n, n]): a confusion matrix of integer classes
    cm = sklearn.metrics.confusion_matrix(label_true, label_predicted)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def generate_interleaved_dataframe(metrics, history_items: list):
    epochs_range = range(EPOCHS)

    val_history_items = ["val_" + item for item in history_items]

    training_data = [metrics.history[item] for item in history_items]
    validation_data = [metrics.history[item] for item in val_history_items]

    def interleave(a, b):
        c = a + b
        c[::2] = a
        c[1::2] = b

        return c

    interleaved_data = np.asarray(interleave(training_data, validation_data)).reshape(-1, len(history_items) * 2)

    interleaved_column_names = interleave(history_items, val_history_items)

    interleaved_dataframe = pd.DataFrame(interleaved_data, index=epochs_range, columns=interleaved_column_names)

    return interleaved_dataframe


def plot_per_epoch(metrics, title: str, history_item: str, legend_location="best", training_legend="Training Average",
                   validation_legend="Validation Average"):
    epochs_range = range(EPOCHS)

    val_history_item = "val_" + history_item

    training_metric_data = metrics.history[history_item]
    validation_metric_data = metrics.history[val_history_item]

    plt.plot(epochs_range, training_metric_data, label=training_legend)
    plt.scatter(epochs_range, training_metric_data, edgecolors="r", s=20)

    plt.plot(epochs_range, validation_metric_data, label=validation_legend)
    plt.scatter(epochs_range, validation_metric_data, edgecolors="r", s=20)

    plt.legend(loc=legend_location)
    plt.title(title)
    plt.xlabel("Epochs")


def model_training_main():
    tpu, strategy = setup_strategy()

    batch_size = get_strategy_batch_size(tpu, strategy)
    print("Batch size is", batch_size)

    enable_accelerator_specific_optimizations(tpu)

    (train_images, train_labels), (test_images, test_labels) = load_mnist_data()

    # strategy aware model setup
    with strategy.scope():
        restored_model = attempt_restore_model(allow_restore=False)
        if restored_model:
            model = restored_model
        else:
            model = build_model()
        model.build([batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS])
        model.summary()

    # strategy aware dataset generation
    with strategy.scope():
        train_dataset = generate_dataset(train_images, train_labels, data_augmentation_training, batch_size)
        test_dataset = generate_dataset(test_images, test_labels, data_augmentation_testing, batch_size)

    # strategy aware callback setup
    with strategy.scope():
        callback = build_callbacks()

    with strategy.scope():
        model.compile(optimizer='nadam',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy', tf.keras.metrics.RootMeanSquaredError()],
                      steps_per_execution=STEPS_PER_EXECUTION)
        metrics = model.fit(x=train_dataset, validation_data=test_dataset, epochs=EPOCHS, callbacks=callback,
                            shuffle=True)

    plt.figure(figsize=(30, 30))
    plt.rc("font", **PLOT_FONT)

    metrics_titles = ['Network Accuracy', 'Network Loss', 'Network RMS Error']
    metrics_items = ['accuracy', 'loss', 'root_mean_squared_error']

    for i in range(len(metrics_items)):
        plt.subplot(2, 2, 1 + i)
        plot_per_epoch(metrics, metrics_titles[i], metrics_items[i])

    plt.subplot(2, 2, 4)
    plot_confusion_matrix(model, test_images, test_labels, title="Test Confusion Matrix")

    formatted_dataframe = generate_interleaved_dataframe(metrics, metrics_items)
    os.makedirs(METRICS_CSV_SAVE_DIR, exist_ok=True)
    formatted_dataframe.to_csv(METRICS_CSV_SAVE_FILE, index_label="Epoch", header=True, encoding="utf-8")

    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    plt.show()


model_training_main()
