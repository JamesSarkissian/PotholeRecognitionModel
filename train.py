"""
train.py - Teach a neural network to tell "pothole" from "clear road".

How to run it:
    python train.py                         # uses ~/Downloads/Pothole
    python train.py --data /path/to/Pothole --epochs 10

What this script does, step by step:
    1. Reads the three _annotations.csv files (train / valid / test) and turns
       every image into a simple yes/no label: 1 = has a pothole, 0 = clear road.
    2. Builds a model using TRANSFER LEARNING: we borrow MobileNetV2, a network
       Google already trained on 1.4 million everyday photos, and only teach a
       small new "head" on top of it to answer our one question.
    3. Trains in two stages:
         Stage 1 - the borrowed network is frozen, only the new head learns.
         Stage 2 - "fine-tuning": we unfreeze the top of the borrowed network
                   and let it adjust slightly to road photos.
    4. Tests the finished model on the test images it has never seen and
       prints how well it did.
    5. Saves the model to pothole_classifier.keras so live.py can use it.
"""

import argparse
import csv
import os

import tensorflow as tf

# ---------------------------------------------------------------------------
# Settings. These are the "knobs" you can turn. Change them and re-run to see
# what happens - that's the best way to learn what each one does.
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224        # MobileNetV2 was trained on 224x224 images, so we match it
BATCH_SIZE = 32         # how many images the model looks at before each update
STAGE1_EPOCHS = 10      # an "epoch" = one full pass over every training image
STAGE2_EPOCHS = 5       # extra passes for fine-tuning
STAGE1_LR = 1e-3        # learning rate: how big each update step is
STAGE2_LR = 1e-5        # much smaller for fine-tuning, so we don't wreck what it knows
# Save the model next to this script, where live.py looks for it
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pothole_classifier.keras")


# ---------------------------------------------------------------------------
# Step 1: load the data
# ---------------------------------------------------------------------------
def load_split(data_dir, split):
    """Read one split's CSV and return (list of image paths, list of 0/1 labels).

    The CSV has one row per pothole BOX, so an image with 3 potholes appears on
    3 rows. Clear-road images appear on one row with the class column blank.
    For a yes/no classifier we only care whether an image has ANY pothole.
    """
    split_dir = os.path.join(data_dir, split)
    has_pothole = {}  # filename -> True / False

    with open(os.path.join(split_dir, "_annotations.csv"), newline="") as f:
        for row in csv.DictReader(f):
            name = row["filename"]
            is_pothole = row["class"] == "pothole"
            # "or" means: once an image is marked as having a pothole, it stays marked
            has_pothole[name] = has_pothole.get(name, False) or is_pothole

    paths = [os.path.join(split_dir, name) for name in has_pothole]
    labels = [1 if has_pothole[name] else 0 for name in has_pothole]
    return paths, labels


def load_image(path, label):
    """Turn a file path into a 224x224 image tensor (a grid of pixel numbers)."""
    data = tf.io.read_file(path)
    image = tf.io.decode_jpeg(data, channels=3)              # RGB pixels, 0-255
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))  # squash to 224x224
    return image, label


def make_dataset(paths, labels, shuffle):
    """Wrap the file list in a tf.data pipeline that loads images in batches."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        # Shuffling stops the model from learning the ORDER of the images
        ds = ds.shuffle(len(paths), seed=42)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    # prefetch loads the next batch while the current one is training
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Step 2: build the model
# ---------------------------------------------------------------------------
def build_model():
    # Data augmentation: random small changes to each training image so the
    # model sees "new" pictures every epoch and doesn't just memorize them.
    # These layers only do anything during training; they switch off later.
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.2),
    ], name="augmentation")

    # The borrowed network. include_top=False drops its original final layer,
    # which picked between 1,000 ImageNet categories (cats, cars, ...).
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze it for stage 1

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = augment(inputs)
    # MobileNetV2 expects pixels scaled from -1 to 1 instead of 0 to 255
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    # The base outputs a 7x7 grid of 1,280 "features". Averaging the grid turns
    # it into one list of 1,280 numbers describing the whole image.
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Dropout randomly ignores 30% of those numbers during training, which
    # makes the model rely on many clues instead of one (less overfitting).
    x = tf.keras.layers.Dropout(0.3)(x)
    # One output neuron. Sigmoid squeezes it to 0..1 = "probability of pothole".
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs), base


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        # binary_crossentropy = the standard "how wrong was I" score for yes/no
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            # precision: when it says "pothole", how often is it right?
            tf.keras.metrics.Precision(name="precision"),
            # recall: of all real potholes, how many did it catch?
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


# ---------------------------------------------------------------------------
# Steps 3-5: train, test, save
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train the pothole classifier.")
    parser.add_argument("--data", default=os.path.expanduser("~/Downloads/Pothole"),
                        help="folder that holds train/, valid/ and test/")
    parser.add_argument("--epochs", type=int, default=STAGE1_EPOCHS,
                        help="stage 1 epochs (stage 2 adds %d more)" % STAGE2_EPOCHS)
    args = parser.parse_args()

    train_paths, train_labels = load_split(args.data, "train")
    val_paths, val_labels = load_split(args.data, "valid")
    test_paths, test_labels = load_split(args.data, "test")
    for name, labels in [("train", train_labels), ("valid", val_labels), ("test", test_labels)]:
        print(f"{name}: {len(labels)} images, {sum(labels)} pothole, "
              f"{len(labels) - sum(labels)} clear road")

    train_ds = make_dataset(train_paths, train_labels, shuffle=True)
    val_ds = make_dataset(val_paths, val_labels, shuffle=False)
    test_ds = make_dataset(test_paths, test_labels, shuffle=False)

    # There are about twice as many pothole images as clear-road ones. Class
    # weights make each clear-road mistake count more, so the model can't win
    # by simply guessing "pothole" every time.
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    class_weight = {0: len(train_labels) / (2 * n_neg), 1: len(train_labels) / (2 * n_pos)}

    model, base = build_model()
    model.summary()

    # Save the best version seen so far (judged on validation data), and stop
    # early if validation stops improving for 3 epochs in a row.
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    print("\n=== Stage 1: training the new head ===")
    compile_model(model, STAGE1_LR)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              class_weight=class_weight, callbacks=callbacks)

    print("\n=== Stage 2: fine-tuning the top of MobileNetV2 ===")
    base.trainable = True
    # Keep the first 100 layers frozen. Early layers detect basic edges and
    # colors, which are useful as-is; later layers detect more specific shapes.
    for layer in base.layers[:100]:
        layer.trainable = False
    compile_model(model, STAGE2_LR)  # must re-compile after changing trainable
    model.fit(train_ds, validation_data=val_ds, epochs=STAGE2_EPOCHS,
              class_weight=class_weight, callbacks=callbacks)

    print("\n=== Final test on images the model has never seen ===")
    loss, acc, precision, recall = model.evaluate(test_ds)
    print(f"accuracy:  {acc:.1%}")
    print(f"precision: {precision:.1%}  (when it says pothole, it's right this often)")
    print(f"recall:    {recall:.1%}  (share of real potholes it caught)")

    model.save(MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}. Try it live with: python live.py")


if __name__ == "__main__":
    main()
