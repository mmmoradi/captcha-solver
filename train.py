import tensorflow as tf
import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

# load the dataset: EMNIST balanced dataset

train_set =  np.loadtxt('dataset/kaggleemnist/emnist-balanced-train.csv', delimiter=",", dtype=np.int64)
train_labels = train_set[:,0]
train_images = train_set[:,1:]

test_set =  np.loadtxt('dataset/kaggleemnist/emnist-balanced-test.csv', delimiter=",", dtype=np.int64)
test_labels = test_set[:,0]
test_images = test_set[:,1:]

# check shape of the dataset

print("Shape of train_set: \t", train_set.shape)
print("Shape of train_images: \t", train_images.shape)
print("Shape of train_labels: \t", train_labels.shape)
print()
print("Shape of test_set: \t", test_set.shape)
print("Shape of test_images: \t", test_images.shape)
print("Shape of test_labels: \t", test_labels.shape)

# Creating label dictionary: Running a loop for ASCII equivalent to character conversion

label_dictionary = {}

for index, label in np.loadtxt("dataset/kaggleemnist/emnist-balanced-mapping.txt", delimiter=" ", dtype=np.int64):
    label_dictionary[index] = chr(label)

print(f'labels: {label_dictionary}')

# normalise and reshape and transpose images

train_images_number = train_images.shape[0]
train_images_height = 28
train_images_width = 28

x_train = train_images.reshape(train_images_number, train_images_height, train_images_width, 1) / 255.0

test_images_number = test_images.shape[0]
test_images_height = 28
test_images_width = 28

x_test = test_images.reshape(test_images_number, test_images_height, test_images_width, 1) / 255.0

# transform labels

number_of_classes = len(np.unique(train_labels))

y_train = tf.keras.utils.to_categorical(train_labels, number_of_classes)
y_test  = tf.keras.utils.to_categorical(test_labels, number_of_classes)

# plot the dataset

fig,axes = plt.subplots(1,5,figsize=(20,20))

for i,ax in enumerate(axes.flat):
    ax.imshow(x_train[i].reshape(28,28))
    ax.set_title(label_dictionary[np.argmax(y_train[i])] + " " + str(train_labels[i]))



# the model

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,3,input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(number_of_classes,activation='softmax')
])

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# check points

MCP = ModelCheckpoint('checkpoint2.model.keras', verbose=1, save_best_only=True, monitor='val_accuracy', mode='max')
ES = EarlyStopping(monitor='val_accuracy', min_delta=0, verbose=1, restore_best_weights=True, patience=3, mode='max')
RLP = ReduceLROnPlateau(monitor='val_loss', patience=1, factor=0.2, min_lr=0.0001, verbose=1)

# train

history = model.fit(x_train,
                    y_train,
                    verbose=1,
                    epochs=50,
                    batch_size=16,
                    validation_data=(x_test, y_test),
                    callbacks=[MCP,ES,RLP]
                   )


# plot history


# TODO
