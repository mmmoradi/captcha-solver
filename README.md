# Captcha Solver

## Directory Structure

- `dataset.ipynb`
- `download.ipynb`: download and save CAPTCHA dataset
- `train.py`: train with
- `test.py`:
- `preprocessing.py`:

## Install pakages

```shell
conda create -n menv python=3.10

pip install matplotlib numpy pandas opencv-python  keras tensorflow[and-cuda]
```

## Samples

- https://keras.io/examples/vision/captcha_ocr
- https://virgool.io/dataio/how-to-break-a-golestan-captcha-system-with-machine-learning-kukzjnwwsqdx
- https://github.com/AmirH-Moosavi/Golestan
- https://pyimagesearch.com/2021/07/14/breaking-captchas-with-deep-learning-keras-and-tensorflow/
- https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710
- https://www.kaggle.com/code/achintyatripathi/emnist-letter-dataset-97-9-acc-val-acc-91-78
- https://www.kaggle.com/code/prajwalkanade/emnist-hand-writing-recognition-using-ann

- [Handwriting recognition](https://en.wikipedia.org/wiki/Handwriting_recognition)

## OpenCV

- main site: https://opencv.org
- docs site: https://docs.opencv.org

Note: C++ and Python docs are mixed.

OpenCV 4.10.0 tutorial for Python:
https://docs.opencv.org/4.10.0/d6/d00/tutorial_py_root.html


### remove noise and lines:

[Image Thresholding](https://docs.opencv.org/4.10.0/d7/d4d/tutorial_py_thresholding.html)

[Morphological Transformations](https://docs.opencv.org/4.10.0/d9/d61/tutorial_py_morphological_ops.html)

[Smoothing Images](https://docs.opencv.org/4.10.0/d4/d13/tutorial_py_filtering.html)

https://stackoverflow.com/questions/71425968/remove-horizontal-lines-with-open-cv


image binarization


## Basic concepts


- Gradient descent
- Batch Gradient Descent
- Mini Batch Gradient Descent
- Stochastic gradient descent

### loss function

### backpropagation

### Optimizers

https://keras.io/api/optimizers/

- SGD
- Adagrad
- Adadelta
- RMSprop
- Adam
- Nadam


## Keras

### Model training 

https://keras.io/api/models/model_training_apis/

### Callbacks 

https://keras.io/api/callbacks/

- `ModelCheckpoint`
- `EarlyStopping`
- `ReduceLROnPlateau`

### Metrics

https://keras.io/api/metrics/

### Saving Model
Two ways to save and load keras model:

- [ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint): Callback to save the Keras model or model weights at some frequency.


- [Whole model saving & loading](https://keras.io/api/models/model_saving_apis/model_saving_and_loading): Saves a model as a .keras file.

  ```python
  import keras

  model = keras.Sequential(
      [
          keras.layers.Dense(5, input_shape=(3,)),
          keras.layers.Softmax(),
      ],
  )
  # saving
  model.save("model.keras")
  # loading
  loaded_model = keras.saving.load_model("model.keras")

  x = keras.random.uniform((10, 3))
  assert np.allclose(model.predict(x), loaded_model.predict(x))
  ```


## TensorFlow

https://www.tensorflow.org/install/pip

limiting GPU memory usage:

https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
