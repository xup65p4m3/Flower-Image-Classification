#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import applications
from sklearn.metrics import confusion_matrix
import seaborn as sn

train_file = "data/train/"
test_file  = "data/test/"
img_size = 224
batch_size = 32

model = applications.densenet.DenseNet169(include_top=True, weights='imagenet')
model.layers.pop()
model.layers[-1].outbound_nodes = []
add_layer = Dense(units=5, activation="softmax")(model.layers[-1].output)
model = Model(model.input, add_layer)

adam = optimizers.Adam(lr=0.001/100, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = optimizers.SGD(lr=0.01/100, momentum=0.9, decay=0.0, nesterov=False)
model.compile(loss = "categorical_crossentropy",
              optimizer = adam,
              metrics = ["accuracy"])

def preprocess_input(x):
    img = applications.densenet.preprocess_input(img_to_array(x))
    return array_to_img(img)

datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 5,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = "nearest",
    validation_split = 0.1)

train_generator = datagen.flow_from_directory(
    directory = train_file,
    target_size = (img_size, img_size),
    class_mode = "categorical",
    batch_size = batch_size,
    shuffle = True,
    subset = "training")

validation_generator = datagen.flow_from_directory(
    directory = train_file,
    target_size = (img_size, img_size),
    class_mode = "categorical",
    batch_size = 1,
    shuffle = True,
    subset = "validation")

checkpoint = ModelCheckpoint(
    filepath = "09-DenseNet169_best_weights.hdf5",
    monitor = "val_acc",
    verbose = 1,
    save_best_only = True,
    mode = "max")

earlystopping = EarlyStopping(
    monitor = "val_loss",
    patience = 10,
    verbose = 1,
    mode = "auto")

callbacks_list = [checkpoint, earlystopping]
total_batch = train_generator.n // batch_size

train_history = model.fit_generator(
    generator = train_generator,
    steps_per_epoch = total_batch,
    epochs = 100,
    verbose = 1,
    validation_data = validation_generator,
    validation_steps = train_generator.n // 10,
    callbacks = callbacks_list)

model.load_weights("09-DenseNet169_best_weights.hdf5")


# In[2]:


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel("train")
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"])
    plt.show()
print("Accuracy:")
show_train_history(train_history, "acc", "val_acc")
print("\nLoss:")
show_train_history(train_history, "loss", "val_loss")


# In[3]:


training_check_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
training_check_generator = training_check_datagen.flow_from_directory(
        directory = train_file,
        target_size = (img_size, img_size),
        class_mode = None,
        batch_size = 1,
        shuffle = False)
training_predict = model.predict_generator(training_check_generator, steps=training_check_generator.n)
training_predict = training_predict.argmax(axis=1)
train_class_order = list(train_generator.class_indices.keys())
training_check_class = [train_class_order[training_predict[i]] for i in range(len(training_predict))]
training_target_class = [train_class_order[training_check_generator.classes[i]] for i in range(len(training_check_generator.classes))]
Confusion_Matrix = confusion_matrix(training_target_class, training_check_class)
Confusion_Matrix = Confusion_Matrix/Confusion_Matrix.sum(axis=1).reshape(5,1)
cm_df = pd.DataFrame(100*Confusion_Matrix.round(6), index = train_class_order, columns = train_class_order)
plt.figure(figsize = (20,15))
sn.heatmap(cm_df, annot=True, annot_kws={"size": 16}, fmt="g", vmin=0, vmax=5)


# In[4]:


test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
test_generator = test_datagen.flow_from_directory(
        directory = test_file,
        target_size = (img_size, img_size),
        class_mode = None,
        batch_size = 1,
        shuffle = False)
test_predict = model.predict_generator(test_generator, steps=test_generator.n)
np.savetxt("09-DenseNet169_predict.csv", test_predict, delimiter=",")
test_predict = test_predict.argmax(axis=1)

train_class_order = list(train_generator.class_indices.keys())
target_class_order = ["daisy","dandelion","rose","sunflower","tulip"]
test_class = [train_class_order[test_predict[i]] for i in range(len(test_predict))]
test_class = [np.where(np.array(test_class[i]) == np.array(target_class_order))[0][0] for i in range(len(test_predict))]
test_id = [i[8:-4] for i in test_generator.filenames]

test_result = pd.DataFrame(columns=["id", "class"])
test_result["id"] = test_id
test_result["class"] = test_class
test_result.to_csv("09-DenseNet169_test_result.csv", index=False, sep=",")


# In[ ]:




