import tensorflow as tf
import random
import numpy as np 
import matplotlib.pyplot as plt 

# i followed the tensorflow fashion classifier tutorial for this code 

fashionDataset = tf.keras.datasets.fashion_mnist    
labelNames  = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# loading dataset returns 4 numpy arrays 
(trainImages, trainLabels) , (testImages, testLabels) = fashionDataset.load_data()


# we want to scale the values between 0-1 before feeding them to the neural network 
#normalization 
trainImages=trainImages/ 255.0
testImages = testImages/ 255.0

#preprocess data

# plt.figure()
# plt.imshow(trainImages[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()

#verify data is in the right format 
#display the first 25 images from the training set and display class below it 
# plt.figure(figsize = (10,10 ))
# for i in range(25): 
#     plt.subplot(5,5 , i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(trainImages[i], cmap = plt.cm.binary)
#     plt.xlabel(labelNames[trainLabels[i]])
# plt.show()




#build the model 

#setup layers 
# the first layer flattens the 28*28 array into a one dimensional 784 length array (28*28 = 784)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation ='relu'),
    tf.keras.layers.Dense(10)
])

#compile model 
#loss function measures accuracy on the model 
#optimizer is how the model is updated based on the loss
#metrics is used to monitor the train and testing steps ; fraction of images correctly classified
model.compile(optimizer = 'adam',
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
metrics =['accuracy'])


#train the model 

model.fit(trainImages, trainLabels, epochs=10)  


#evaluate accuracy 
testLoss , testAcc = model.evaluate(testImages, testLabels, verbose = 2)

print("\n Test Accuracy:" , testAcc) # Since the test accuracy performs worse the model is overfitted






#make predictions 

#attach a softmax layer to the model to convert the logit output to a probability output 

probModel = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probModel.predict(testImages)


# since each predictions is going to be an array of ten numbers we can represent the guess by using the max of the array 

#Graph  to look at a full set of 10 predictions 


def plotImage(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(labelNames[predicted_label],
                                100*np.max(predictions_array),
                                labelNames[true_label]),
                                color=color)

def plotValueArray(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


#Verify predictions 

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  randItem = random.randint(0, len(testImages)-1)
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plotImage(randItem, predictions[randItem], testLabels, testImages)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plotValueArray(randItem, predictions[randItem], testLabels)
plt.tight_layout()
plt.show()



