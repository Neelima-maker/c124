import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("keras_model.h5")


vid = cv2.VideoCapture(0)

while True:
    check , image = vid.read()
# resizing
    img = cv2.resize(image,(224,224))
  
# Expansion of image to 4d from 3d
#expand axis=0/axis=1
    test_img = np.array(img,dtype = np.float32)
    test_img = np.expand_dims(test_img, axis=0)
# Normalization==>divide each pixel with highest value of color =255 
    normalized_image = test_img/255.0

    prediction = model.predict(normalized_image)
    print("Predicted :  ",  prediction)


    cv2.imshow("result",image)

    if cv2.waitKey(1)==32:
        print("closing..")
        break

vid.release()


    
