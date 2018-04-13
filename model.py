import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Lambda, Cropping2D

from nvidianet import NvidiaNet

# Configuration of cropping dimensions
CROP_TOP = 60
CROP_BOTTOM = 25

CORRECTION = 0.2

DO_FLIP = False
USE_LEFT = False
USE_RIGHT = False

# Network hyper parameter configuration
EPOCHS=5
BATCH_SIZE=32

def main():
    lines = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            
    images = []
    measurements = []
    for line in lines:
        steering_center = float(line[3])
        # create adjusted steering measurements for the side camera images
        # this is a parameter to tune
        steering_left = steering_center + CORRECTION
        steering_right = steering_center - CORRECTION
        image_center = process_image(line[0])
        image_left = process_image(line[1])
        image_right = process_image(line[2])
        images.append(image_center)
        measurements.append(steering_center)
        
        if (DO_FLIP):
            # add flipped image
            images.append(cv2.flip(image_center,1))
            measurements.append(-steering_center)
        if (USE_LEFT):
            images.append(image_left)
            measurements.append(steering_left)
            if (DO_FLIP):
                # add flipped image
                images.append(cv2.flip(image_left,1))
                measurements.append(-steering_left)
        if (USE_RIGHT):
            images.append(image_right)
            measurements.append(steering_right)
            if (DO_FLIP):
                # add flipped image
                images.append(cv2.flip(image_right,1))
                measurements.append(-steering_right)

    X_train = np.array(images)
    y_train = np.array(measurements)
    model = Sequential()
    model.add(Cropping2D(cropping=((CROP_TOP,CROP_BOTTOM), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model = NvidiaNet(model)
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=EPOCHS)
    model.save('model.h5')
    return
                                
                                
def process_image(source_path):
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

if __name__ == '__main__':
    main()
