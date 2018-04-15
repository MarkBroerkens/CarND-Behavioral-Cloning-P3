import csv
import cv2
import numpy as np
import sklearn

from nvidianet import NvidiaNet
from sklearn.model_selection import train_test_split


# Parameters for training data augmentation
DO_FLIP = True
USE_SIDE_CAMERAS = True
CORRECTION = 0.2

# Network hyper parameter configuration
EPOCHS=10
BATCH_SIZE=32

def main():
    # get the set of available samples
    samples = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    
    # use 80% for training and 20% for validation
    train_samples, validation_samples = train_test_split(samples, test_size=0.2, shuffle=True)

    # print training data statistics
    print("Number of sample lines "+str(len(samples)))
    print("Number of train lines "+str(len(train_samples)))
    print("Number of validation lines "+str(len(validation_samples)))

    print("Augment by Flipping: " + str(DO_FLIP))
    print("Use Side Cameras: " + str(USE_SIDE_CAMERAS))
    sampleFactor = getSampleFactor()
    print("Sample Factor: " + str(sampleFactor))
    print("Number of augmented train images "+str(len(train_samples) * sampleFactor))
    print("Number of augmented validation images "+str(len(validation_samples) * sampleFactor))

    # get the generators for the training and validation data
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

    # create the model
    model = NvidiaNet()

    # train the model
    sampleFactor = getSampleFactor()
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*sampleFactor, validation_data=validation_generator, nb_val_samples=len(validation_samples)*sampleFactor, nb_epoch=EPOCHS, verbose=1)

    # save the model
    model.save('model.h5')
    
    return


# Using a Python generator to provide data,
# in order to avoid out of memory exception in case of big training sets.
#
def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                addSample(images, angles, batch_sample)
            
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
    return

# Calculates how many items are added to the traininf or validation set for each sample
# provided by the simulator.
# If flipped images are added (DO_FLIP==True), then the number of items is doubled.
# If we additionally enable the data from the side cameras (USE_SIDE_CAMERAS==True),
# then we have 6 data items per sample
def getSampleFactor():
    sampleFactor = 1
    if (DO_FLIP):
        sampleFactor = 2
    
    if (USE_SIDE_CAMERAS):
        sampleFactor = sampleFactor * 3
    
    return sampleFactor

# Add training or validation data based on the description of a sample from the simulator
# the format of a sample is:
# sample[0]: center_image
# sample[1]: left_image
# sample[2]: right_image
# sample[3]: steering_angle
# If DO_FLIP==True, then a flipped image and angle is added
# If USE_SIDE_CAMERAS==True then the data from the left and right camera is added.
def addSample(images, angles, sample):
    # add sample data of center camera
    center_image = getimage(sample[0])
    center_angle = float(sample[3])
    addImage(images, angles, center_image, center_angle)
    
    # add sample data of side cameras
    if (USE_SIDE_CAMERAS):
        left_image = getimage(sample[1])
        addImage(images, angles, left_image, center_angle+CORRECTION)
        
        right_image = getimage(sample[2])
        addImage(images, angles, right_image, center_angle-CORRECTION)
    return

# Adds an image and an angle to the training or validation set.
# Additionally adds a flipped image in order to avoid bias
def addImage(images, angles, image, angle):
    images.append(image)
    angles.append(angle)
    if(DO_FLIP):
        images.append(cv2.flip(image,1))
        angles.append(-angle)
    return


# Reads the image that is identified by the last segment of path.
# Note: The image is expected in folder "data/IMG/"
def getimage(source_path):
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

if __name__ == '__main__':
    main()
