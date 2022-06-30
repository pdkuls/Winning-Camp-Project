# Import Libraries
import cv2
import numpy as np

#Reading Models and prototypes
GENDER_MODEL = 'weights/deploy_gender.prototxt'
GENDER_PROTO = 'weights/gender_net.caffemodel'
FACE_PROTO = 'weights/deploy.prototxt'
FACE_MODEL = 'weights/face_net.caffemodel'
AGE_MODEL = 'weights/deploy_age.prototxt'
AGE_PROTO = 'weights/age_net.caffemodel'

# Each Caffe Model impose the shape of the input image also
# image preprocessing is required like mean subtraction to eliminate
# the effect of illumination changes
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

GENDER_LIST = ['Male', 'Female']

# Represent the 8 age classes of this CNN probability layer
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

# Initialize frame size
frame_width = 1280
frame_height = 720

# load models
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

#Function to retrieve faces and display a rectangular box around it.
def get_faces(frame, confidence_threshold=0.5):

    # convert the frame into a blob to be ready for NN input
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))

    # set the image as input to the NN
    face_net.setInput(blob)

    # perform inference and get predictions
    output = np.squeeze(face_net.forward())

    # initialize the result list
    faces = []

    # Loop over the faces detected
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * \
                np.array([frame.shape[1], frame.shape[0],
                         frame.shape[1], frame.shape[0]])

            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int)

            # widen the box a little
            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y

            # append to our list
            faces.append((start_x, start_y, end_x, end_y))
    return faces


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):

    # initialize the dimensions of the image to be resized and grab the
    # image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:

        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:

        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    return cv2.resize(image, dim, interpolation = inter)

#Function to get the gender predictions.
def get_gender_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    gender_net.setInput(blob)
    return gender_net.forward()

#Function to get the age predictions.
def get_age_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False
    )
    age_net.setInput(blob)
    return age_net.forward()


#Function to display face data and predict gender and age data.
def predict_age_and_gender():
    """Predict the gender of the faces showing in the image"""

    #Capturing the image from the webcam.
    cap = cv2.VideoCapture(0)

    while True:
        _, img = cap.read()

        # Take a copy of the initial image to resize it
        frame = img.copy()

        # resize if higher than frame_width
        if frame.shape[1] > frame_width:
            frame = image_resize(frame, width=frame_width)

        # predict the faces
        faces = get_faces(frame)

        # Loop over the faces detected
        # for idx, face in enumerate(faces):
        for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
            face_img = frame[start_y: end_y, start_x: end_x]

            #Calling the age prediction function
            age_preds = get_age_predictions(face_img)

            #Calling the gender detection function
            gender_preds = get_gender_predictions(face_img)

            #Getting the reference age value from the gender list.
            i = gender_preds[0].argmax()
            gender = GENDER_LIST[i]
            gender_confidence_score = gender_preds[0][i]

            #Getting the reference age value from the age list.
            i = age_preds[0].argmax()
            age = AGE_INTERVALS[i]
            age_confidence_score = age_preds[0][i]

            # Draw the box
            label = f"{gender}-{gender_confidence_score*100:.1f}%, \
            {age}-{age_confidence_score*100:.1f}%"
            print(label)
            yPos = start_y - 15
            while yPos < 15:
                yPos += 15
            box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
            cv2.rectangle(frame,(start_x,start_y),(end_x,end_y),box_color,2)

            # Labeling the processed image
            cv2.putText(frame, label, (start_x, yPos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.54, box_color, 2)

            # Displaying the processed image
        cv2.imshow("Gender Estimator", frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    # Cleanup
    cv2.destroyAllWindows()

#Main function calling.
if __name__ == "__main__":
    predict_age_and_gender()
