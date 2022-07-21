import cv2
from random import randrange

# load some pre-trained data on face frontals from opencv (haar cascade algorithm) (training part)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Select images to detect
#img = cv2.imread('pep.jpg')
img = cv2.imread('BEU.jpg')

# convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# print(face_coordinates)

# Draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (234, 112, 96), 4)


# Display the image
cv2.imshow('Hemie\'s Face Detector', img)

# wait key: waiting for a key to be pressed to exit the code: you cannot display your image without waitKey() in cv
cv2.waitKey()

print('Code Completed')


# Let's try with a live video you can comment out line 4 - line 33

# load some pre-trained data on face frontals from opencv (haar cascade algorithm) (training part)
# trained_face_data = cv2.CascadeClassifier(
#     'haarcascade_frontalface_default.xml')

# # Testing webcam: You can use a default video camera by giving it an index of 0 or you can replace 0 with the video file you want to use
# webcam = cv2.VideoCapture(0)

# # Using a while loop to iterate forever over frames till the webcam is turned off or the video ends
# while True:
#     # reading the current frame, successful_frame_read serves as the placeholder: it is not a most to have it in the code
#     successful_frame_read, frame = webcam.read()

#     # convert to grayscale cv reads RBG from the back hence BGR
#     grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # detect faces
#     face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

#     # Draw rectangles around the faces and testing random range
#     for (x, y, w, h) in face_coordinates:
#         cv2.rectangle(frame, (x, y), (x+w, y+h),
#                       (randrange(234), randrange(112), randrange(96)), 4)

#     # Display the image
#     cv2.imshow('Hemie\'s Face Detector', frame)
#     key = cv2.waitKey()

#     # Stop if Q key is pressed (check ASCII code for Q) to get out of the loop
#     if key == 81 or key == 113:
#         break
# # to release VideoCapture Object
# webcam.release()

# print('Code Completed')
