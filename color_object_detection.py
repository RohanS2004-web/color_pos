import cv2 # do all camera work 
import numpy as np #frames are stored in the form of numpy array 

# WebCam IP address
url = "http://192.168.100.25:8080/video" 

# video remove - OpenCV will fail to read frames ,it will process it as a HTML interface
# Capture the video 

cap = cv2.VideoCapture(url) #read video frame by frame, 1 - USB Camera 0 - Laptop webcam

while True:
    ret, frame = cap.read() #ret: True - frame captured False - frame did not capture frame: read single frame 
    
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (640, 480)) #less pixels more processing speed

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #H - Hue(color type),S - Saturation (color intensity),V - Value(brightness)
    # HSV separates color information from brightness, making color detection more accurate and less sensetive to lighting variations.

    # RED color range
    # why we have given two ranges of red ? - beacuse HUE spectrum of red color is detected in two ranges 0 - 10; 170 - 180;
    lower_red1 = np.array([0, 120, 70]) 
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170,120,70])
    upper_red2 = np.array([180,255,255])

# create a binary image where red color = white, rest = black 
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2

    # Remove noise - brightness error (if not removed : false detection,tiny contours)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)  
# GausianBlur(image,(kernal_width,kernal_height),sigma)
    # kernal : part(no.of pixels horizonatlly,no of pixels vertically)

    # Find contours (objects)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    # contours tell where the object is? how big it is? and where to move

    height, width = frame.shape[:2] #shape[height,weight,color channel (sliced)]

    # Draw center lines
    # cv2.line(image,(x1,y1),(x2,y2),thickness)
    cv2.line(frame, (int(width/3), 0), (int(width/3), height), (255,255,255), 2) 
    cv2.line(frame, (int(2*width/3), 0), (int(2*width/3), height), (255,255,255), 2)

    decision = "FORWARD"

    for cnt in contours: #cnt is contour among multiple cotours
        area = cv2.contourArea(cnt)

        if area > 2000:  # Ignore small noise adjust area of contour
            x, y, w, h = cv2.boundingRect(cnt) 

            # Draw rectangle
            #drawing rectangle along the contour such that it covers whole contour

            # cv2.rectangle(image,(starting point coordinates, ending point coordinates,color,thickness))
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            center_x = x + w//2 #find center of object
        
            # Decision logic
            if center_x < width/3:
                decision = "RIGHT"
            elif center_x > 2*width/3:
                decision = "LEFT"
            else:
                decision = "STOP"

    # Display decision
    cv2.putText(frame, decision, (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,0,255), 3)

    print("Decision:", decision)

    cv2.imshow("Robot View", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()