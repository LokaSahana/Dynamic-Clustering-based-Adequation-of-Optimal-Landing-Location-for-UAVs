import cv2

# Load the image and convert it to grayscale
img = cv2.imread('C:/Users/sahan/Documents/ZED/Explorer_HD2K_SN2084_16-09-23.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to segment the image and find the contours
thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area and draw it on the original image
max_contour = max(contours, key=cv2.contourArea)
cv2.drawContours(img, [max_contour], 0, (0, 255, 0), 2)

# Get the bounding rectangle of the contour and calculate the height of the box
x, y, w, h = cv2.boundingRect(max_contour)
height_pixels = h

# Calculate the height of the box in feet and millimeters
pixels_to_feet = 0.000869
height_feet = height_pixels * pixels_to_feet 
millimeter = 304.8 * height_feet

# Print the height of the box in feet and millimeters
print(f"The height of the object is {height_pixels:.2f} pixels or {height_feet:.2f} feet or {millimeter:.2f} millimeters.")

# Display the result
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()





