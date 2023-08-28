import cv2

# Load the image and convert it to grayscale
image = cv2.imread('C:/Users/sahan/Documents/ZED/Explorer_HD2K_SN2084_13-28-40.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to create a binary image
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours in the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours and draw a rectangle around each one
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
# Measure the height of the object in pixels
height_pixels = h

# Print the height in pixels
print('Height of the object in pixels:', height_pixels)

# conversion factor from pixels to feet
pixels_to_feet = 0.000869

# height in feet
height_feet = height_pixels * pixels_to_feet 
print(f"The height of the object is {height_feet:.2f} feet.")

# height in mm
millimeter = 304.8 * height_feet
print("The object is", millimeter, "millimeter.")

# Show the image with the detected boxes 
cv2.namedWindow('Detected Boxes', cv2.WINDOW_NORMAL)
cv2.imshow('Detected Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()