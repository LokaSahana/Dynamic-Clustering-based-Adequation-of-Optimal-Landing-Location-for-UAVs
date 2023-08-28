import cv2
import numpy as np

# Load the image
img = cv2.imread('C:/Users/sahan/Documents/ZED/Explorer_HD2K_SN2084_13-37-54.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Find contours in the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Define the minimum and maximum size of the contours to filter out small objects
min_size = 100
max_size = 10000

# Define the size of the drone
drone_size = 50

# Create an empty mask image to draw the contours on
mask = np.zeros_like(gray)

# Define the threshold values for the criteria
avg_depth_threshold = 150
max_depth_threshold = 200
size_threshold = 500

# Define the best landing places
best_landing_places = []

# Define the number of clusters
num_clusters = len(best_landing_places)

# Define a variable to keep track of the cluster number
cluster_num = 0

# Get the height of the camera from the user input
height = input("Enter the height of the camera from the ground (in meters): ")
height = float(height)

# Loop over the contours
for i in range(len(contours)):
    # Compute the area and the perimeter of the contour
    area = cv2.contourArea(contours[i])
    perimeter = cv2.arcLength(contours[i], True)

    # Filter out contours that are too small or too large
    if area > min_size and area < max_size:
        # Increment the cluster number
        cluster_num += 1
        
        # Compute the average depth, maximum depth, and size of the contour
        depths = img[contours[i][:,0][:,1], contours[i][:,0][:,0]].astype(np.float32)
        depths = depths / 256.0  # Convert depth values from 16-bit to 8-bit
        depths = depths * height  # Convert depth values from pixels to meters
        avg_depth = np.mean(depths)
        max_depth = np.max(depths)
        size = len(depths)

        # Check if the contour satisfies the criteria for a good landing place
        if avg_depth >= avg_depth_threshold and max_depth >= max_depth_threshold and size >= size_threshold:
            # Loop over the points in the contour
            for j in range(len(contours[i])):
                # Get the x and y coordinates of the point
                x, y = contours[i][j][0][0], contours[i][j][0][1]
                # Check if a 50x50 pixel square centered at the point satisfies the criteria
                if x >= drone_size//2 and y >= drone_size//2 and x+drone_size//2 < img.shape[1] and y+drone_size//2 < img.shape[0]:
                    square = img[y-drone_size//2:y+drone_size//2+1, x-drone_size//2:x+drone_size//2+1]
                    avg_depth_square = np.mean(square)
                    max_depth_square = np.max(square)
                    if avg_depth_square >= avg_depth_threshold and max_depth_square >= max_depth_threshold:
                        best_landing_places.append((x, y))
                        
        # Draw a circle around the contour on the mask image with the cluster number
        cv2.drawContours(mask, [contours[i]], 0, cluster_num, -1)
        (x,y),radius = cv2.minEnclosingCircle(contours[i])
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(img, center, radius, (0, 0, 255), 2)
        cv2.putText(img, str(cluster_num), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Print the average depth and maximum depth from the contour with the cluster number
        print("Cluster {}: Average depth: {:.2f}, Maximum depth: {:.2f}".format(cluster_num, avg_depth, max_depth))

# Get the cluster numbers from the user input
cluster1 = int(input("Enter the first cluster number: "))
cluster2 = int(input("Enter the second cluster number: "))

# Get the depth values of the chosen clusters
depths1 = []
depths2 = []

for i in range(len(contours)):
    if hierarchy[0][i][3] == cluster1:
        depths1.extend(img[contours[i][:,0][:,1], contours[i][:,0][:,0]].astype(np.float32) / 256.0 * height)
    elif hierarchy[0][i][3] == cluster2:
        depths2.extend(img[contours[i][:,0][:,1], contours[i][:,0][:,0]].astype(np.float32) / 256.0 * height)

# Calculate the depth difference between the two clusters
depth_diff = (np.mean(depths2) - np.mean(depths1)) * 100

#Print the depth difference
print("The depth difference between cluster {} and cluster {} is {:.2f} centimeters.".format(cluster1, cluster2, depth_diff))

# Display the output images
cv2.namedWindow('Input Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Mask Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Gray Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Threshold Image', cv2.WINDOW_NORMAL)
cv2.imshow('Input Image', img)
cv2.imshow('Mask Image', mask)
cv2.imshow('Gray Image', gray)
cv2.imshow('Threshold Image', thresh)
cv2.waitKey(0)
