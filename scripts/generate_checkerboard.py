import cv2
import numpy as np

# 9x6 inner corners means 10x7 squares
squares_x = 10
squares_y = 7
pixels_per_square = 200

width = squares_x * pixels_per_square
height = squares_y * pixels_per_square

image = np.zeros((height, width), dtype=np.uint8)

for i in range(squares_y):
    for j in range(squares_x):
        if (i + j) % 2 == 0:
            image[i*pixels_per_square:(i+1)*pixels_per_square,
                  j*pixels_per_square:(j+1)*pixels_per_square] = 255

# Add a white border so it prints nicely
border = pixels_per_square // 2
image_with_border = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=255)

cv2.imwrite('/home/andrew/Downloads/robot-dev-bench/checkerboard_9x6_printable.png', image_with_border)
print("Checkerboard saved to checkerboard_9x6_printable.png")
