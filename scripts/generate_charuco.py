#!/usr/bin/env python3
import cv2
import os

def generate_charuco_board(output_path="charuco_10x7.png"):
    # 10x7 squares
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    # The units here don't matter for the image generation as long as the ratio is correct.
    # Standard: square size = 0.025m, marker size = 0.015m (so ratio is 0.025 to 0.015)
    board = cv2.aruco.CharucoBoard((10, 7), 0.025, 0.015, dictionary)
    
    # Generate a high-resolution image
    img = board.generateImage((2000, 1400), marginSize=100, borderBits=1)
    
    cv2.imwrite(output_path, img)
    print(f"Successfully generated ChArUco board and saved to {os.path.abspath(output_path)}")
    print("When printing, ensure you print at 100% scale (Actual Size) with no scaling to fit.")

if __name__ == "__main__":
    generate_charuco_board()
