import os

import cv2
import numpy as np
import torch

boards = ["3f9", "042", "84b", "105", "120", "374", "407", "681", "a26", "afd"]
images = map(lambda f: os.path.join("./data/real_image_data/2025_11_05/images", f), os.listdir("./data/real_image_data/2025_11_05/images"))


def get_homography(pts1, pts2):
    def ax(p, q):
        return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q):
        return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = np.stack([[pts2[i][j] for i in range(4) for j in range(2)]], axis=0).T
    flat_homography = np.linalg.lstsq(a_mat, p_mat)[0].T
    homography = np.ones(shape=(3, 3))
    homography[0, :] = flat_homography[0][:3]
    homography[1, :] = flat_homography[0][3:6]
    homography[2, :2] = flat_homography[0][6:]
    return torch.as_tensor(homography, dtype=torch.float32)


for image in images:
    if os.path.exists(f"real_homographies{os.path.basename(image)}.pt"):
        continue
    homographies = {}
    for board in boards:
        # Load your image
        img = cv2.imread(image)
        if img is None:
            raise ValueError("Could not read image file.")

        points_src = []

        # Mouse callback to record clicks
        def get_points(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points_src.append([x, y])
                cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
                cv2.imshow(f"Select {board} in {image}", img)
                print(f"Point {len(points_src)}: ({x}, {y})")

        # Display image and register callback
        cv2.namedWindow(f"Select {board} in {image}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Select {board} in {image}", 2000, 1300)
        cv2.imshow(f"Select {board} in {image}", img)
        cv2.setMouseCallback(f"Select {board} in {image}", get_points)

        print("Click on 4 points in the image (top-left to bottom-right). Press 'q' to quit.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or len(points_src) == 4:
                break
            elif key == ord('r'):
                points_src = []

        cv2.destroyAllWindows()

        # Ensure we have 4 points
        if len(points_src) != 4:
            continue

        homographies[(image, board)] = get_homography(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), points_src)
    torch.save(homographies, f"real_homographies{os.path.basename(image)}.pt")
