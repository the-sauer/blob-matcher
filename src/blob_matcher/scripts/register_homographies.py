# Copyright 2026 Hendrik Sauer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2 as cv
import numpy as np
import pdf2image


def main():
    relative_homographies_path = "./data/cotracker_results/C0001_sonyfx3_tamron20mm_f8.MP4_rel_homographies.npy"
    relative_homographies = np.load(relative_homographies_path)
    initial_homopgraphy_path = "./data/cotracker_results/first_homography.npy"
    initial_homography = np.load(initial_homopgraphy_path)

    refined_homographies = []

    blobboard_size = (1040, 1040)
    blobboard = np.array(
        pdf2image.convert_from_path(
            "./data/cotracker_results/blob_board_336_0997a.pdf",
            grayscale=True,
            size=(1200, 1200)
        )[0]
    )[80:1120, 80:1120, np.newaxis]
    assert blobboard.shape[:2] == blobboard_size

    cv.imwrite("./data/cotracker_results/blob_pattern.png", blobboard)

    prev_points = cv.goodFeaturesToTrack(blobboard, maxCorners=1000, qualityLevel=0.01, minDistance=8, blockSize=19)

    for i in range(1, 613):
        print(f"Refining homography for image {i}")
        unrefined_homography = relative_homographies[i-2] @ refined_homographies[i-2] if i > 1 else initial_homography
        img = cv.imread(f"./data/cotracker_results/frames/frame{i:03d}.png", cv.IMREAD_GRAYSCALE)
        warped_img = cv.warpPerspective(img, np.diag([*blobboard_size, 1]) @ np.linalg.inv(unrefined_homography), blobboard_size)

        A = np.stack([blobboard.flatten(), np.ones(blobboard.size)], axis=1)
        b = warped_img.flatten()
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        gain = x[0]
        bias = x[1]
        normalized_blobboard = (gain * blobboard + bias).astype(np.uint8)

        cv.imwrite(f"./data/cotracker_results/warped_frames/frame{i:03d}.png", warped_img)
        cv.imwrite(f"./data/cotracker_results/normalized_blobboards/frame{i:03d}.png", normalized_blobboard)

        refined_next_points, *_ = cv.calcOpticalFlowPyrLK(normalized_blobboard, warped_img, prev_points, None, winSize=(21, 21), maxLevel=3)

        flow_visualization = np.ones((blobboard_size[1], blobboard_size[0], 3), dtype=np.uint8)
        for (x1, y1), (x2, y2) in zip(prev_points.reshape(-1, 2), refined_next_points.reshape(-1, 2)):
            cv.arrowedLine(flow_visualization, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=1, tipLength=0.3)
        cv.imwrite(f"./data/cotracker_results/optical_flow/frame{i:03d}.png", flow_visualization)

        homography_refinement = cv.findHomography(prev_points.reshape(-1, 2), refined_next_points.reshape(-1, 2))[0]
        refined_homography = homography_refinement @ unrefined_homography
        refined_homographies.append(refined_homography)

        pts_src = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ], dtype=np.float32)
        pts_src = pts_src.reshape(-1, 1, 2)
        pts_dst = cv.perspectiveTransform(pts_src, refined_homography)

        cv.polylines(img, [np.int32(pts_dst)], isClosed=True, color=(255, 0, 0), thickness=1)
        cv.imwrite(f"./data/cotracker_results/annotated_frames/frame{i:03d}.png", img)

    np.save("./data/cotracker_results/refined_homographies.npy", np.stack(refined_homographies))


if __name__ == "__main__":
    main()
