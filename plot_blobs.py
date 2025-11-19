import PIL
import pdf2image

from blob_matcher.keypoints import keypoints_to_torch
from blob_matcher.utils import read_json

blob_info = read_json("./data/real_image_data/2025_11_05/boards/blob_board_3f9.json")
img = pdf2image.convert_from_path(
    "./data/real_image_data/2025_11_05/boards/blob_board_3f9.pdf",
    dpi=blob_info["preamble"]["board_config"]["print_density"]["value"],
    #grayscale=True
)[0]

keypoints = keypoints_to_torch(blob_info)

draw = PIL.ImageDraw.Draw(img)
for keypoint in keypoints:
    draw.circle(keypoint[:2], keypoint[2], outline=(255, 0, 0), width=5)
img.show()
