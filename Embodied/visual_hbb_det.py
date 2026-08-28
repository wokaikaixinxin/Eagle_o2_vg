import re
from PIL import Image, ImageDraw
from locateanything_worker import LocateAnythingWorker

def parse_locany_boxes(answer, image_width, image_height):
    pattern = r"<ref>(.*?)</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>"
    results = []

    for m in re.finditer(pattern, answer):
        label = m.group(1)
        x1, y1, x2, y2 = map(int, m.groups()[1:])

        # 跳过模型输出的非法框
        if x2 <= x1 or y2 <= y1:
            print("skip invalid box:", label, [x1, y1, x2, y2])
            continue

        results.append({
            "label": label,
            "box": [
                x1 / 1000 * image_width,
                y1 / 1000 * image_height,
                x2 / 1000 * image_width,
                y2 / 1000 * image_height,
            ],
        })

    return results


def draw_boxes(image, boxes, save_path):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for item in boxes:
        label = item["label"]
        x1, y1, x2, y2 = item["box"]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 14)), label, fill="red")

    img.save(save_path)
    return img


worker = LocateAnythingWorker("/root/autodl-tmp/Eagle/Embodied/work_dirs/locany_dior_lora_single_97g_8192/checkpoint-4412")

img_path = "/root/autodl-tmp/dior/JPEGImages-test/11726.jpg"
img = Image.open(img_path).convert("RGB")

answer = worker.detect(img, ["airplane", "bridge"])["answer"]
print(answer)
boxes = parse_locany_boxes(answer, img.width, img.height)

draw_boxes(
    img,
    boxes,
    "/root/autodl-tmp/11726_plane_vis.jpg"
)