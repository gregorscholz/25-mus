import csv
import os
import shutil

PATH = "datasets/juggling_balls"
KEY_NAMES = [
    "filename",
    "rhx",
    "rhy",
    "lhx",
    "lhy",
    "b1x",
    "b1y",
    "b2x",
    "b2y",
    "b3x",
    "b3y",
]
annotations = os.listdir(f"{PATH}/annotations")


def split_data(file):
    with open(f"datasets/juggling_balls/{file}") as f:
        data = []
        for line in f:
            line = line.replace(".csv", "")
            data.append(line.replace("\n", ""))
        return data


train_data = split_data("trainvideos")
print(train_data)
test_data = split_data("testvideos")
val_data = split_data("validationvideos")


def normalize(val, imgsz=256):
    return int(val) / imgsz


width = 10
height = 10

for annotation in annotations:

    for row in csv.reader(open(f"{PATH}/annotations/{annotation}")):
        data = {k: v for k, v in zip(KEY_NAMES, row) if v != ""}
        filename = data["filename"].replace(".png", ".txt")

        video = filename[:-8]
        category = (
            "train" if video in train_data else "val" if video in val_data else "test"
        )

        # copy frames
        shutil.copyfile(
            f"{PATH}/frames/{data['filename']}",
            f"{PATH}/images/{category}/{data['filename']}",
        )

        # create labels

        with open(f"{PATH}/labels/{category}/{filename}", "w") as file:

            if "rhx" in data and "rhy" in data:
                line = f"0 {normalize(data['rhx'])} {normalize(data['rhy'])} {normalize(width)} {normalize(height)}\n"
                file.write(line)

            if "lhx" in data and "lhy" in data:
                line = f"1 {normalize(data['lhx'])} {normalize(data['lhy'])} {normalize(width)} {normalize(height)}\n"
                file.write(line)

            if "b1x" in data and "b1y" in data:
                line = f"2 {normalize(data['b1x'])} {normalize( data['b1y'])} {normalize(width)} {normalize(height)}\n"
                file.write(line)

            if "b2x" in data and "b2y" in data:
                line = f"2 {normalize(data['b2x'])} {normalize(data['b2y'])} {normalize(width)} {normalize(height)}\n"
                file.write(line)

            if "b3x" in data and "b3y" in data:
                line = f"2 {normalize(data['b3x'])} {normalize(data['b3y'])} {normalize(width)} {normalize(height)}\n"
                file.write(line)
