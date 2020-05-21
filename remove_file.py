import os

file_paths = open("./error.txt").readlines()
file_paths = list(map(lambda filepath: filepath[:-1], file_paths))

for file_path in file_paths:
    if os.path.exists(file_path):
        # print("Existed path")
        continue
        # os.remove(file_path)
        # print("Remove successfully")
    else:
        print("Wrong file path")
# print(file_paths)