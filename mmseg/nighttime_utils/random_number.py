import random
import json

motion_blur_params = []

for i in range(10000):
    degree=random.randint(5, 20)
    angle=random.uniform(0, 360)
    motion_blur_params.append(dict(degree=degree, angle=angle))

with open('motion_blur_params.json', 'w', encoding='utf-8') as file:
    json.dump(motion_blur_params, file, ensure_ascii=False, indent=4)