import requests
import os

from tqdm import tqdm

URL = "https://cryptics.georgeho.org/data/clues.csv?_next={}&_size=max"

blocks = range(0, 663652, 1000)

for i, block in enumerate(tqdm(blocks)):
    response = requests.get(URL.format(block))

    assert response.status_code == 200

    file_path = os.path.join("data", "raw_data",  f"clues_{i}.csv")
    with open(file_path, 'wb+') as file:
        file.write(response.content)
