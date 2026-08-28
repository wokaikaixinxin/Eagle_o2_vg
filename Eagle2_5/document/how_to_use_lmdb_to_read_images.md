## original 

```json
{
    "conversations": [
        {"from": "human", "value": "<image> what is this?"},
        {"from": "gpt", "value": "It is an apple."}
    ],
    "image": "path/to/image.jpg"
}
```

```python
from PIL import Image
pil_image = Image.open("path/to/image.jpg")
```


### lmdb 

```json
# sample
{
    "conversations": [
        {"from": "human", "value": "<image> what is this?"},
        {"from": "gpt", "value": "It is an apple."}
    ],
    "image": {
        "lmdb_file": "path/to/lmdb/file",
        "lmdb_key": "image_key"
    }
}
```

```python
import lmdb
import cv2
from PIL import Image
import io
import numpy as np

image_meta = sample["image"]
lmdb_file = image_meta["lmdb_file"]
lmdb_key = image_meta["lmdb_key"]
env = lmdb.open(lmdb_file, readonly=True, lock=False)
with env.begin(write=False) as txn:
    image_bin = txn.get(lmdb_key.encode('ascii'))
buf = io.BytesIO(image_bin)
try:
    pil_image = Image.open(buf)
except Exception as e:
    image_np = np.frombuffer(image_bin, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
```
