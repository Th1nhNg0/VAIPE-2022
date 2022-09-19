![Untitled Diagram](https://user-images.githubusercontent.com/26407823/189125230-054bd435-1db2-42dd-889c-6a1cab53066e.png)

# Các thư viện sử dụng:

Sử dụng [**tesseract-ocr**](https://github.com/tesseract-ocr/tesseract) để lấy chữ từ ảnh pres

Sử dụng [**U-2-Net**](https://github.com/xuebinqin/U-2-Net) để preprocess ảnh pill cho object detection

Sử dụng [**yolov5**](https://github.com/ultralytics/yolov5) cho object detection. [Pretrained model](https://drive.google.com/file/d/1-slos4_7v9bMOYFEs40HKJFJ4GI8BfzJ/view?usp=sharing)

Sử dụng [**resnet50**](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50) cho classification. [Pretrained model](https://drive.google.com/file/d/1-U253UBmypqAZDRJZ2fgC3hQE0-ZBSAJ/view?usp=sharing)

# Câu lệnh tạo docker image:

```console
docker build . -t "ai4vn"
```

# Chạy container đã tạo:

## Run container:

Trong đó:

- `{path_to_public_train}` là đường dẫn tới public train.
- `{path_to_public_test}` là đường dẫn tới public test.

Thêm flags `--gpus all` để sử dụng GPU.

```console
docker run -d --ipc=host -it --gpus all --name ai4vn-AISIA-VAIPE-01 -v {path_to_public_test}:/app/public_test -v {path_to_public_train}:/app/public_train ai4vn:latest
```

Cấu trúc thư mục trong container sau khi khởi chạy thành công:

```console
.
|-- requirements.txt
|-- pres_df.csv (sử dụng cho inference)
|-- new_label.zip (sử dụng cho trainning)
|-- code
|   |-- create_train_data.py
|   |-- create_press_df.py
|   |-- label_correct.py
|   `-- inference.py
|-- models
|   |-- resnet50.h5
|   `-- yolo.pt
|-- tessdata
|   `-- eng.traineddata
|-- public_train
|   |-- pill_pres_map.json
|   |-- pill
|   |   |-- image
|   |   `-- label
|   `-- prescription
|       |-- image
|       `-- label
|-- public_test
|   |-- pill_pres_map.json
|   |-- pill
|   |   `-- image/
|   `-- prescription
|       `-- image/
`-- yolov5/
```

Cấu trúc file `pill_pres_map.json` trong folder test có dạng:

```json
{
    "20220314_201448870519": [
        "IMG_20220831_160853.jpg",
        "IMG_20220831_160909.jpg",
        "IMG_20220831_160858.jpg",
        "IMG_20220831_160900.jpg",
        "IMG_20220831_160855.jpg"
    ],
    "20220304_133947832691": [
        "94de26d1a14f64113d5e.jpg",
        "06c620cba755620b3b44.jpg",
        "5abc54cad354160a4f45.jpg",
        "ade1fee87976bc28e567.jpg",
        "e42f892c0eb2cbec92a3.jpg"
    ],
    "20220110_214018763730": [
        "IMG_9110.JPG",
        "IMG_9112.JPG",
        "IMG_9113.JPG",
        "IMG_9111.JPG",
        "IMG_9109.JPG"
    ],
    ...
}
```

## Trainning:

### Chuẩn bị data:

Theo các bước sau:

1. Đánh lại bounding box bằng U2Net và contour detection. Folder `new_label`
2. Tạo file csv chứa tên của các loại thuốc từ ảnh prescription -> `pres_df.csv`
3. Sử dụng label ở bước 1 để tạo data cho yolov5. Folder `yolo_pill` sẽ được tạo ra.
4. Cắt ảnh các viên thuốc riêng lẻ ra folder `image_crop` để train model classification.

**Bước 1:** Chạy lệnh docker sau để đánh lại label, folder được lưu vào đường dẫn: `/app/gen_train_data/new_label/`

```console
docker exec -it ai4vn-AISIA-VAIPE-01 python3 code/label_correct.py
```

Tuy nhiên chạy đoạn code này chạy khá lâu, và thường xuyên xảy ra lỗi mạng trong lúc tải model. Nên có thể sử dụng file đã chạy từ trước: [new_label.zip](new_label.zip)

```console
docker exec -it ai4vn-AISIA-VAIPE-01 mkdir gen_train_data
docker exec -it ai4vn-AISIA-VAIPE-01 unzip new_label.zip -d /app/gen_train_data/
```

**Bước 2,3,4:**

```console
docker exec -it ai4vn-AISIA-VAIPE-01 python3 code/create_train_data.py
```

### Train model yolov5

```console
docker exec -it ai4vn-AISIA-VAIPE-01 python3 yolov5/train.py --data pill.yaml --cfg yolov5x.yaml --img 640 --batch-size -1 --epochs 100 --name yolo_model_5x --project /app/models
```

Sau khi train xong model sẽ nằm ở đường dẫn: `/app/models/yolo_model_5x/weights/best.pt`

### Train model classification

```console
docker exec -it ai4vn-AISIA-VAIPE-01 python3 code/train_classification.py
```

Sau khi train xong sẽ có ouput tên đường dẫn của model, thường sẽ nằm trong thư mục: `/app/models/`

## Inference:

```console
docker exec -it  ai4vn-AISIA-VAIPE-01 python3 code/inference.py
```

Mặc định code sẽ chạy 2 model pretrained từ trước được tải từ drive. Có thể sử dụng 2 flag sau cho 2 model khác:

- `--yolo_model {path}`: đường dẫn tới model yolov5 object detection
- `--class_model {path}`: đường dẫn tới model classification

```console
docker exec -it  ai4vn-AISIA-VAIPE-01 python3 code/inference.py --yolo_model /app/models/yolo.pt --class_model /app/models/resnet50.h5
```

Sau khi chạy xong kết quả sẽ ở `/app/results.csv`
