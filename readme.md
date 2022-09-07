# Các thư viện sử dụng:

Sử dụng [**tesseract-ocr**](https://github.com/tesseract-ocr/tesseract) để lấy chữ từ ảnh pres

Sử dụng [**U-2-Net**](https://github.com/xuebinqin/U-2-Net) để preprocess ảnh pill cho object detection

Sử dụng [**yolov5**](https://github.com/ultralytics/yolov5) cho object detection. [Pretrained model](https://drive.google.com/file/d/1-slos4_7v9bMOYFEs40HKJFJ4GI8BfzJ/view?usp=sharing)

Sử dụng [**resnet50**](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50) cho classification. [Pretrained model](https://drive.google.com/file/d/1-U253UBmypqAZDRJZ2fgC3hQE0-ZBSAJ/view?usp=sharing)

# Câu lệnh tạo docker image:

```console
$ docker build . -t "ai4vn"
```

# Chạy container đã tạo:

## Run container:

Trong đó:

    - `{path_to_public_train}` là đường dẫn tới public train.
    - `{path_to_public_test}` là đường dẫn tới public test.

Thêm flags `--gpus all` để sử dụng GPU.

```console
$ docker run -d -it --gpus all --name ai4vn-AISIA-VAIPE-01 -v {path_to_public_test}:/app/public_test -v {path_to_public_train}:/app/public_train ai4vn:latest
```

Cấu trúc thư mục trong container sau khi khởi chạy thành công:

```console
.
|-- code
|   |-- create_press_df.py
|   `-- inference.py
|-- models
|   |-- resnet50.h5
|   `-- yolo.pt
|-- pres_df.csv
|-- requirements.txt
|-- tessdata
|   `-- eng.traineddata
|-- public_test
|   |-- pill_pres_map.json
|   |-- pill
|   |   `-- image/
|   `-- prescription
|       `-- image/
`-- yolov5/
```

## inference:

```console
$ docker exec -it  ai4vn-AISIA-VAIPE-01 python3 code/inference.py
```

Mặc định code sẽ chạy 2 model pretrained từ trước được tải từ drive. Có thể sử dụng 2 flag sau cho 2 model khác:

- `--yolo_model {path}`: đường dẫn tới model yolov5 object detection
- `--class_model {path}`: đường dẫn tới model classification

```console
$ docker exec -it  ai4vn-AISIA-VAIPE-01 python3 code/inference.py --yolo_model /app/models/yolo.pt --class_model /app/models/resnet50.h5
```

Sau khi chạy xong kết quả sẽ ở `/app/results.csv`

### Run Train:

```console
$ docker exec -it  ai4vn-AISIA-VAIPE-01 python3 code/inference.py
```

Mặc định code sẽ chạy 2 model pretrained từ trước được tải từ drive. Có thể sử dụng 2 flag sau cho 2 model khác:

- `--yolo_model {path}`: đường dẫn tới model yolov5 object detection
- `--class_model {path}`: đường dẫn tới model classification

```console
$ docker exec -it  ai4vn-AISIA-VAIPE-01 python3 code/inference.py --yolo_model /app/models/yolo.pt --class_model /app/models/resnet50.h5
```

Sau khi chạy xong kết quả sẽ ở `/app/results.csv`
