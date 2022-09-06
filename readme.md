# Các thư viện sử dụng:

Sử dụng [**tesseract-ocr**](https://github.com/tesseract-ocr/tesseract) để lấy chữ từ ảnh pres

Sử dụng [**U-2-Net**](https://github.com/xuebinqin/U-2-Net) để preprocess ảnh pill cho object detection

Sử dụng [**yolov5**](https://github.com/ultralytics/yolov5) cho object detection. Pretrained model [yolo_5x_bb_best.pt](https://drive.google.com/file/d/1-slos4_7v9bMOYFEs40HKJFJ4GI8BfzJ/view?usp=sharing)

Sử dụng [**resnet50**](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50) cho classification. Pretrained model [class_resnet50-new_yolo-20220829-164052.h5](https://drive.google.com/file/d/1-U253UBmypqAZDRJZ2fgC3hQE0-ZBSAJ/view?usp=sharing)

# Câu lệnh tạo docker image:

```bash
docker build . -t "ai4vn"
```

# Chạy container đã tạo:

## Inference mode:

Run container: trong đó `{path_to_public_test}` là đường dẫn tới public test. Thêm flags `--gpus all` để sử dụng GPU.

```bash
docker run -d -it --gpus all --name ai4vn-AISIA-VAIPE-01 -v {path_to_public_test}:/app/public_test ai4vn:latest
```

Run inference:

```bash
docker exec -it  ai4vn-AISIA-VAIPE-01 python3 code/inference.py
```

Sau khi chạy xong kết quả sẽ ở `/app/results.csv`
