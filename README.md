# IIKS 用の物体検出プログラム #

https://github.com/google/automl を複製し、iiks-http-server で使うためのスクリプトなどを追加したもの。

# 起動方法 #

### TensorFlow Serving ###

efficientdet の推論APIサーバーを起動する。

```
MODEL_D4="$(pwd)/saved_models/efficientdet_d4"
sudo docker run -t --rm -p 8501:8501 -v "$MODEL_D4:/models/efficientdet" -e MODEL_NAME=efficientdet tensorflow/serving
MODEL_D5="$(pwd)/saved_models/efficientdet_d5"
sudo docker run -t --rm -p 8502:8501 -v "$MODEL_D5:/models/efficientdet" -e MODEL_NAME=efficientdet tensorflow/serving
```

# 追加したスクリプト #

efficientdet/detection.sh : 推論APIサーバーにリクエストを投げて、結果を YOLO の出力に合わせてやるためのスクリプト。

