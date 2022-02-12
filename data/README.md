# Dataset
* [CIFAR Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)에서 data 파일 다운로드

* File explain
    ```bash
    data/
    ├── data/
        ├── download_data.py        # prepare CiFar-10 Dataset
        ├── train_img2labels.json   # train images & labels (40000)
        ├── valid_img2labels.json   # valid images & labels (10000)
        └── test_img2labels.json    # test images & labels (10000)
    ```

* execute
    ```bash
    python download_data.py --t train
    ```
    * `--t`: Download (train/valid/test)Dataset Image & GT Labels (default:train)
    * Train Dataset 50000장 중 10000장을 Valid Dataset으로 사용 <br/>
    (train: data_batch_1~4, valid: data_batch_5, test: test_batch)
