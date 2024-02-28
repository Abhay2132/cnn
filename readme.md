# cnn
Transfer Learning with DenseNet201 Architectue

# Installing Dependencies

```
pip install -r requirements.txt
```

# Downloading Dataset

Download your dataset from `kaggle` using your `kaggle.json` file.

## In Console
```
mkdir -p /home/codespace/.kaggle
cp 'your json file' /home/codespace/.kaggle
kaggle datasets download -d pmigdal/alien-vs-predator-images -p datasets/densenet
```

# Train your Model

```
python src/densenet/app.py
```