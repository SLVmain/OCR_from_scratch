### Пример OCR с использованием CTC Loss

В качестве домашнего задания реализована архитектура CRNN из [статьи](https://arxiv.org/abs/1507.05717)
!(CRNN.png)

Запуск проекта:
1. Выполнить сборку образа
```
docker build -t ocr .
```
2. Внутри контейнера запустить обучение
```
python train.py
```

### Исходный проект:
* https://github.com/dredwardhyde/crnn-ctc-loss-pytorch
* https://github.com/Alek-dr/OCR-Example
