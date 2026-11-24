# Conditional Flow Matching for PDE Data

Прототип генеративной модели на основе **Conditional Flow Matching** для генерации решений двух типов PDE:

- Allen–Cahn (ACE), класс 1.
- Wave-Gauss, класс 2.

## Данные

Источники:
- [ACE Dataset](https://huggingface.co/datasets/camlab-ethz/ACE)
- [Wave-Gauss Dataset](https://huggingface.co/datasets/camlab-ethz/Wave-Gauss)

Для каждого класса подготовлено:
- 100 Train samples
- 20 Validation samples
- 20 Test samples

Размер одной траектории: `(16, 128, 128)`.

Данные нормализованы отдельно для каждого класса.

## Модель

Используется одна условная CNN с Residual Blocks.

\[
v_\theta(X_s,s,c)
\]

Условия:
- \(X_s\) — промежуточное состояние;
- \(s\) — время Flow Matching;
- \(c\) — класс PDE.

Количество параметров: **66 352**.

Обучение выполняется с использованием Conditional Flow Matching Loss (MSE).

## Структура проекта

```text
project/
├── data/
│   └── normalized/
├── dataset.py
├── model.py
├── flow_matching.py
├── train.py
├── generate.py
├── visualize.py
├── cfm_model.pth
└── README.md
```

## Запуск

Установка библиотек:

```bash
pip install torch numpy matplotlib
```

Обучение:

```bash
python train.py
```

Генерация:

```bash
python generate.py
```

Визуализация:

```bash
python visualize.py
```

## Результаты

Обучение: 10 эпох, AdamW, learning rate `1e-4`, batch size `2`, CUDA.

| Метрика | Значение |
|---|---:|
| Final Train Loss | 1.0349 |
| Final Validation Loss | 1.0389 |
