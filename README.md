# Current Idea

For this project, I use a pretrained BERT model for binary sentiment classification.

The input review is converted into the standard BERT format:

```text
[CLS] review tokens [SEP]
```

I use the final hidden state of the `[CLS]` token as the representation of the whole review. Then I pass it into a small classifier head to predict whether the review is negative or positive.

I freeze the BERT encoder and only train the classifier head. This makes the model simpler and faster to train, because BERT is used as a fixed feature extractor.

The classifier is:

```text
Dropout -> Linear -> ReLU -> Dropout -> Linear
```

The labels are mapped as:

```python
negative = 0
positive = 1
```

For training, I use `CrossEntropyLoss` and SGD. The optimizer only updates the classifier head, not BERT.

The overall workflow is:

1. Load and clean the review data.
2. Convert labels to 0 and 1.
3. Split the data using shuffled indices.
4. Use a custom PyTorch Dataset and DataLoader.
5. Use BERT to extract the `[CLS]` representation.
6. Train only the classifier head.
7. Evaluate the model on the validation set.


# 当前思路

这个项目中，我使用预训练 BERT 模型做二分类情感分析，判断影评是 negative 还是 positive。

输入文本会被转换成标准的 BERT 分类格式：

```text
[CLS] review tokens [SEP]
```

我使用 `[CLS]` token 最后一层的 hidden state 作为整篇 review 的表示，然后把它输入到一个小的分类头中进行分类。

当前版本中，我冻结 BERT 的参数，只训练后面的 classifier head。这样可以把 BERT 当作一个固定的文本特征提取器，使训练过程更简单、更快，也更容易解释。

分类头结构是：

```text
Dropout -> Linear -> ReLU -> Dropout -> Linear
```

标签映射为：

```python
negative = 0
positive = 1
```

训练时使用 `CrossEntropyLoss` 作为 loss function，并使用 SGD 优化器。由于 BERT 被 freeze，optimizer 只会更新 classifier head 的参数。

整体流程是：

1. 读取并处理 review 数据。
2. 把文本标签转换成 0 和 1。
3. 用 shuffled indices 划分训练集和验证集。
4. 使用自定义 PyTorch Dataset 和 DataLoader。
5. 用 BERT 提取 `[CLS]` 表示。
6. 只训练 classifier head。
7. 在 validation set 上评估模型表现。
