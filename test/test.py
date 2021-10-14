import paddlenlp

# MODEL_NAME = "ernie-1.0"
# # ernie_model = paddlenlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)
# # model = paddlenlp.transformers.ErnieForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=3)
# tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)
# encoded_text = tokenizer(text=",./。？#")
# input_ids = paddle.to_tensor([encoded_text['input_ids']])
# print("input_ids : {}".format(input_ids))
# token_type_ids = paddle.to_tensor([encoded_text['token_type_ids']])
# print("token_type_ids : {}".format(token_type_ids))

train_ds, dev_ds, test_ds = paddlenlp.datasets.load_dataset(
    'chnsenticorp', splits=['train', 'dev', 'test'])

print(train_ds)
print(train_ds.label_list)
for i in range(10):
    print(train_ds[i])