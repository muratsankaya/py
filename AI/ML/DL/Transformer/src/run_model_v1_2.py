from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from data.tokenizer_helinski_nlp_opus import (
    wmt14_train_subset,
    wmt14_test_subset,
    wmt14_validation_subset,
    tokenizer,
)
from src.transformer_v1 import Transformer
from src.trainer_v1_2 import Trainer

batch_size = 16
# print(tokenizer.vocab_size)
train_data_loader = DataLoader(wmt14_train_subset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(wmt14_test_subset, batch_size=batch_size, shuffle=True)
valdiation_data_loader = DataLoader(
    wmt14_validation_subset, batch_size=batch_size, shuffle=True
)

model = Transformer(
    batch_size=batch_size,
    seq_len=32,
    d_model=32,
    number_of_heads=4,
    # +1 for bos token, it is documented that
    # tokenizer.vocab_size does not include the
    # added tokens
    input_vocab_size=tokenizer.vocab_size + 1,
    output_vocab_size=tokenizer.vocab_size + 1,
    dropout_p=0.1,
)

d_model = 64


def lr_lambda(step_num):
    if step_num == 0:
        step_num = 1
    return d_model ** (-0.5) * step_num ** (-0.5)


trainer_args = {
    "lr": 1.0,
    "optimizer": Adam,
}

scheulder = {"scheduler": lr_scheduler.LambdaLR, "lr_lambda": lr_lambda}
trainer = Trainer(
    model=model, loss_func=CrossEntropyLoss(), scheduler=scheulder, **trainer_args
)

# Create a dictionary of data loaders
data_loader = {
    "test": test_data_loader,
    "train": train_data_loader,
    "validation": valdiation_data_loader,
}

trainer.train(data_loader=data_loader, n_epochs=10, pad_token_id=tokenizer.pad_token_id)
# trainer.run_one_epoch(
#     data_loader=data_loader["train"], verbose=True, pad_token_id=tokenizer.pad_token_id
# )
# first_data = next(iter(data_loader))
# print(first_data["attention_mask"].size())
