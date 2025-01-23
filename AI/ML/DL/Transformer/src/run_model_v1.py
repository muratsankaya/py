from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from data.tokenizer_helinski_nlp_opus import tokenized_wmt14_subset, tokenizer
from src.transformer_v1 import Transformer
from src.trainer_v1 import Trainer

# print(tokenizer.vocab_size)
data_loader = DataLoader(tokenized_wmt14_subset, batch_size=16, shuffle=True)
model = Transformer(
    batch_size=16,
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

# trainer.train(data_loader=data_loader, n_epochs=20, pad_token_id=tokenizer.pad_token_id)
trainer.run_one_epoch(
    data_loader=data_loader, verbose=True, pad_token_id=tokenizer.pad_token_id
)
# first_data = next(iter(data_loader))
# print(first_data["attention_mask"].size())
