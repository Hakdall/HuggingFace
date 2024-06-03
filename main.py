import torch
import torch.nn.functional as F
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
results = classifier(["We are happy to show you the :) Transformers Library.",
                 "We hope you don't hate it."])

for result in results:
    print(result)

tokens = tokenizer.tokenize("We are happy to show you the :) Transformers Library.")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer.tokenize("We are happy to show you the :) Transformers Library.")

X_train = ["We are happy to show you the :) Transformers Library.",
            "We hope you don't hate it."]

batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")

with torch.no_grad():
    outputs = model(**batch, labels=torch.tensor([1, 0]))
    print(outputs)
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print(labels)


model_name = "savasy/bert-base-turkish-sentiment-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

X_train_turkish = ["Sonuç iyi değildi", "Bu adil değildi", "Bu iyi değildi",
        "Bu ortalamaydı", "Bu iyidi", "Mavi bir araba sürüyor"]

batch = tokenizer(X_train_turkish, padding=True, truncation=True, max_length=512, return_tensors="pt")
print(batch)

with torch.no_grad():
    outputs = model(**batch)
    label_ids = torch.argmax(outputs.logits, dim=1)
    print(label_ids)
    labels = [model.config.id2label[label_id] for label_id in label_ids.tolist()]
    print(labels)
