import pandas as pd
import spacy
from spacy.tokens import DocBin
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
import torch
from fastapi import FastAPI
from elasticsearch import Elasticsearch

# Custom dataset class for BERT
class FinancialNERDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tags, tokenizer, label2id):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tags = self.tags[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        # Align labels with tokenized text
        word_ids = encoding.word_ids()
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(self.label2id[tags[word_idx]])
                
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids)
        }

# Training pipeline
def train_model(train_data, val_data, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, 
        num_labels=len(label2id)
    )
    
    training_args = TrainingArguments(
        output_dir="./financial-ner-model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data
    )
    
    trainer.train()
    return model, tokenizer

# FastAPI service
app = FastAPI()
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

@app.post("/extract_entities")
async def extract_entities(text: str):
    # Process text with model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    
    # Convert predictions to entities
    entities = []
    for idx, pred in enumerate(predictions[0]):
        if pred != -100:  # Skip special tokens
            entity = {
                'text': tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][idx]),
                'label': id2label[pred.item()],
                'position': idx
            }
            entities.append(entity)
    
    # Store in Elasticsearch
    doc = {
        'text': text,
        'entities': entities,
        'timestamp': datetime.now()
    }
    es.index(index='financial-ner', body=doc)
    
    return {"entities": entities}
