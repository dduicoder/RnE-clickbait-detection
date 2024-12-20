import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

class NewsConsistencyDataset(Dataset):
    def __init__(self, headlines, article_bodies, labels, tokenizer, max_len=512):
        self.headlines = headlines
        self.article_bodies = article_bodies
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, idx):
        headline = str(self.headlines[idx])
        article = str(self.article_bodies[idx])
        
        # Tokenize inputs
        headline_encoding = self.tokenizer.encode_plus(
            headline,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        article_encoding = self.tokenizer.encode_plus(
            article,
            add_special_tokens=True,
            max_length=384,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'headline_ids': headline_encoding['input_ids'].flatten(),
            'headline_mask': headline_encoding['attention_mask'].flatten(),
            'article_ids': article_encoding['input_ids'].flatten(),
            'article_mask': article_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, _ = self.attention(query, key, value, key_padding_mask=key_padding_mask)
        return self.layer_norm(query + attn_output)

class NewsConsistencyChecker(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Cross-attention layers
        self.headline_to_article = CrossAttention(hidden_size)
        self.article_to_headline = CrossAttention(hidden_size)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2)  # 2 classes: consistent (0) or inconsistent (1)
        )

    def forward(self, headline_ids, headline_mask, article_ids, article_mask):
        # Process headline and article through BERT
        headline_output = self.bert(
            headline_ids,
            attention_mask=headline_mask,
            return_dict=True
        )
        article_output = self.bert(
            article_ids,
            attention_mask=article_mask,
            return_dict=True
        )
        
        headline_embeddings = headline_output.last_hidden_state
        article_embeddings = article_output.last_hidden_state
        
        # Cross-attention between headline and article
        headline_attended = self.headline_to_article(
            headline_embeddings,
            article_embeddings,
            article_embeddings,
            key_padding_mask=~article_mask.bool()
        )
        
        article_attended = self.article_to_headline(
            article_embeddings,
            headline_embeddings,
            headline_embeddings,
            key_padding_mask=~headline_mask.bool()
        )
        
        # Pool the embeddings
        headline_pool = torch.mean(headline_attended, dim=1)
        article_pool = torch.mean(article_attended, dim=1)
        headline_max, _ = torch.max(headline_attended, dim=1)
        article_max, _ = torch.max(article_attended, dim=1)
        
        # Concatenate different features
        combined = torch.cat([headline_pool, article_pool, headline_max, article_max], dim=1)
        
        # Classification
        logits = self.classifier(combined)
        return logits

def train_model(model, train_loader, val_loader, device, num_epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            inputs = {
                'headline_ids': batch['headline_ids'].to(device),
                'headline_mask': batch['headline_mask'].to(device),
                'article_ids': batch['article_ids'].to(device),
                'article_mask': batch['article_mask'].to(device)
            }
            
            labels = batch['labels'].to(device)
            
            outputs = model(**inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = {
                    'headline_ids': batch['headline_ids'].to(device),
                    'headline_mask': batch['headline_mask'].to(device),
                    'article_ids': batch['article_ids'].to(device),
                    'article_mask': batch['article_mask'].to(device)
                }
                
                labels = batch['labels'].to(device)
                outputs = model(**inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Epoch {epoch + 1}:')
        print(f'Training Loss: {total_loss / len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss / len(val_loader):.4f}')
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')

# Example usage:
def main():
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create dummy data (replace with your actual data)
    headlines = ["Example headline 1", "Example headline 2"]
    articles = ["Article body 1", "Article body 2"]
    labels = [0, 1]  # 0: consistent, 1: inconsistent
    
    # Create dataset
    dataset = NewsConsistencyDataset(headlines, articles, labels, tokenizer)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=8)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NewsConsistencyChecker().to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, device)

if __name__ == "__main__":
    main()