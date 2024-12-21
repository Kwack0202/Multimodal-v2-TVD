from common_imports import *

class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image features
        img_5 = Image.open(row['img_5_X']).convert('RGB')
        img_20 = Image.open(row['img_20_X']).convert('RGB')
        img_60 = Image.open(row['img_60_X']).convert('RGB')
        img_120 = Image.open(row['img_120_X']).convert('RGB')

        if self.transform:
            img_5 = self.transform(img_5)
            img_20 = self.transform(img_20)
            img_60 = self.transform(img_60)
            img_120 = self.transform(img_120)

        y = row['Y']

        return {'img_5': img_5, 'img_20': img_20, 'img_60': img_60, 'img_120': img_120, 'y': y}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Benchmark_ViTModel(nn.Module):
    def __init__(self):
        super(Benchmark_ViTModel, self).__init__()

        # ===========================================
        self.hidden_unit = 512
        self.num_layer = 12
        self.num_attention_heads = 16

        self.dropout = 0.5

        self.mhal_num_heads = 16
        self.intermediate_size = 1024

        self.fc_unit = 512
        # ===========================================

        config = ViTConfig(
            hidden_size=self.hidden_unit,
            num_hidden_layers=self.num_layer,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.dropout
        )
        self.vit = ViTModel(config)

        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_unit * 4, num_heads=self.mhal_num_heads)

        self.fc = nn.Linear(self.hidden_unit * 4, self.fc_unit)

    def forward(self, img_5, img_20, img_60, img_120):
        h_5 = self.vit(pixel_values=img_5).last_hidden_state[:, 0]
        h_20 = self.vit(pixel_values=img_20).last_hidden_state[:, 0]
        h_60 = self.vit(pixel_values=img_60).last_hidden_state[:, 0]
        h_120 = self.vit(pixel_values=img_120).last_hidden_state[:, 0]

        h = torch.cat((h_5, h_20, h_60, h_120), dim=1)
        h, _ = self.attention(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
        h = h.squeeze(0)
        h = self.fc(h)
        return h

    def get_model_name(self):
        model_name = f"(ViT_{self.hidden_unit}_{self.num_layer}_{self.num_attention_heads}_{self.intermediate_size}_{self.fc_unit})_"
        return model_name

class StockPredictorimgViT(nn.Module):
    def __init__(self):
        super(StockPredictorimgViT, self).__init__()
        self.vit_model = Benchmark_ViTModel()
        self.vit_model_name = self.vit_model.get_model_name()

        # MLP Layer
        self.fc1 = nn.Linear(self.vit_model.fc_unit, self.vit_model.fc_unit // 2)  # 간단한 MLP
        self.bn1 = nn.BatchNorm1d(self.vit_model.fc_unit // 2)
        self.fc2 = nn.Linear(self.vit_model.fc_unit // 2, 1)
        self.dropout = nn.Dropout(p=0.5)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, img_5, img_20, img_60, img_120):
        h_vit = self.vit_model(img_5, img_20, img_60, img_120)

        # Pass through MLP layers
        h = self.fc1(h_vit)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropout(h)

        output = self.fc2(h)  # Final prediction

        return output

    def get_model_name(self):
        model_name = "Benchmark_"
        model_name += self.vit_model_name
        model_name += f"(MLP_{self.vit_model.fc_unit}_{self.vit_model.fc_unit // 2})" 
        return model_name
    
# ===================================================================================================
# 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            img_5 = batch['img_5'].to(device)
            img_20 = batch['img_20'].to(device)
            img_60 = batch['img_60'].to(device)
            img_120 = batch['img_120'].to(device)
            labels = batch['y'].float().to(device)

            optimizer.zero_grad()
            outputs = model(img_5, img_20, img_60, img_120).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Apply sigmoid to convert logits to probabilities
            probabilities = torch.sigmoid(outputs)

            predicted = (probabilities > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

    # GPU 메모리 해제
    torch.cuda.empty_cache()

# 추론 함수
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    results = []
    with torch.no_grad():
        for batch in test_loader:
            img_5 = batch['img_5'].to(device)
            img_20 = batch['img_20'].to(device)
            img_60 = batch['img_60'].to(device)
            img_120 = batch['img_120'].to(device)
            labels = batch['y'].float().to(device)

            outputs = model(img_5, img_20, img_60, img_120).squeeze(1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Apply sigmoid to convert logits to probabilities
            probabilities = torch.sigmoid(outputs)

            predicted = (probabilities > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            results.extend(zip(labels.cpu().numpy(), probabilities.cpu().numpy()))

    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # GPU 메모리 해제
    torch.cuda.empty_cache()

    return results

# 모델 저장 함수
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

# 데이터 로드 및 처리
def load_and_split_data(ticker, day):
    df = pd.read_pickle(f'./csv/3_Multimodal_data/v2_25/{day}/{ticker}_all_img.pkl')

    split_num = 753

    train_df = df[:-split_num]
    test_df = df[-split_num:]

    return train_df, test_df

# ===================================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock prediction model training using ViT only')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    args = parser.parse_args()

    tickers = [
        "AAPL", "MSFT", "NVDA", "GOOG", "AMZN",

        "BRK-B", "LLY", "AVGO", "TSLA", "JPM", 

        "WMT", "UNH", "V", "XOM", "MA", 

        "PG", "COST", "JNJ", "ORCL", "HD", 

        "BAC", "KO", "NFLX", "MRK",  "CVX", 

        "CRM", "ADBE", "AMD", "PEP", "TMO"
    ]
    days = ['1day', '5day']

    for ticker in tickers:
        for day in days:
            
            print(f'Processing {ticker} for {day}...')

            train_df, test_df = load_and_split_data(ticker, day)

            # Use the updated ImageDataset
            train_dataset = ImageDataset(train_df, transform=transform)
            test_dataset = ImageDataset(test_df, transform=transform)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = StockPredictorimgViT().to(device)
            
            model_name = model.get_model_name()
            print(model_name)
            
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)
            
            train_model(model, train_loader, criterion, optimizer, args.epochs, device)
            save_model(model, f'./saved_model/benchmark_only_IMG/{day}/{ticker}.pth')
            
            loaded_model = StockPredictorimgViT().to(device)
            loaded_model.load_state_dict(torch.load(f'./saved_model/benchmark_only_IMG/{day}/{ticker}.pth'))
            loaded_model.eval()  # Set to evaluation mode

            results = test_model(loaded_model, test_loader, criterion, device)
            
            results_df = pd.DataFrame(results, columns=['Actual', 'Predicted'])
            
            output_dir = f'./csv/4_model_results/benchmark_only_IMG/{day}/'
            os.makedirs(output_dir, exist_ok=True)
            results_df.to_csv(f'{output_dir}/{ticker}.csv', index=False)