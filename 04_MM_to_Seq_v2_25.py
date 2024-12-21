from common_imports import *

# ===================================================================================================
class MultiStockDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.transform = transform

    def __len__(self):
        return len(next(iter(self.data_dict.values()))['Y'])

    def __getitem__(self, idx):
        ta_dict = {key: torch.tensor(df['TA_X'].iloc[idx], dtype=torch.float) for key, df in self.data_dict.items()}
        img_dict = {key: Image.open(df['img_X'].iloc[idx]).convert('RGB') for key, df in self.data_dict.items()}
        if self.transform:
            img_dict = {key: self.transform(img) for key, img in img_dict.items()}
        label = torch.tensor(next(iter(self.data_dict.values()))['Y'].iloc[idx], dtype=torch.float)
        return ta_dict, img_dict, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class StockPredictor(nn.Module):
    def __init__(self):
        super(StockPredictor, self).__init__()
        
        # ===========================================
        # Number of layers for both LSTM and ViT
        self.hidden_unit = 512
        self.num_layers = 12 
        self.dropout_prob = 0.5
        
        # LSTM parameters
        self.input_size = 25
        
        # ViT parameters
        self.num_attention_heads = 16
        self.intermediate_size = 1024
        
        # Multi-head attention layer
        self.mhal_num_heads = 16
        
        # MLP Adapter
        self.mlp_hidden_unit = 512
        # ===========================================
        
        # LSTM for Time-series data
        self.lstm = nn.LSTM(input_size=self.input_size, 
                            hidden_size=self.hidden_unit, 
                            num_layers=self.num_layers, 
                            batch_first=True, 
                            dropout=self.dropout_prob)
        self.fc_ts = nn.Linear(self.hidden_unit, self.hidden_unit)  # 시계열 데이터 차원을 변경하는 FC 레이어

        # Vision Transformer for Image data
        config = ViTConfig(hidden_size=self.hidden_unit, 
                           num_hidden_layers=self.num_layers, 
                           num_attention_heads=self.num_attention_heads, 
                           intermediate_size=self.intermediate_size, 
                           hidden_dropout_prob=self.dropout_prob)
        self.vit = ViTModel(config)
        
        # Attention layers for Fusion
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.hidden_unit, num_heads=self.mhal_num_heads)
        
        # Fully Connected Layer for final prediction (MLP Layer)
        self.fc1 = nn.Linear(self.hidden_unit * 4, self.mlp_hidden_unit)  # Feature fusion layer
        self.bn1 = nn.BatchNorm1d(self.mlp_hidden_unit)      # Batch normalization
        self.fc2 = nn.Linear(self.mlp_hidden_unit, 1)  # Final prediction layer 
        self.dropout = nn.Dropout(p=self.dropout_prob)
        
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, ta_dict, img_dict):
        fused_outputs = []

        for key in ta_dict.keys():
            ta, _ = self.lstm(ta_dict[key])
            ta = self.fc_ts(ta[:, -1, :])  # (batch_size, hidden_size)

            vit_out = self.vit(img_dict[key]).last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

            # 어텐션 적용
            ta = ta.unsqueeze(0)  # (batch_size, 1, hidden_size)
            vit_out = vit_out.unsqueeze(0)  # (batch_size, 1, hidden_size)
            attn_output, _ = self.attention_layer(ta, vit_out, vit_out)
            attn_output = attn_output.squeeze(0)  # (batch_size, hidden_size)

            # 각 시점의 퓨전된 출력을 리스트에 저장
            fused_outputs.append(attn_output)

        # 모든 시점의 출력을 연결
        attn_output = torch.cat(fused_outputs, dim=1)  # (batch_size, num_pkl * hidden_size)

        # MLP를 통한 최종 예측
        attn_output = self.fc1(attn_output)
        attn_output = self.bn1(attn_output)
        attn_output = self.relu(attn_output)
        attn_output = self.dropout(attn_output)

        output = self.fc2(attn_output)

        return output
    
    def get_model_name(self):
        model_name = "MM_to_seq_"
        model_name += f"(LSTM_{self.input_size}_{self.hidden_unit}_{self.num_layers})_"
        model_name += f"(ViT_{self.hidden_unit}_{self.num_layers}_{self.num_attention_heads}_{self.intermediate_size})_"
        model_name += f"(MHAL_{self.hidden_unit}_{self.mhal_num_heads})_"
        model_name += f"(MLP_{self.hidden_unit*4}_{self.mlp_hidden_unit})"
        
        return model_name
        

# ===================================================================================================
# 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        correct = 0
        total = 0
        for ta_dict, img_dict, labels in train_loader:
            ta_dict = {key: ta.to(device) for key, ta in ta_dict.items()}
            img_dict = {key: img.to(device) for key, img in img_dict.items()}
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(ta_dict, img_dict).squeeze(1)
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
    model.eval()  # 모델을 평가 모드로 설정
    test_loss = 0.0
    correct = 0
    total = 0
    results = []

    with torch.no_grad():  # 그라디언트 계산 비활성화
        for ta_dict, img_dict, labels in test_loader:
            # 데이터를 디바이스로 이동
            ta_dict = {key: ta.to(device) for key, ta in ta_dict.items()}
            img_dict = {key: img.to(device) for key, img in img_dict.items()}
            labels = labels.to(device)

            # 모델 추론
            outputs = model(ta_dict, img_dict).squeeze(1)
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

# ===================================================================================================
# 모델 저장 함수
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

# 데이터 로드 및 처리
def load_and_process_data(day, ticker):
    data_paths = {
        '5': f'./csv/3_Multimodal_data/v2_25/{day}/{ticker}_5_update.pkl',
        '20': f'./csv/3_Multimodal_data/v2_25/{day}/{ticker}_20_update.pkl',
        '60': f'./csv/3_Multimodal_data/v2_25/{day}/{ticker}_60_update.pkl',
        '120': f'./csv/3_Multimodal_data/v2_25/{day}/{ticker}_120_update.pkl'
    }

    data = {key: pd.read_pickle(path) for key, path in data_paths.items()}

    split_num = 753

    train_data = {key: df.iloc[:-split_num] for key, df in data.items()}
    test_data = {key: df.iloc[-split_num:] for key, df in data.items()}
    
    return train_data, test_data

# ===================================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock prediction model training')
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

    # 메인 코드
    for day in days:
        for ticker in tickers:
            
            print(f'Processing {ticker} for {day}...')
            
            # ticker 별 데이터 준비 부분
            train_data, test_data = load_and_process_data(day, ticker)
            train_dataset = MultiStockDataset(train_data, transform=transform)
            test_dataset = MultiStockDataset(test_data, transform=transform)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            
            # divice 설정 및 모델 준비
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = StockPredictor().to(device)

            model_name = model.get_model_name()
            print(model_name)
            
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)

            # 학습 및 저장
            train_model(model, train_loader, criterion, optimizer, args.epochs, device)
            save_model(model, f'./saved_model/{model_name}/{day}/{ticker}.pth')

            # 나중에 모델 로드 및 평가
            loaded_model = StockPredictor().to(device)
            loaded_model.load_state_dict(torch.load(f'./saved_model/{model_name}/{day}/{ticker}.pth'))

            # 평가
            results = test_model(model, test_loader, criterion, device)
            
            results_df = pd.DataFrame(results, columns=['Actual', 'Predicted'])

            os.makedirs(f'./csv/4_model_results/{model_name}/{day}/', exist_ok=True)
            results_df.to_csv(f'./csv/4_model_results/{model_name}/{day}/{ticker}.csv', index=False)