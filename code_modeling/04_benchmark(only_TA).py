from common_imports import *

class TADataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # TA features
        ta_5 = torch.tensor(row['TA_5_X'])
        ta_20 = torch.tensor(row['TA_20_X'])
        ta_60 = torch.tensor(row['TA_60_X'])
        ta_120 = torch.tensor(row['TA_120_X'])

        y = row['Y']

        return {'ta_5': ta_5, 'ta_20': ta_20, 'ta_60': ta_60, 'ta_120': ta_120, 'y': y}

class Benchmark_LSTMModel(nn.Module):
    def __init__(self):
        super(Benchmark_LSTMModel, self).__init__()
        
        # ===========================================
        self.input_size = 25
        self.hidden_unit = 512
        self.num_layer = 12
        self.dropout = 0.5
        
        self.mhal_num_heads = 16
        
        self.fc_unit = 512
        # ===========================================
        
        self.lstm_5 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_unit, num_layers=self.num_layer, batch_first=True, dropout=self.dropout)
        self.lstm_20 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_unit, num_layers=self.num_layer, batch_first=True, dropout=self.dropout)
        self.lstm_60 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_unit, num_layers=self.num_layer, batch_first=True, dropout=self.dropout)
        self.lstm_120 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_unit, num_layers=self.num_layer, batch_first=True, dropout=self.dropout)
        
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_unit * 4, num_heads=self.mhal_num_heads)
        
        self.fc = nn.Linear(self.hidden_unit * 4, self.fc_unit)

    def forward(self, ta_5, ta_20, ta_60, ta_120):
        h_5, _ = self.lstm_5(ta_5)
        h_20, _ = self.lstm_20(ta_20)
        h_60, _ = self.lstm_60(ta_60)
        h_120, _ = self.lstm_120(ta_120)

        h = torch.cat((h_5[:, -1, :], h_20[:, -1, :], h_60[:, -1, :], h_120[:, -1, :]), dim=1)
        h, _ = self.attention(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
        h = h.squeeze(0)
        h = self.fc(h)
        return h
    
    def get_model_name(self):
        model_name = f"(LSTM_{self.input_size}_{self.hidden_unit}_{self.num_layer}_{self.fc_unit})_"
        return model_name

class StockPredictorTALSTM(nn.Module):
    def __init__(self):
        super(StockPredictorTALSTM, self).__init__()
        self.lstm_model = Benchmark_LSTMModel()
        self.lstm_model_name = self.lstm_model.get_model_name()
        
        # MLP Layer
        self.fc1 = nn.Linear(self.lstm_model.fc_unit, self.lstm_model.fc_unit // 2)  # 간단한 MLP
        self.bn1 = nn.BatchNorm1d(self.lstm_model.fc_unit // 2)
        self.fc2 = nn.Linear(self.lstm_model.fc_unit // 2, 1)
        self.dropout = nn.Dropout(p=0.5)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, ta_5, ta_20, ta_60, ta_120):
        h_lstm = self.lstm_model(ta_5, ta_20, ta_60, ta_120)
        
        # MLP layers
        h = self.fc1(h_lstm)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropout(h)
        
        output = self.fc2(h)
        
        return output
    
    def get_model_name(self):
        model_name = "Benchmark_"
        model_name += self.lstm_model_name
        model_name += f"(MLP_{self.lstm_model.fc_unit}_{self.lstm_model.fc_unit // 2})"
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
            ta_5 = batch['ta_5'].float().to(device)
            ta_20 = batch['ta_20'].float().to(device)
            ta_60 = batch['ta_60'].float().to(device)
            ta_120 = batch['ta_120'].float().to(device)
            labels = batch['y'].float().to(device)

            optimizer.zero_grad()
            outputs = model(ta_5, ta_20, ta_60, ta_120).squeeze(1)
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
            ta_5 = batch['ta_5'].float().to(device)
            ta_20 = batch['ta_20'].float().to(device)
            ta_60 = batch['ta_60'].float().to(device)
            ta_120 = batch['ta_120'].float().to(device)
            labels = batch['y'].float().to(device)

            outputs = model(ta_5, ta_20, ta_60, ta_120).squeeze(1)
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
    df = pd.read_pickle(f'./csv/3_Multimodal_data/v2_25/{day}/{ticker}_all_TA.pkl')
    df['TA_5_X'] = df['TA_5_X'].apply(lambda x: np.array(x))
    df['TA_20_X'] = df['TA_20_X'].apply(lambda x: np.array(x))
    df['TA_60_X'] = df['TA_60_X'].apply(lambda x: np.array(x))
    df['TA_120_X'] = df['TA_120_X'].apply(lambda x: np.array(x))

    split_num = 753

    train_df = df[:-split_num]
    test_df = df[-split_num:]

    return train_df, test_df

# ===================================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock prediction model training (TA only)')
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

            train_dataset = TADataset(train_df)
            test_dataset = TADataset(test_df)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = StockPredictorTALSTM().to(device)
            
            model_name = model.get_model_name()
            print(model_name)
            
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)
            
            train_model(model, train_loader, criterion, optimizer, args.epochs, device)
            save_model(model, f'./saved_model/benchmark_only_TA/{day}/{ticker}.pth')
            
            loaded_model = StockPredictorTALSTM().to(device)
            loaded_model.load_state_dict(torch.load(f'./saved_model/benchmark_only_TA/{day}/{ticker}.pth'))
            loaded_model.eval()
            
            results = test_model(loaded_model, test_loader, criterion, device)
            
            results_df = pd.DataFrame(results, columns=['Actual', 'Predicted'])
            
            output_dir = f'./csv/4_model_results/benchmark_only_TA/{day}/'
            os.makedirs(output_dir, exist_ok=True)
            results_df.to_csv(f'{output_dir}/{ticker}.csv', index=False)
