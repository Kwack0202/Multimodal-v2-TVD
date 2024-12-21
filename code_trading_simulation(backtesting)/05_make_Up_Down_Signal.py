from common_imports import *

# ===================================================================================================
# Define the paths
base_path = './csv/4_model_results/'
output_base_path = './csv/5_Up_Down_signal/'

# Subdirectories
model_types = []

# 폴더 이름을 읽어와서 리스트에 추가
if os.path.exists(base_path):
    for folder_name in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, folder_name)):
            model_types.append(folder_name)
            
            
days = ['1day', '5day']
tickers = [
        "AAPL", "MSFT", "NVDA", "GOOG", "AMZN",
    
        "BRK-B", "LLY", "AVGO", "TSLA", "JPM", 
    
        "WMT", "UNH", "V", "XOM", "MA", 
    
        "PG", "COST", "JNJ", "ORCL", "HD", 
    
        "BAC", "KO", "NFLX", "MRK",  "CVX", 
    
        "CRM", "ADBE", "AMD", "PEP", "TMO"
        ]

# Create the output base path if it does not exist1
os.makedirs(output_base_path, exist_ok=True)

# Function to process CSV files
def process_csv(file_path, output_path):
    df = pd.read_csv(file_path)
    mean_predicted = df['Predicted'].median()
    df['Predicted'] = df['Predicted'].apply(lambda x: 1 if x > mean_predicted else 0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

# Iterate over the folders and files
for model in tqdm(model_types):
    for day in days:
        for ticker in tickers:
            input_file_path = os.path.join(base_path, model, day, f'{ticker}.csv')
            output_file_path = os.path.join(output_base_path, model, day, f'{ticker}.csv')
            process_csv(input_file_path, output_file_path)
