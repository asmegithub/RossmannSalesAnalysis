from data_loader import load_data, preprocess_data
from feature_engineering import create_features
from analysis import analyze_sales
from visualization import plot_top_sales

def main():
    filepath = "data/raw/sales_data.csv"
    data = load_data(filepath)
    data = preprocess_data(data)
    data = create_features(data)
    
    top_sales = analyze_sales(data)
    print(top_sales)
    
    plot_top_sales(top_sales, title="Top 10 Stores by Sales")

if __name__ == "__main__":
    main()
