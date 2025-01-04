import matplotlib.pyplot as plt

def plot_top_sales(sales_data, title='Top Sales'):
    """Visualizes the top sales."""
    sales_data.plot(kind='bar', figsize=(10, 6))
    plt.title(title)
    plt.show()
