def plot_sales_and_customers(logger, train_data, test_data, plt, sns):
    try:
        # Check if 'Sales' and 'Customers' columns are present in the training data
        if 'Sales' in train_data.columns and 'Customers' in train_data.columns:
            # Plotting the sales and customers distributions for the training data
            plt.figure(figsize=(10, 6))
            sns.histplot(train_data['Sales'], kde=True, color='blue', label='Sales (Train)', bins=50)
            sns.histplot(train_data['Customers'], kde=True, color='orange', label='Customers (Train)', bins=50)
            plt.title('Sales and Customers Distribution (Train)')
            plt.legend()
            plt.show()
        else:
            logger.warning("Columns 'Sales' and/or 'Customers' are missing in the training data")
            raise KeyError("Columns 'Sales' and/or 'Customers' are missing in the training data")
    except Exception as e:
        logger.error("Error occurred in plot_sales_and_customers: %s", str(e))
        
def plot_promotions_distribution(logger, train_data, test_data, plt, sns):
    try:
        logger.info('Plotting distribution of Promo...')
        
        # Ensure 'Promo' column exists in both datasets
        if 'Promo' in train_data.columns and 'Promo' in test_data.columns:
            # Plot using subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

            sns.countplot(x='Promo', data=train_data, ax=axes[0], color='blue')
            axes[0].set_title('Promo Distribution (Train)')
            axes[0].set_xlabel('Promo')
            axes[0].set_ylabel('Count')

            sns.countplot(x='Promo', data=test_data, ax=axes[1], color='green')
            axes[1].set_title('Promo Distribution (Test)')
            axes[1].set_xlabel('Promo')

            plt.suptitle('Promo Distribution (Train vs Test)')
            plt.tight_layout()
            plt.show()
        else:
            logger.warning("Column 'Promo' is missing in either the training or test data")

    except Exception as e:
        logger.error("Error occurred in plot_promotions_distribution: %s", str(e))
