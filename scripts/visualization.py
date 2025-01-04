
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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


def analyze_holiday_sales(logger, train_data, plt, sns):
    try:
        logger.info("Analyzing sales behavior around holidays...")

        # Create new columns for sales before and after holidays
        train_data['Date'] = pd.to_datetime(train_data['Date'])
        train_data['DaysBeforeHoliday'] = train_data['StateHoliday'].replace('0', None).notnull().astype(int).cumsum()
        train_data['DaysAfterHoliday'] = train_data['StateHoliday'].replace('0', None).notnull().astype(int).iloc[::-1].cumsum()

        # Aggregate sales by holiday proximity
        sales_by_holiday = train_data.groupby(['StateHoliday', 'DaysBeforeHoliday', 'DaysAfterHoliday'])['Sales'].mean().reset_index()

        # Plot
        plt.figure(figsize=(15, 8))
        sns.lineplot(data=sales_by_holiday, x='DaysBeforeHoliday', y='Sales', hue='StateHoliday', label='Before Holiday')
        sns.lineplot(data=sales_by_holiday, x='DaysAfterHoliday', y='Sales', hue='StateHoliday', label='After Holiday')
        plt.title("Sales Before and After Holidays", fontsize=16)
        plt.xlabel("Days", fontsize=12)
        plt.ylabel("Average Sales", fontsize=12)
        plt.legend()
        plt.show()

        logger.info("Holiday sales analysis completed.")

    except Exception as e:
        logger.error("Error occurred during holiday sales analysis: %s", str(e))


def seasonal_behavior_analysis(logger, train_data, plt):
    try:
        logger.info("Analyzing seasonal purchase behaviors...")

        # Filter Christmas and Easter periods
        train_data['Month'] = train_data['Date'].dt.month
        christmas_data = train_data[train_data['Month'] == 12]
        easter_data = train_data[train_data['Month'].isin([3, 4])]

        # Plot seasonal sales
        plt.figure(figsize=(15, 8))
        plt.plot(christmas_data['Date'], christmas_data['Sales'], label="Christmas Sales", color="red")
        plt.plot(easter_data['Date'], easter_data['Sales'], label="Easter Sales", color="green")
        plt.title("Seasonal Sales Behavior", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Sales", fontsize=12)
        plt.legend()
        plt.show()

        logger.info("Seasonal behavior analysis completed.")

    except Exception as e:
        logger.error("Error occurred during seasonal behavior analysis: %s", str(e))


