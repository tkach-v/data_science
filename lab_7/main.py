import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel('Data_Set_3.xls')

sns.scatterplot(x='Unit Cost', y='Units', data=df)
plt.title('Scatter Plot of Units vs. Unit Cost')
plt.xlabel('Unit Cost')
plt.ylabel('Units')
plt.show()

item_distribution = df['Item'].value_counts()
plt.pie(item_distribution, labels=item_distribution.index, autopct='%1.1f%%', startangle=90)
plt.title('Item Distribution')
plt.show()

df['OrderDate'] = pd.to_datetime(df['OrderDate'])
df['Month'] = df['OrderDate'].dt.month
monthly_sales = df.groupby('Month')['Total'].sum()
plt.plot(monthly_sales.index, monthly_sales.values, marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()

rep_sales = df.groupby('Rep')['Total'].sum().sort_values(ascending=False)
plt.bar(rep_sales.index, rep_sales.values)
plt.title('Sales by Sales Representatives')
plt.xlabel('Sales Representative')
plt.ylabel('Total Sales')
plt.show()

region_sales = df.groupby('Region')['Total'].sum().sort_values()
plt.bar(region_sales.index, region_sales.values)
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.show()

average_unit_cost = df['Unit Cost'].mean()
average_units_sold = df['Units'].mean()

print(f"Average Unit Cost: ${average_unit_cost:.2f}")
print(f"Average Units Sold: {average_units_sold:.2f}")

most_profitable_item = df.groupby('Item')['Total'].sum().idxmax()
most_popular_seller = df.groupby('Rep')['Units'].sum().idxmax()

print(f"Most Profitable Item: {most_profitable_item}")
print(f"Most Popular Seller: {most_popular_seller}")

plt.tight_layout()
plt.savefig('all_plots.png')
plt.show()
