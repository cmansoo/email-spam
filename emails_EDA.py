import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv('clean.csv')

# descriptive stats
print(df.head())
print(df.shape)
print(df.describe())
print(df['spam'].value_counts()) # number of spams vs hams

ham = df[df['spam'] == 0]
print(ham.describe())

spam = df[df['spam'] == 1]
print(spam.describe())

# pie chart
ham_perc = df['spam'].value_counts(normalize=True)[0]
spam_perc = df['spam'].value_counts(normalize=True)[1]
sizes = [ham_perc, spam_perc]
labels = ['ham', 'spam']
plt.title('Pie chart of ham and spam')
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.show()

# histogram
plt.title('Distribution of email text length')
plt.hist(ham['length'], bins=100, color='blue', label='Ham')
plt.hist(spam['length'], bins=100, color='orange', label='Spam')
plt.ylabel('frequency')
plt.xlabel('text length')
plt.legend()
plt.show()

# scatter plot
plt.title('Scatter plot of email text length')
plt.scatter(ham['spam'], ham['length'], color='blue', label='Ham')
plt.scatter(spam['spam'], spam['length'], color='orange', label='Spam')
plt.ylabel('Text length')
plt.legend()
plt.show()

# density plot
sb.displot(df, x='word_count', hue='spam', kind='kde').set(title='Density of word count')
plt.show()
