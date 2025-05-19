import pandas as pd
import matplotlib.pyplot as plt


df_heart = pd.read_csv('/Users/lukemacdonald/Documents/Coding/Python/csv_data/heart-1.csv')
df_iris = pd.read_csv('/Users/lukemacdonald/Documents/Coding/Python/csv_data/iris.csv')
df_titanic = pd.read_csv('/Users/lukemacdonald/Documents/Coding/Python/csv_data/titanic-1.csv')

# ---------- HEART DATASET ----------
df_heart['sex'] = df_heart['sex'].map({0: 'female', 1: 'male'})  # Replace 0/1 with labels
df_heart['target'] = df_heart['target'].map({0: 'no disease', 1: 'has disease'})  # Make target readable

fig_heart = plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
df_heart['age'].hist(bins=10, color='skyblue', edgecolor='black')
plt.title('Heart: Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sex_target_counts = df_heart.groupby('sex')['target'].value_counts().unstack()
sex_target_counts.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='coolwarm')
plt.title('Heart: Disease by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Heart Condition')
plt.tight_layout()

# ---------- IRIS DATASET ---------- this is an edited comment
fig_iris = plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
df_iris['sepal.length'].hist(bins=10, color='mediumpurple', edgecolor='black')
plt.title('Iris: Sepal Length Distribution')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
for variety in df_iris['variety'].unique():
    subset = df_iris[df_iris['variety'] == variety]
    plt.scatter(subset['sepal.length'], subset['petal.length'], label=variety)
plt.title('Iris: Sepal vs Petal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend(title='Variety')
plt.tight_layout()

# ---------- TITANIC DATASET ----------
fig_titanic = plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
df_titanic['Age'].dropna().hist(bins=10, color='orange', edgecolor='black')
plt.title('Titanic: Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')

plt.subplot(1, 2, 2)
sex_class_counts = df_titanic.groupby('Sex')['Pclass'].value_counts().unstack()
sex_class_counts.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='viridis')
plt.title('Titanic: Class Distribution by Sex')
plt.xlabel('Sex')
plt.ylabel('Passenger Count')
plt.legend(title='Pclass')
plt.tight_layout()

# Show all figures at once
plt.show()
# this is a new commentThis is Ali contribution
