import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

url = 'https://gist.githubusercontent.com/netj/8836201/raw/iris.csv'
df = pd.read_csv(url)

print(df.head()) # 5 baris pertama
print(df.tail()) # 5 baris terakhir
print(df.sample(5)) # 5 baris acak

# Informasi jumlah data dan nama kolom
print(f"Jumlah total data: {df.shape[0]} baris")
print(f"Jumlah kolom: {df.shape[1]} kolom")
print(f"Nama kolom: {list(df.columns)}")

df.info()
df.describe()

print(df.isna().sum()) # Cek missing value
df = df.fillna("Missing") # Contoh pengisian data kosong

X = df.drop('variety', axis=1)
y = df['variety']

# Mapping: Setosa -> 0, Versicolor -> 1, Virginica -> 2
y = y.map({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2})

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Jumlah data training: {len(X_train)}")
print(f"Jumlah data testing: {len(X_test)}")

from sklearn.naive_bayes import GaussianNB

model_nb = GaussianNB()
model_nb.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier, plot_tree

model_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
model_dt.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Evaluasi Naive Bayes
y_pred_nb = model_nb.predict(X_test)
print(f"Akurasi Naive Bayes: {accuracy_score(y_test, y_pred_nb)*100:.2f}%")
print(classification_report(y_test, y_pred_nb))

# Evaluasi Decision Tree
y_pred_dt = model_dt.predict(X_test)
print("\nConfusion Matrix for Decision Tree:")
print(confusion_matrix(y_test, y_pred_dt))

plt.figure(figsize=(20,10))
plot_tree(model_dt, feature_names=X.columns, class_names=['Setosa', 'Versicolor', 'Virginica'])
plt.title("Decision Tree Classifier")
plt.show()

cm = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Setosa', 'Versicolor', 'Virginica'], 
            yticklabels=['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Langkah 19: Scatter plot sepal length vs petal length
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal.length', y='petal.length')
plt.title('Scatter Plot: Sepal Length vs Petal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.show()

# Langkah 20: Visualisasi per kelas menggunakan parameter 'hue'
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal.length', y='petal.length', hue='variety', palette='viridis')
plt.title('Scatter Plot per Kelas: Sepal Length vs Petal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend(title='Variety')
plt.show()

from sklearn.svm import SVC

# 1. Menambahkan model SVM
model_svm = SVC(random_state=42)
model_svm.fit(X_train, y_train)

# 2. Prediksi dan Evaluasi model SVM
y_pred_svm = model_svm.predict(X_test)

print("\nClassification Report for SVM:")
print(classification_report(y_test, y_pred_svm, target_names=['Setosa', 'Versicolor', 'Virginica']))

# 3. Bandingkan Akurasi Ketiga Model
akurasi_nb = accuracy_score(y_test, y_pred_nb)
akurasi_dt = accuracy_score(y_test, y_pred_dt)
akurasi_svm = accuracy_score(y_test, y_pred_svm)

print("\n--- Perbandingan Performa Model ---")
print(f"Akurasi Naive Bayes : {akurasi_nb*100:.2f}%")
print(f"Akurasi Decision Tree : {akurasi_dt*100:.2f}%")
print(f"Akurasi SVM           : {akurasi_svm*100:.2f}%")

"""
REFLEKSI PRAKTIKUM:
1. Apa kesulitan selama praktikum?
   Jawaban: (Isi dengan pengalaman Anda)

2. Apa insight yang diperoleh?
   Jawaban: (Isi dengan pemahaman baru yang Anda dapatkan)

3. Bagaimana cara meningkatkan performa model?
   Jawaban: (Contoh: Melakukan hyperparameter tuning, menambah jumlah data, atau mencoba teknik preprocessing yang berbeda)
"""
