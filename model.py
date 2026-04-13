import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

url = 'https://gist.githubusercontent.com/netj/8836201/raw/iris.csv'
df = pd.read_csv(url)

print("--- 5 Baris Pertama ---")
print(df.head())
print("\n--- 5 Baris Terakhir ---")
print(df.tail())
print("\n--- 5 Baris Acak ---")
print(df.sample(5))

print(f"\nJumlah total data: {df.shape[0]} baris")
print(f"Jumlah kolom: {df.shape[1]} kolom")
print(f"Nama kolom: {list(df.columns)}")

print("\n--- Info Dataset ---")
df.info()
print("\n--- Statistik Deskriptif ---")
print(df.describe())

print("\n--- Pengecekan Missing Value ---")
print(df.isna().sum())

# Menangani missing value dengan mengisi teks "Missing"
df = df.fillna("Missing")

X = df.drop('variety', axis=1)
y = df['variety']

print("\nFitur (X) head:")
print(X.head())
print("\nLabel (y) head:")
print(y.head())

y = y.map({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2})

print("\nLabel (y) setelah encoding:")
print(y.unique())
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nJumlah data training: {len(X_train)}")
print(f"Jumlah data testing: {len(X_test)}")

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Training Model
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)

# Prediksi dan Akurasi
y_pred_nb = model_nb.predict(X_test)
akurasi_nb = accuracy_score(y_test, y_pred_nb)

print(f"\nAkurasi Model Naive Bayes: {akurasi_nb * 100:.2f}%")
print("\nClassification Report (Naive Bayes):")
print(classification_report(y_test, y_pred_nb, target_names=['Setosa', 'Versicolor', 'Virginica']))

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix

# Training Model
model_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
model_dt.fit(X_train, y_train)

# Prediksi dan Evaluasi
y_pred_dt = model_dt.predict(X_test)

print("\nClassification Report for Decision Tree:")
print(classification_report(y_test, y_pred_dt, target_names=['Setosa', 'Versicolor', 'Virginica']))

print("\nConfusion Matrix for Decision Tree:")
print(confusion_matrix(y_test, y_pred_dt))

# Visualisasi Pohon Keputusan
plt.figure(figsize=(20,10))
plot_tree(model_dt, feature_names=X.columns, class_names=['Setosa', 'Versicolor', 'Virginica'])
plt.title("Decision Tree Classifier")
plt.show()

# Visualisasi Confusion Matrix menggunakan Heatmap
cm = confusion_matrix(y_test, y_pred_dt)
class_names = ['Setosa', 'Versicolor', 'Virginica']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix for Decision Tree')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

from sklearn.model_selection import cross_val_score

# Menggunakan fungsi cross_val_score dengan 5-fold
cv_scores = cross_val_score(model_dt, X, y, cv=5)
print("\n--- Hasil Cross-Validation (5-fold) ---")
print(f"Skor tiap fold: {cv_scores}")
print(f"Rata-rata akurasi: {cv_scores.mean() * 100:.2f}%")

# Scatter plot sepal length vs petal length (Umum)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal.length', y='petal.length')
plt.title('Scatter Plot: Sepal Length vs Petal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.show()

# Visualisasi per kelas menggunakan 'hue'
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal.length', y='petal.length', hue='variety', palette='viridis')
plt.title('Scatter Plot per Kelas: Sepal Length vs Petal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend(title='Variety')
plt.show()

from sklearn.svm import SVC

# Challenge: Tambahkan model lain (SVM)
model_svm = SVC(random_state=42)
model_svm.fit(X_train, y_train)
y_pred_svm = model_svm.predict(X_test)
akurasi_svm = accuracy_score(y_test, y_pred_svm)

print("\n--- Perbandingan Performa Semua Model ---")
print(f"Akurasi Naive Bayes   : {akurasi_nb * 100:.2f}%")
print(f"Akurasi Decision Tree : {accuracy_score(y_test, y_pred_dt) * 100:.2f}%")
print(f"Akurasi SVM           : {akurasi_svm * 100:.2f}%")

\"\"\"
REFLEKSI PRAKTIKUM
1. Apa kesulitan selama praktikum?
   - (Silakan diisi sesuai pengalaman pribadi)

2. Apa insight yang diperoleh?
   - (Silakan diisi mengenai hal baru yang dipelajari)

3. Bagaimana cara meningkatkan performa model?
   - (Silakan diisi, contoh: mencari parameter terbaik/hyperparameter tuning, scaling data, dsb.)
\"\"\"
