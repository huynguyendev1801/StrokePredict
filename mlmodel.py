#Import các thư viện cần thiết cho việc huấn luyện
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder

print('Đã import thư viện')

# Đọc dataset không bao gồm header
dataset = pd.read_csv('healthcare-dataset-stroke-data.csv', header=None)

# Định nghĩa tên các cột
columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
           'residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']

# Gán tên các cột cho dataset
dataset.columns = columns

# Thay thế giá trị 'NaN' của cột 'bmi' với giá trị trung bình
dataset['bmi'].replace(np.nan, dataset['bmi'].astype(float).mean(), inplace=True)


# Khởi tạo đối tượng LabelEncoder
label_encoder = LabelEncoder()
# Tạo dictionary để lưu trữ label_encoder
mapper={}
# Danh sách các cột cần mã hóa giá trị
categorical_columns = ['work_type', 'smoking_status']

# Mã hóa giá trị ở các cột chỉ định và lưu trữ vào dictionary để tham chiếu
for column in categorical_columns:
    mapper[column]= LabelEncoder()
for column in categorical_columns:
    dataset[column]=mapper[column].fit_transform(dataset.__getattr__(column))   
# Lưu dictionary vào file label_encoders_dict.pkl
joblib.dump(mapper, 'label_encoders_dict.pkl')
# In số dòng và cột của dataset
print('Số dòng và cột dataset:', dataset.shape)
# In danh sách các cột của dataset
print('Danh sách các cột của dataset:', dataset.columns)
#In tên cột và các dòng dữ liệu tương ứng từ dataset
print(dataset.head(10))
# Chia dữ liệu thành biến độc lập và biến phụ thuộc
X = dataset.iloc[:, 0:10].values  # Lấy các cột từ 0 đến 10
y = dataset.iloc[:, 10].values    # Lấy cột cuối cùng (cột 'stroke')

# In ra 1000 dòng đầu tiên của biến độc lập (X)
print('Danh sách 1000 dòng đầu tiên của biến độc lập X: ')
print(X[:1000, :])
                               
# In ra 1000 dòng đầu tiên của biến phụ thuộc (y)
print('Danh sách 1000 dòng đầu tiên của biến phụ thuộc y: ')
print(y[:1000])
# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Điều chỉnh cho tập dữ liệu khớp với thuật toán
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)
# Kết quả dự đoán
y_pred = classifier.predict(X_test)
print(' '.join(map(str, y_pred)))
# Lưu mô hình đã huấn luyện vào file randomforestmodel.pkl
joblib.dump(classifier, 'randomforestmodel.pkl') 
accuracy = accuracy_score(y_test, y_pred)
# Hiển thị độ chính xác
print(f'Độ chính xác: {accuracy * 100:.2f}%')