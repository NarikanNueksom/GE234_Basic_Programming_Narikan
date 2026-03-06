Code Repository

📗Machine Learning
วัตถุประสงค์ของ Code

เพื่อเปลี่ยน "ข้อมูล" ให้เป็น "คำทำนาย" (Prediction) การนำข้อมูลในอดีตมาให้คอมพิวเตอร์ประมวลผล เพื่อหา "แนวโน้ม" ในอนาคต
เพื่อหา "รูปแบบ" ที่มนุษย์มองไม่เห็น (Pattern Recognition) ใช้ Scikit-learn ในการวิเคราะห์ข้อมูลจำนวนมหาศาล เพื่อง่ายต่อการทำงาน
เพื่อ "ลดขั้นตอน" การเขียนโปรแกรม (Automation) Scikit-learn ช่วยลดขั้นตอนการป้อนคำสั่งในการเขียนโค้ด เช่น if-else
หลักการทำงานของ code

Machine Learning คือ การสั่งค่าคอมพิวเตอร์ให้เรียนรู้รูปแบบจากข้อมูลเพื่อทำนายผลเองได้

Scikit-learn คือ เครื่องมือใน Python ที่รวมอัลกอริทึมสำเร็จรูปไว้ให้เรียกใช้ได้ทันที

สรุปด้วยขั้นตอนมาตรฐาน (Scikit-learn Workflow)

Prepare: เตรียมข้อมูล โดยการ import model

Fit: ป้อนคำสั่งให้โมเดล เพื่อให้ทำการเรียนรู้ข้อมูลเหล่านั้น
Predict: หลังการเรียนรู้ผล แล้วจึงสามารถทำนายผลได้

วิธีการใช้งานโค้ด

นำเข้าข้อมูล (Data) เปรียบเทียบ: คุณมี กองผลไม้ อยู่ตรงหน้า (ข้อมูล) และคุณรู้ว่าลูกไหน หวาน ลูกไหน เปรี้ยว (เฉลย)
X (Features): คือลักษณะ (สี, ผิวขรุขระ, กลิ่น)

y (Target): คือคำตอบ (หวาน หรือ เปรี้ยว)
ป้อนคำสั่ง(Fit) เปรียบเทียบ: คุณหยิบผลไม้ส่งให้ "ลิ้นสมองกล" (Model) ลองชิมทีละลูก เพื่อให้มันจำว่า "อ๋อ... ผิวขรุขระแบบนี้คือเปรี้ยวนะ" ใน Scikit-learn เราใช้คำสั่งสั้นๆ แค่ model.fit(X, y) สมองเสี้ยวนี้ทำงานแค่: "จำรูปแบบ" (Pattern)

ทำนายผล (Predict) เปรียบเทียบ: คราวนี้หยิบผลไม้ลูกใหม่ที่ "ไม่เคยเห็น" มาวาง แล้วถามลิ้นนั้นว่า "ลูกนี้รสอะไร?" ถ้าลิ้นจำเก่ง (Model ดี) มันจะทายถูกว่า "ลูกนี้หวานแน่ๆ!" ใน Scikit-learn คือคำสั่ง model.predict(ข้อมูลใหม่)
การวิเคราะห์ปัจจัยเชิงพื้นที่และโมเดลการเรียนรู้ของเครื่องเพื่อทำนายผลสัมฤทธิ์ทางการศึกษาของนักเรียน

(Spatial Analysis and Machine Learning Models for Predicting Student Educational Outcomes)

1. การนำเข้าไลบรารี (Import Libraries)
ส่วนนี้เป็นการนำเข้าไลบรารีที่จำเป็นสำหรับการทำงานในโค้ดนี้:

numpy as np: ใช้สำหรับการคำนวณทางคณิตศาสตร์ โดยเฉพาะการทำงานกับอาร์เรย์และเมทริกซ์
pandas as pd: ใช้สำหรับจัดการและวิเคราะห์ข้อมูล โครงสร้างข้อมูลหลักคือ DataFrame ซึ่งคล้ายตารางข้อมูล
matplotlib.pyplot as plt: ใช้สำหรับสร้างกราฟและแผนภาพต่างๆ
seaborn as sns: ใช้สำหรับสร้างกราฟสถิติที่สวยงามและใช้งานง่าย สร้างขึ้นบน Matplotlib อีกที
sklearn.model_selection.train_test_split: ฟังก์ชันสำหรับแบ่งข้อมูลออกเป็นชุดสำหรับฝึกโมเดล (Training set) และชุดสำหรับทดสอบโมเดล (Test set)
sklearn.linear_model.LinearRegression: คลาสสำหรับสร้างแบบจำลองการถดถอยเชิงเส้น ซึ่งเป็นอัลกอริทึม Machine Learning สำหรับทำนายค่าตัวเลขต่อเนื่อง
sklearn.metrics.mean_squared_error, r2_score: ฟังก์ชันสำหรับคำนวณตัวชี้วัดประสิทธิภาพของโมเดลการถดถอย คือ Mean Squared Error (MSE) และ R-squared

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

2. การเตรียมข้อมูลจำลอง (Dataset Preparation)
data = {...}: สร้างข้อมูลจำลองในรูปแบบ Dictionary โดยมี 3 คอลัมน์หลัก ได้แก่ Hours_Studied (ชั่วโมงอ่านหนังสือ), Distance_KM (ระยะทางจากบ้านเป็นกิโลเมตร), และ Exam_Score (คะแนนสอบ)
df = pd.DataFrame(data): แปลง Dictionary data ให้เป็น Pandas DataFrame ชื่อ df เพื่อให้ง่ายต่อการจัดการข้อมูล

# 1. เตรียมข้อมูลจำลอง (Dataset)
# Hours: ชั่วโมงการอ่าน, Distance: ระยะทางจากบ้าน(กม.), Score: คะแนนสอบ
data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9, 10, 3, 5, 2, 8, 4, 1],
    'Distance_KM': [15, 12, 10, 5, 2, 3, 1, 0.5, 1, 20, 8, 25, 2, 10, 30],
    'Exam_Score': [55, 60, 68, 75, 82, 88, 92, 95, 98, 50, 70, 45, 90, 65, 35]
}

df = pd.DataFrame(data)

3. การกำหนดระยะทางเป็นอัตรภาคชั้น (Categorizing Distance)
bins = [...] และ labels = [...]: กำหนดช่วงของระยะทาง (bins) และป้ายชื่อ (labels) สำหรับแต่ละช่วง เช่น <5 KM สำหรับระยะทางน้อยกว่า 5 กม.
df['Distance_Category'] = pd.cut(...): ใช้ฟังก์ชัน pd.cut เพื่อแบ่งคอลัมน์ Distance_KM ออกเป็นหมวดหมู่ตามช่วงที่กำหนด และสร้างคอลัมน์ใหม่ชื่อ Distance_Category
distance_dummies = pd.get_dummies(...): ทำการแปลง Distance_Category เป็น One-Hot Encoding ซึ่งเป็นการแปลงข้อมูลหมวดหมู่ให้อยู่ในรูปแบบตัวเลขไบนารี (0 หรือ 1) โดยสร้างคอลัมน์ใหม่สำหรับแต่ละหมวดหมู่ เพื่อให้โมเดล Machine Learning สามารถนำไปใช้ได้

# 2. กำหนดระยะทางเป็นอัตรภาคชั้น
bins = [0, 5, 10, 15, 20, np.inf] # Bins: <5, 6-10, 11-15, 16-20, >20
labels = ['<5 KM', '6-10 KM', '11-15 KM', '16-20 KM', '>20 KM']
df['Distance_Category'] = pd.cut(df['Distance_KM'], bins=bins, labels=labels, right=True, include_lowest=True)

# Perform one-hot encoding on 'Distance_Category'
distance_dummies = pd.get_dummies(df['Distance_Category'], prefix='Distance')

4. กำหนดตัวแปร X (Features) และ y (Target)
X = pd.concat(...): กำหนดตัวแปรอิสระ (Features) หรือคุณลักษณะที่ใช้ในการทำนาย โดยรวมคอลัมน์ Hours_Studied เข้ากับคอลัมน์ distance_dummies ที่ได้จากการทำ One-Hot Encoding
y = df['Exam_Score']: กำหนดตัวแปรตาม (Target) หรือค่าที่เราต้องการทำนาย ซึ่งก็คือ Exam_Score

# 3. กำหนดตัวแปร X (Features) และ y (Target)
# รวม Hours_Studied กับ One-Hot Encoded Distance Categories
X = pd.concat([df[['Hours_Studied']], distance_dummies], axis=1)
y = df['Exam_Score']

5. แบ่งข้อมูลเป็นชุดฝึก (Train) และชุดทดสอบ (Test)
X_train, X_test, y_train, y_test = train_test_split(...): ใช้ฟังก์ชัน train_test_split เพื่อแบ่งข้อมูล X และ y ออกเป็น 2 ส่วน คือ:
X_train, y_train: ข้อมูลสำหรับฝึกโมเดล (80% ของข้อมูลทั้งหมด)
X_test, y_test: ข้อมูลสำหรับทดสอบประสิทธิภาพของโมเดล (20% ของข้อมูลทั้งหมด)
test_size=0.2: กำหนดสัดส่วนของข้อมูลทดสอบเป็น 20%
random_state=42: กำหนดค่า seed เพื่อให้การแบ่งข้อมูลมีผลลัพธ์เหมือนเดิมทุกครั้งที่รันโค้ด

# 4. แบ่งข้อมูลเป็นชุด Train (80%) และ Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

6. สร้างและฝึกสอนโมเดล (Training)
model = LinearRegression(): สร้างวัตถุ (object) ของโมเดล Linear Regression
model.fit(X_train, y_train): สั่งให้โมเดลเรียนรู้จากข้อมูลชุดฝึก (X_train และ y_train) ในขั้นตอนนี้ โมเดลจะหาความสัมพันธ์เชิงเส้นระหว่างตัวแปรอิสระและตัวแปรตาม

# 5. สร้างและฝึกสอนโมเดล (Training)
model = LinearRegression()
model.fit(X_train, y_train)

7. ทำนายผลและประเมินประสิทธิภาพ (Prediction and Evaluation)
y_pred = model.predict(X_test): ใช้โมเดลที่ฝึกแล้วทำนายคะแนนสอบจากข้อมูลชุดทดสอบ (X_test)
mse = mean_squared_error(y_test, y_pred): คำนวณค่า Mean Squared Error (MSE) ซึ่งเป็นค่าเฉลี่ยของกำลังสองของผลต่างระหว่างค่าจริง (y_test) กับค่าที่ทำนาย (y_pred) ยิ่งค่า MSE น้อย โมเดลยิ่งมีความแม่นยำ
r2 = r2_score(y_test, y_pred): คำนวณค่า R-squared (ค่าสัมประสิทธิ์การตัดสินใจ) ซึ่งบ่งบอกว่าโมเดลสามารถอธิบายความแปรปรวนของข้อมูลเป้าหมายได้ดีเพียงใด ค่า R-squared ที่ใกล้ 1.0 แสดงว่าโมเดลมีประสิทธิภาพสูง

# 6. ทำนายผลและประเมินประสิทธิภาพ
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

8. แสดงผลการวิเคราะห์ด้วยกราฟ (Data Visualization)
ส่วนนี้สร้างกราฟ 3 แบบเพื่อแสดงความสัมพันธ์ของข้อมูล:

กราฟ 1: Hours vs Score (Hours_Studied vs Exam_Score):

ใช้ sns.regplot เพื่อแสดงความสัมพันธ์ระหว่าง Hours_Studied และ Exam_Score พร้อมเส้นถดถอย (regression line) ซึ่งแสดงแนวโน้ม ยิ่งชั่วโมงอ่านหนังสือมาก คะแนนสอบก็มักจะสูงขึ้น
กราฟ 2: Distance (Original) vs Score (Distance_KM vs Exam_Score):

ใช้ sns.regplot เพื่อแสดงความสัมพันธ์ระหว่าง Distance_KM (ระยะทางเดิมที่เป็นตัวเลขต่อเนื่อง) และ Exam_Score เส้นถดถอยจะแสดงแนวโน้ม ยิ่งบ้านไกล คะแนนสอบก็อาจจะต่ำลง
กราฟ 3: Distance Category vs Score (หมวดหมู่ระยะทาง vs Exam_Score):

ใช้ sns.boxplot เพื่อแสดงการกระจายตัวของ Exam_Score สำหรับแต่ละหมวดหมู่ระยะทาง (Distance_Category) ทำให้เห็นว่าคะแนนสอบแตกต่างกันอย่างไรในแต่ละกลุ่มระยะทาง
print(f"--- ผลการทดสอบโมเดล ---")
print(f"ค่าความคลาดเคลื่อน (MSE): {mse:.2f}")
print(f"ความแม่นยำของโมเดล (R-squared): {r2:.2f}") # ยิ่งใกล้ 1.0 ยิ่งแม่นยำ

# 7. แสดงผลการวิเคราะห์ด้วยกราฟ (กราฟสำหรับ Distance_KM จะแสดงก่อนการแปลงเป็นประเภท)
plt.figure(figsize=(10, 5))

# กราฟความสัมพันธ์ระหว่าง ชั่วโมงการอ่าน กับ คะแนน
plt.subplot(1, 2, 1)
sns.regplot(x='Hours_Studied', y='Exam_Score', data=df, color='blue')
plt.title('Hours vs Score')

# กราฟความสัมพันธ์ระหว่าง ระยะทาง (แบบต่อเนื่องเดิม) กับ คะแนน
plt.subplot(1, 2, 2)
sns.regplot(x='Distance_KM', y='Exam_Score', data=df, color='red')
plt.title('Distance (Original) vs Score')

plt.tight_layout()
plt.show()

# กราฟสำหรับ Distance Category vs Score
plt.figure(figsize=(8, 5))
sns.boxplot(x='Distance_Category', y='Exam_Score', data=df.sort_values('Distance_Category'), palette='viridis')
plt.title('Distance Category vs Score')
plt.xlabel('Distance Category')
plt.ylabel('Exam Score')
plt.show()

9. ลองนำไปใช้งาน (Prediction for a New Student)
ส่วนนี้สาธิตการนำโมเดลที่ฝึกแล้วไปใช้ทำนายคะแนนของนักเรียนคนใหม่:

preprocess_new_distance function: เป็นฟังก์ชันที่สร้างขึ้นมาเพื่อแปลงค่าระยะทาง (distance_km) ของนักเรียนคนใหม่ให้เป็น One-Hot Encoding ในรูปแบบเดียวกับข้อมูลที่ใช้ฝึกโมเดล เพื่อให้สอดคล้องกับคุณลักษณะ X_train
new_student_hours = 6, new_student_distance_km = 15: กำหนดข้อมูลของนักเรียนคนใหม่ (อ่าน 6 ชม. บ้านไกล 15 กม.)
new_distance_encoded = preprocess_new_distance(...): แปลงระยะทางของนักเรียนคนใหม่เป็น One-Hot Encoding
new_student_features = pd.DataFrame(...): สร้าง DataFrame ของคุณลักษณะสำหรับนักเรียนคนใหม่ โดยรวม Hours_Studied กับ new_distance_encoded
new_student_features = new_student_features.reindex(...): ปรับลำดับคอลัมน์ของคุณลักษณะนักเรียนใหม่ให้ตรงกับ X_train ซึ่งเป็นสิ่งสำคัญเพื่อให้โมเดลสามารถทำนายได้อย่างถูกต้อง
predicted_score = model.predict(new_student_features): ใช้โมเดลที่ฝึกไว้ทำนายคะแนนสอบของนักเรียนคนใหม่
print(...): แสดงผลคะแนนที่ทำนายได้ออกมา
# 8. ลองนำไปใช้งาน (Prediction)
# สมมติว่านักเรียน A: อ่านหนังสือ 6 ชม. และบ้านไกล 15 กม.
# ฟังก์ชันช่วยแปลงระยะทางเป็น One-Hot Encoding สำหรับการทำนาย
def preprocess_new_distance(distance_km, bins, labels, all_distance_categories):
    category = pd.cut([distance_km], bins=bins, labels=labels, right=True, include_lowest=True)[0]
    # Create a dummy dataframe with all possible categories, then fill the one for the new distance
    dummy_df = pd.DataFrame(0, index=[0], columns=[f'Distance_{cat}' for cat in all_distance_categories])
    dummy_df[f'Distance_{category}'] = 1
    return dummy_df

# Get all possible categories from the training data for consistent encoding
all_distance_categories = sorted(distance_dummies.columns.str.replace('Distance_', '').tolist())

new_student_hours = 6
new_student_distance_km = 15

# Preprocess the new student's distance
new_distance_encoded = preprocess_new_distance(new_student_distance_km, bins, labels, all_distance_categories)

# Prepare the feature array for the new student
# Ensure the order of columns matches X_train
new_student_features = pd.DataFrame([[new_student_hours]], columns=['Hours_Studied'])
new_student_features = pd.concat([new_student_features, new_distance_encoded], axis=1)

# Reindex to ensure column order consistency with X_train
new_student_features = new_student_features.reindex(columns=X_train.columns, fill_value=0)

predicted_score = model.predict(new_student_features)
print(f"\nทำนายคะแนนของนักเรียน A: {predicted_score[0]:.2f} คะแนน")

ผู้จัดทำ

จิตราภรณ์ วรรณวิลัย

ชลิดา ทิตศานติกุล

จิรัชยา โชคกำเนิด

นรีกานต์ นึกสม
