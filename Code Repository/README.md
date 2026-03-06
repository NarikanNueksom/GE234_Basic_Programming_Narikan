{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "**Code Repository**\n",
        "\n",
        "\n",
        "📗Machine Learning\n"
      ],
      "metadata": {
        "id": "qTFRasLhQd3q"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**วัตถุประสงค์ของ Code**\n",
        "1. เพื่อเปลี่ยน \"ข้อมูล\" ให้เป็น \"คำทำนาย\" (Prediction)\n",
        "การนำข้อมูลในอดีตมาให้คอมพิวเตอร์ประมวลผล เพื่อหา \"แนวโน้ม\" ในอนาคต\n",
        "2. เพื่อหา \"รูปแบบ\" ที่มนุษย์มองไม่เห็น (Pattern Recognition)\n",
        "ใช้ Scikit-learn ในการวิเคราะห์ข้อมูลจำนวนมหาศาล เพื่อง่ายต่อการทำงาน\n",
        "3. เพื่อ \"ลดขั้นตอน\" การเขียนโปรแกรม (Automation)\n",
        "Scikit-learn ช่วยลดขั้นตอนการป้อนคำสั่งในการเขียนโค้ด เช่น if-else"
      ],
      "metadata": {
        "id": "oEC-RdGL5vtF"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**หลักการทำงานของ code**\n",
        "\n",
        "Machine Learning คือ การสั่งค่าคอมพิวเตอร์ให้เรียนรู้รูปแบบจากข้อมูลเพื่อทำนายผลเองได้\n",
        "\n",
        "Scikit-learn คือ เครื่องมือใน Python ที่รวมอัลกอริทึมสำเร็จรูปไว้ให้เรียกใช้ได้ทันที\n",
        "\n",
        "\n",
        "---\n",
        "\n",
        "\n",
        "**สรุปด้วยขั้นตอนมาตรฐาน (Scikit-learn Workflow)**\n",
        "\n",
        "Prepare: เตรียมข้อมูล โดยการ import model\n",
        "\n",
        "Fit: ป้อนคำสั่งให้โมเดล เพื่อให้ทำการเรียนรู้ข้อมูลเหล่านั้น\n",
        "\n",
        "Predict: หลังการเรียนรู้ผล แล้วจึงสามารถทำนายผลได้\n"
      ],
      "metadata": {
        "id": "Bc5dU1sZ5z9z"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**วิธีการใช้งานโค้ด**\n",
        "\n",
        "1.   นำเข้าข้อมูล (Data) เปรียบเทียบ: คุณมี กองผลไม้ อยู่ตรงหน้า (ข้อมูล) และคุณรู้ว่าลูกไหน หวาน ลูกไหน เปรี้ยว (เฉลย)\n",
        "\n",
        "    X (Features): คือลักษณะ (สี, ผิวขรุขระ, กลิ่น)\n",
        "\n",
        "    y (Target): คือคำตอบ (หวาน หรือ เปรี้ยว)\n",
        "\n",
        "2. ป้อนคำสั่ง(Fit)\n",
        "เปรียบเทียบ: คุณหยิบผลไม้ส่งให้ \"ลิ้นสมองกล\" (Model) ลองชิมทีละลูก เพื่อให้มันจำว่า \"อ๋อ... ผิวขรุขระแบบนี้คือเปรี้ยวนะ\"\n",
        "ใน Scikit-learn เราใช้คำสั่งสั้นๆ แค่ model.fit(X, y)\n",
        "สมองเสี้ยวนี้ทำงานแค่: \"จำรูปแบบ\" (Pattern)\n",
        "\n",
        "3. ทำนายผล (Predict)\n",
        "เปรียบเทียบ: คราวนี้หยิบผลไม้ลูกใหม่ที่ \"ไม่เคยเห็น\" มาวาง แล้วถามลิ้นนั้นว่า \"ลูกนี้รสอะไร?\"\n",
        "ถ้าลิ้นจำเก่ง (Model ดี) มันจะทายถูกว่า \"ลูกนี้หวานแน่ๆ!\"\n",
        "ใน Scikit-learn คือคำสั่ง model.predict(ข้อมูลใหม่)\n"
      ],
      "metadata": {
        "id": "881-ZoFB5ffa"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**การวิเคราะห์ปัจจัยเชิงพื้นที่และโมเดลการเรียนรู้ของเครื่องเพื่อทำนายผลสัมฤทธิ์ทางการศึกษาของนักเรียน**\n",
        "\n",
        "**(Spatial Analysis and Machine Learning Models for Predicting Student Educational Outcomes)**"
      ],
      "metadata": {
        "id": "AMdfD00n_wA5"
      }
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9829f6ec"
      },
      "source": [
        "### 1. การนำเข้าไลบรารี (Import Libraries)\n",
        "\n",
        "ส่วนนี้เป็นการนำเข้าไลบรารีที่จำเป็นสำหรับการทำงานในโค้ดนี้:\n",
        "\n",
        "*   **`numpy as np`**: ใช้สำหรับการคำนวณทางคณิตศาสตร์ โดยเฉพาะการทำงานกับอาร์เรย์และเมทริกซ์\n",
        "*   **`pandas as pd`**: ใช้สำหรับจัดการและวิเคราะห์ข้อมูล โครงสร้างข้อมูลหลักคือ DataFrame ซึ่งคล้ายตารางข้อมูล\n",
        "*   **`matplotlib.pyplot as plt`**: ใช้สำหรับสร้างกราฟและแผนภาพต่างๆ\n",
        "*   **`seaborn as sns`**: ใช้สำหรับสร้างกราฟสถิติที่สวยงามและใช้งานง่าย สร้างขึ้นบน Matplotlib อีกที\n",
        "*   **`sklearn.model_selection.train_test_split`**: ฟังก์ชันสำหรับแบ่งข้อมูลออกเป็นชุดสำหรับฝึกโมเดล (Training set) และชุดสำหรับทดสอบโมเดล (Test set)\n",
        "*   **`sklearn.linear_model.LinearRegression`**: คลาสสำหรับสร้างแบบจำลองการถดถอยเชิงเส้น ซึ่งเป็นอัลกอริทึม Machine Learning สำหรับทำนายค่าตัวเลขต่อเนื่อง\n",
        "*   **`sklearn.metrics.mean_squared_error, r2_score`**: ฟังก์ชันสำหรับคำนวณตัวชี้วัดประสิทธิภาพของโมเดลการถดถอย คือ Mean Squared Error (MSE) และ R-squared"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.metrics import mean_squared_error, r2_score"
      ],
      "metadata": {
        "id": "KpSNbLCO_to2"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "eb061233"
      },
      "source": [
        "### 2. การเตรียมข้อมูลจำลอง (Dataset Preparation)\n",
        "\n",
        "*   **`data = {...}`**: สร้างข้อมูลจำลองในรูปแบบ Dictionary โดยมี 3 คอลัมน์หลัก ได้แก่ `Hours_Studied` (ชั่วโมงอ่านหนังสือ), `Distance_KM` (ระยะทางจากบ้านเป็นกิโลเมตร), และ `Exam_Score` (คะแนนสอบ)\n",
        "*   **`df = pd.DataFrame(data)`**: แปลง Dictionary `data` ให้เป็น Pandas DataFrame ชื่อ `df` เพื่อให้ง่ายต่อการจัดการข้อมูล"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 1. เตรียมข้อมูลจำลอง (Dataset)\n",
        "# Hours: ชั่วโมงการอ่าน, Distance: ระยะทางจากบ้าน(กม.), Score: คะแนนสอบ\n",
        "data = {\n",
        "    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9, 10, 3, 5, 2, 8, 4, 1],\n",
        "    'Distance_KM': [15, 12, 10, 5, 2, 3, 1, 0.5, 1, 20, 8, 25, 2, 10, 30],\n",
        "    'Exam_Score': [55, 60, 68, 75, 82, 88, 92, 95, 98, 50, 70, 45, 90, 65, 35]\n",
        "}\n",
        "\n",
        "df = pd.DataFrame(data)"
      ],
      "metadata": {
        "id": "NuUxKD0eBbvy"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6dfa3d9b"
      },
      "source": [
        "### 3. การกำหนดระยะทางเป็นอัตรภาคชั้น (Categorizing Distance)\n",
        "\n",
        "*   **`bins = [...]`** และ **`labels = [...]`**: กำหนดช่วงของระยะทาง (bins) และป้ายชื่อ (labels) สำหรับแต่ละช่วง เช่น `<5 KM` สำหรับระยะทางน้อยกว่า 5 กม.\n",
        "*   **`df['Distance_Category'] = pd.cut(...)`**: ใช้ฟังก์ชัน `pd.cut` เพื่อแบ่งคอลัมน์ `Distance_KM` ออกเป็นหมวดหมู่ตามช่วงที่กำหนด และสร้างคอลัมน์ใหม่ชื่อ `Distance_Category`\n",
        "*   **`distance_dummies = pd.get_dummies(...)`**: ทำการแปลง `Distance_Category` เป็น **One-Hot Encoding** ซึ่งเป็นการแปลงข้อมูลหมวดหมู่ให้อยู่ในรูปแบบตัวเลขไบนารี (0 หรือ 1) โดยสร้างคอลัมน์ใหม่สำหรับแต่ละหมวดหมู่ เพื่อให้โมเดล Machine Learning สามารถนำไปใช้ได้"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 2. กำหนดระยะทางเป็นอัตรภาคชั้น\n",
        "bins = [0, 5, 10, 15, 20, np.inf] # Bins: <5, 6-10, 11-15, 16-20, >20\n",
        "labels = ['<5 KM', '6-10 KM', '11-15 KM', '16-20 KM', '>20 KM']\n",
        "df['Distance_Category'] = pd.cut(df['Distance_KM'], bins=bins, labels=labels, right=True, include_lowest=True)\n",
        "\n",
        "# Perform one-hot encoding on 'Distance_Category'\n",
        "distance_dummies = pd.get_dummies(df['Distance_Category'], prefix='Distance')"
      ],
      "metadata": {
        "id": "1RJFQ4fyAFum"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0da0919e"
      },
      "source": [
        "### 4. กำหนดตัวแปร X (Features) และ y (Target)\n",
        "\n",
        "*   **`X = pd.concat(...)`**: กำหนดตัวแปรอิสระ (Features) หรือคุณลักษณะที่ใช้ในการทำนาย โดยรวมคอลัมน์ `Hours_Studied` เข้ากับคอลัมน์ `distance_dummies` ที่ได้จากการทำ One-Hot Encoding\n",
        "*   **`y = df['Exam_Score']`**: กำหนดตัวแปรตาม (Target) หรือค่าที่เราต้องการทำนาย ซึ่งก็คือ `Exam_Score`"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 3. กำหนดตัวแปร X (Features) และ y (Target)\n",
        "# รวม Hours_Studied กับ One-Hot Encoded Distance Categories\n",
        "X = pd.concat([df[['Hours_Studied']], distance_dummies], axis=1)\n",
        "y = df['Exam_Score']"
      ],
      "metadata": {
        "id": "lvCQ5FhcAQR5"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1e77e16e"
      },
      "source": [
        "### 5. แบ่งข้อมูลเป็นชุดฝึก (Train) และชุดทดสอบ (Test)\n",
        "\n",
        "*   **`X_train, X_test, y_train, y_test = train_test_split(...)`**: ใช้ฟังก์ชัน `train_test_split` เพื่อแบ่งข้อมูล `X` และ `y` ออกเป็น 2 ส่วน คือ:\n",
        "    *   **`X_train, y_train`**: ข้อมูลสำหรับฝึกโมเดล (80% ของข้อมูลทั้งหมด)\n",
        "    *   **`X_test, y_test`**: ข้อมูลสำหรับทดสอบประสิทธิภาพของโมเดล (20% ของข้อมูลทั้งหมด)\n",
        "*   **`test_size=0.2`**: กำหนดสัดส่วนของข้อมูลทดสอบเป็น 20%\n",
        "*   **`random_state=42`**: กำหนดค่า seed เพื่อให้การแบ่งข้อมูลมีผลลัพธ์เหมือนเดิมทุกครั้งที่รันโค้ด"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 4. แบ่งข้อมูลเป็นชุด Train (80%) และ Test (20%)\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
      ],
      "metadata": {
        "id": "v2VkKF2XAaqS"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "c3092689"
      },
      "source": [
        "### 6. สร้างและฝึกสอนโมเดล (Training)\n",
        "\n",
        "*   **`model = LinearRegression()`**: สร้างวัตถุ (object) ของโมเดล Linear Regression\n",
        "*   **`model.fit(X_train, y_train)`**: สั่งให้โมเดลเรียนรู้จากข้อมูลชุดฝึก (`X_train` และ `y_train`) ในขั้นตอนนี้ โมเดลจะหาความสัมพันธ์เชิงเส้นระหว่างตัวแปรอิสระและตัวแปรตาม"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 5. สร้างและฝึกสอนโมเดล (Training)\n",
        "model = LinearRegression()\n",
        "model.fit(X_train, y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 80
        },
        "id": "0U-04TCnAs4L",
        "outputId": "5620e212-cc2e-4154-fbfa-2b3ddc85b452"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LinearRegression()"
            ],
            "text/html": [
              "<style>#sk-container-id-1 {\n",
              "  /* Definition of color scheme common for light and dark mode */\n",
              "  --sklearn-color-text: #000;\n",
              "  --sklearn-color-text-muted: #666;\n",
              "  --sklearn-color-line: gray;\n",
              "  /* Definition of color scheme for unfitted estimators */\n",
              "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
              "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
              "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
              "  --sklearn-color-unfitted-level-3: chocolate;\n",
              "  /* Definition of color scheme for fitted estimators */\n",
              "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
              "  --sklearn-color-fitted-level-1: #d4ebff;\n",
              "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
              "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
              "\n",
              "  /* Specific color for light theme */\n",
              "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
              "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
              "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
              "  --sklearn-color-icon: #696969;\n",
              "\n",
              "  @media (prefers-color-scheme: dark) {\n",
              "    /* Redefinition of color scheme for dark theme */\n",
              "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
              "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
              "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
              "    --sklearn-color-icon: #878787;\n",
              "  }\n",
              "}\n",
              "\n",
              "#sk-container-id-1 {\n",
              "  color: var(--sklearn-color-text);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 pre {\n",
              "  padding: 0;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 input.sk-hidden--visually {\n",
              "  border: 0;\n",
              "  clip: rect(1px 1px 1px 1px);\n",
              "  clip: rect(1px, 1px, 1px, 1px);\n",
              "  height: 1px;\n",
              "  margin: -1px;\n",
              "  overflow: hidden;\n",
              "  padding: 0;\n",
              "  position: absolute;\n",
              "  width: 1px;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-dashed-wrapped {\n",
              "  border: 1px dashed var(--sklearn-color-line);\n",
              "  margin: 0 0.4em 0.5em 0.4em;\n",
              "  box-sizing: border-box;\n",
              "  padding-bottom: 0.4em;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-container {\n",
              "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
              "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
              "     so we also need the `!important` here to be able to override the\n",
              "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
              "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
              "  display: inline-block !important;\n",
              "  position: relative;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-text-repr-fallback {\n",
              "  display: none;\n",
              "}\n",
              "\n",
              "div.sk-parallel-item,\n",
              "div.sk-serial,\n",
              "div.sk-item {\n",
              "  /* draw centered vertical line to link estimators */\n",
              "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
              "  background-size: 2px 100%;\n",
              "  background-repeat: no-repeat;\n",
              "  background-position: center center;\n",
              "}\n",
              "\n",
              "/* Parallel-specific style estimator block */\n",
              "\n",
              "#sk-container-id-1 div.sk-parallel-item::after {\n",
              "  content: \"\";\n",
              "  width: 100%;\n",
              "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
              "  flex-grow: 1;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-parallel {\n",
              "  display: flex;\n",
              "  align-items: stretch;\n",
              "  justify-content: center;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  position: relative;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-parallel-item {\n",
              "  display: flex;\n",
              "  flex-direction: column;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-parallel-item:first-child::after {\n",
              "  align-self: flex-end;\n",
              "  width: 50%;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-parallel-item:last-child::after {\n",
              "  align-self: flex-start;\n",
              "  width: 50%;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-parallel-item:only-child::after {\n",
              "  width: 0;\n",
              "}\n",
              "\n",
              "/* Serial-specific style estimator block */\n",
              "\n",
              "#sk-container-id-1 div.sk-serial {\n",
              "  display: flex;\n",
              "  flex-direction: column;\n",
              "  align-items: center;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  padding-right: 1em;\n",
              "  padding-left: 1em;\n",
              "}\n",
              "\n",
              "\n",
              "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
              "clickable and can be expanded/collapsed.\n",
              "- Pipeline and ColumnTransformer use this feature and define the default style\n",
              "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
              "*/\n",
              "\n",
              "/* Pipeline and ColumnTransformer style (default) */\n",
              "\n",
              "#sk-container-id-1 div.sk-toggleable {\n",
              "  /* Default theme specific background. It is overwritten whether we have a\n",
              "  specific estimator or a Pipeline/ColumnTransformer */\n",
              "  background-color: var(--sklearn-color-background);\n",
              "}\n",
              "\n",
              "/* Toggleable label */\n",
              "#sk-container-id-1 label.sk-toggleable__label {\n",
              "  cursor: pointer;\n",
              "  display: flex;\n",
              "  width: 100%;\n",
              "  margin-bottom: 0;\n",
              "  padding: 0.5em;\n",
              "  box-sizing: border-box;\n",
              "  text-align: center;\n",
              "  align-items: start;\n",
              "  justify-content: space-between;\n",
              "  gap: 0.5em;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 label.sk-toggleable__label .caption {\n",
              "  font-size: 0.6rem;\n",
              "  font-weight: lighter;\n",
              "  color: var(--sklearn-color-text-muted);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 label.sk-toggleable__label-arrow:before {\n",
              "  /* Arrow on the left of the label */\n",
              "  content: \"▸\";\n",
              "  float: left;\n",
              "  margin-right: 0.25em;\n",
              "  color: var(--sklearn-color-icon);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {\n",
              "  color: var(--sklearn-color-text);\n",
              "}\n",
              "\n",
              "/* Toggleable content - dropdown */\n",
              "\n",
              "#sk-container-id-1 div.sk-toggleable__content {\n",
              "  max-height: 0;\n",
              "  max-width: 0;\n",
              "  overflow: hidden;\n",
              "  text-align: left;\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-toggleable__content.fitted {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-toggleable__content pre {\n",
              "  margin: 0.2em;\n",
              "  border-radius: 0.25em;\n",
              "  color: var(--sklearn-color-text);\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-toggleable__content.fitted pre {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
              "  /* Expand drop-down */\n",
              "  max-height: 200px;\n",
              "  max-width: 100%;\n",
              "  overflow: auto;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
              "  content: \"▾\";\n",
              "}\n",
              "\n",
              "/* Pipeline/ColumnTransformer-specific style */\n",
              "\n",
              "#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  color: var(--sklearn-color-text);\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "/* Estimator-specific style */\n",
              "\n",
              "/* Colorize estimator box */\n",
              "#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-label label.sk-toggleable__label,\n",
              "#sk-container-id-1 div.sk-label label {\n",
              "  /* The background is the default theme color */\n",
              "  color: var(--sklearn-color-text-on-default-background);\n",
              "}\n",
              "\n",
              "/* On hover, darken the color of the background */\n",
              "#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {\n",
              "  color: var(--sklearn-color-text);\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "/* Label box, darken color on hover, fitted */\n",
              "#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
              "  color: var(--sklearn-color-text);\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "/* Estimator label */\n",
              "\n",
              "#sk-container-id-1 div.sk-label label {\n",
              "  font-family: monospace;\n",
              "  font-weight: bold;\n",
              "  display: inline-block;\n",
              "  line-height: 1.2em;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-label-container {\n",
              "  text-align: center;\n",
              "}\n",
              "\n",
              "/* Estimator-specific */\n",
              "#sk-container-id-1 div.sk-estimator {\n",
              "  font-family: monospace;\n",
              "  border: 1px dotted var(--sklearn-color-border-box);\n",
              "  border-radius: 0.25em;\n",
              "  box-sizing: border-box;\n",
              "  margin-bottom: 0.5em;\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-estimator.fitted {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-0);\n",
              "}\n",
              "\n",
              "/* on hover */\n",
              "#sk-container-id-1 div.sk-estimator:hover {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-estimator.fitted:hover {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
              "\n",
              "/* Common style for \"i\" and \"?\" */\n",
              "\n",
              ".sk-estimator-doc-link,\n",
              "a:link.sk-estimator-doc-link,\n",
              "a:visited.sk-estimator-doc-link {\n",
              "  float: right;\n",
              "  font-size: smaller;\n",
              "  line-height: 1em;\n",
              "  font-family: monospace;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  border-radius: 1em;\n",
              "  height: 1em;\n",
              "  width: 1em;\n",
              "  text-decoration: none !important;\n",
              "  margin-left: 0.5em;\n",
              "  text-align: center;\n",
              "  /* unfitted */\n",
              "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
              "  color: var(--sklearn-color-unfitted-level-1);\n",
              "}\n",
              "\n",
              ".sk-estimator-doc-link.fitted,\n",
              "a:link.sk-estimator-doc-link.fitted,\n",
              "a:visited.sk-estimator-doc-link.fitted {\n",
              "  /* fitted */\n",
              "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
              "  color: var(--sklearn-color-fitted-level-1);\n",
              "}\n",
              "\n",
              "/* On hover */\n",
              "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
              ".sk-estimator-doc-link:hover,\n",
              "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
              ".sk-estimator-doc-link:hover {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-3);\n",
              "  color: var(--sklearn-color-background);\n",
              "  text-decoration: none;\n",
              "}\n",
              "\n",
              "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
              ".sk-estimator-doc-link.fitted:hover,\n",
              "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
              ".sk-estimator-doc-link.fitted:hover {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-3);\n",
              "  color: var(--sklearn-color-background);\n",
              "  text-decoration: none;\n",
              "}\n",
              "\n",
              "/* Span, style for the box shown on hovering the info icon */\n",
              ".sk-estimator-doc-link span {\n",
              "  display: none;\n",
              "  z-index: 9999;\n",
              "  position: relative;\n",
              "  font-weight: normal;\n",
              "  right: .2ex;\n",
              "  padding: .5ex;\n",
              "  margin: .5ex;\n",
              "  width: min-content;\n",
              "  min-width: 20ex;\n",
              "  max-width: 50ex;\n",
              "  color: var(--sklearn-color-text);\n",
              "  box-shadow: 2pt 2pt 4pt #999;\n",
              "  /* unfitted */\n",
              "  background: var(--sklearn-color-unfitted-level-0);\n",
              "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
              "}\n",
              "\n",
              ".sk-estimator-doc-link.fitted span {\n",
              "  /* fitted */\n",
              "  background: var(--sklearn-color-fitted-level-0);\n",
              "  border: var(--sklearn-color-fitted-level-3);\n",
              "}\n",
              "\n",
              ".sk-estimator-doc-link:hover span {\n",
              "  display: block;\n",
              "}\n",
              "\n",
              "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
              "\n",
              "#sk-container-id-1 a.estimator_doc_link {\n",
              "  float: right;\n",
              "  font-size: 1rem;\n",
              "  line-height: 1em;\n",
              "  font-family: monospace;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  border-radius: 1rem;\n",
              "  height: 1rem;\n",
              "  width: 1rem;\n",
              "  text-decoration: none;\n",
              "  /* unfitted */\n",
              "  color: var(--sklearn-color-unfitted-level-1);\n",
              "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 a.estimator_doc_link.fitted {\n",
              "  /* fitted */\n",
              "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
              "  color: var(--sklearn-color-fitted-level-1);\n",
              "}\n",
              "\n",
              "/* On hover */\n",
              "#sk-container-id-1 a.estimator_doc_link:hover {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-3);\n",
              "  color: var(--sklearn-color-background);\n",
              "  text-decoration: none;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 a.estimator_doc_link.fitted:hover {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-3);\n",
              "}\n",
              "</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>LinearRegression</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LinearRegression.html\">?<span>Documentation for LinearRegression</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></div></label><div class=\"sk-toggleable__content fitted\"><pre>LinearRegression()</pre></div> </div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "cd34072d"
      },
      "source": [
        "### 7. ทำนายผลและประเมินประสิทธิภาพ (Prediction and Evaluation)\n",
        "\n",
        "*   **`y_pred = model.predict(X_test)`**: ใช้โมเดลที่ฝึกแล้วทำนายคะแนนสอบจากข้อมูลชุดทดสอบ (`X_test`)\n",
        "*   **`mse = mean_squared_error(y_test, y_pred)`**: คำนวณค่า **Mean Squared Error (MSE)** ซึ่งเป็นค่าเฉลี่ยของกำลังสองของผลต่างระหว่างค่าจริง (`y_test`) กับค่าที่ทำนาย (`y_pred`) ยิ่งค่า MSE น้อย โมเดลยิ่งมีความแม่นยำ\n",
        "*   **`r2 = r2_score(y_test, y_pred)`**: คำนวณค่า **R-squared** (ค่าสัมประสิทธิ์การตัดสินใจ) ซึ่งบ่งบอกว่าโมเดลสามารถอธิบายความแปรปรวนของข้อมูลเป้าหมายได้ดีเพียงใด ค่า R-squared ที่ใกล้ 1.0 แสดงว่าโมเดลมีประสิทธิภาพสูง"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 6. ทำนายผลและประเมินประสิทธิภาพ\n",
        "y_pred = model.predict(X_test)\n",
        "mse = mean_squared_error(y_test, y_pred)\n",
        "r2 = r2_score(y_test, y_pred)\n",
        "\n",
        "print(f\"--- ผลการทดสอบโมเดล ---\")\n",
        "print(f\"ค่าความคลาดเคลื่อน (MSE): {mse:.2f}\")\n",
        "print(f\"ความแม่นยำของโมเดล (R-squared): {r2:.2f}\") # ยิ่งใกล้ 1.0 ยิ่งแม่นยำ"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "s6fI7_5CA2An",
        "outputId": "5f454e78-ca95-44eb-df4a-bd04fc737667"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "--- ผลการทดสอบโมเดล ---\n",
            "ค่าความคลาดเคลื่อน (MSE): 34.13\n",
            "ความแม่นยำของโมเดล (R-squared): -1.05\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "f408f6d9"
      },
      "source": [
        "### 8. แสดงผลการวิเคราะห์ด้วยกราฟ (Data Visualization)\n",
        "\n",
        "ส่วนนี้สร้างกราฟ 3 แบบเพื่อแสดงความสัมพันธ์ของข้อมูล:\n",
        "\n",
        "*   **กราฟ 1: `Hours vs Score` (Hours_Studied vs Exam_Score)**:\n",
        "    *   ใช้ `sns.regplot` เพื่อแสดงความสัมพันธ์ระหว่าง `Hours_Studied` และ `Exam_Score` พร้อมเส้นถดถอย (regression line) ซึ่งแสดงแนวโน้ม ยิ่งชั่วโมงอ่านหนังสือมาก คะแนนสอบก็มักจะสูงขึ้น\n",
        "\n",
        "*   **กราฟ 2: `Distance (Original) vs Score` (Distance_KM vs Exam_Score)**:\n",
        "    *   ใช้ `sns.regplot` เพื่อแสดงความสัมพันธ์ระหว่าง `Distance_KM` (ระยะทางเดิมที่เป็นตัวเลขต่อเนื่อง) และ `Exam_Score` เส้นถดถอยจะแสดงแนวโน้ม ยิ่งบ้านไกล คะแนนสอบก็อาจจะต่ำลง\n",
        "\n",
        "*   **กราฟ 3: `Distance Category vs Score` (หมวดหมู่ระยะทาง vs Exam_Score)**:\n",
        "    *   ใช้ `sns.boxplot` เพื่อแสดงการกระจายตัวของ `Exam_Score` สำหรับแต่ละหมวดหมู่ระยะทาง (`Distance_Category`) ทำให้เห็นว่าคะแนนสอบแตกต่างกันอย่างไรในแต่ละกลุ่มระยะทาง"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 7. แสดงผลการวิเคราะห์ด้วยกราฟ (กราฟสำหรับ Distance_KM จะแสดงก่อนการแปลงเป็นประเภท)\n",
        "plt.figure(figsize=(10, 5))\n",
        "\n",
        "# กราฟความสัมพันธ์ระหว่าง ชั่วโมงการอ่าน กับ คะแนน\n",
        "plt.subplot(1, 2, 1)\n",
        "sns.regplot(x='Hours_Studied', y='Exam_Score', data=df, color='blue')\n",
        "plt.title('Hours vs Score')\n",
        "\n",
        "# กราฟความสัมพันธ์ระหว่าง ระยะทาง (แบบต่อเนื่องเดิม) กับ คะแนน\n",
        "plt.subplot(1, 2, 2)\n",
        "sns.regplot(x='Distance_KM', y='Exam_Score', data=df, color='red')\n",
        "plt.title('Distance (Original) vs Score')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# กราฟสำหรับ Distance Category vs Score\n",
        "plt.figure(figsize=(8, 5))\n",
        "sns.boxplot(x='Distance_Category', y='Exam_Score', data=df.sort_values('Distance_Category'), palette='viridis')\n",
        "plt.title('Distance Category vs Score')\n",
        "plt.xlabel('Distance Category')\n",
        "plt.ylabel('Exam Score')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "IO_Qgum5A6yk",
        "outputId": "fee52def-ddc1-4c00-d50a-6dcb8f1315e3"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x500 with 2 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAA90AAAHqCAYAAAAZLi26AAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAy9lJREFUeJzs3Xd4k1X7B/Bv0r33ZFZUEAHtC8oPBZGhgIAiKONFAWUpiLJEUEGZBcSFiogDcOBAQZaCSBVFKyovAgIWRGahhc50rzy/P27TJLSlSZrd7+e6esWeJ3lyWmnOcz/nnPtWKYqigIiIiIiIiIisTu3oDhARERERERG5KwbdRERERERERDbCoJuIiIiIiIjIRhh0ExEREREREdkIg24iIiIiIiIiG2HQTURERERERGQjDLqJiIiIiIiIbIRBNxEREREREZGNMOgmIiIiIiIishEG3URERERkM88//zxUKpWju+EUli5dilatWkGr1drk/KdOnYJKpcKaNWsser1KpcLzzz9v1T5d7vbbb8ftt99e9f2RI0fg6emJP//806bvS+RIDLqJbGTNmjVQqVT4/fffazx+++23o02bNnbulWvYs2cP+vTpg0aNGsHX1xdNmzZF//79sW7dOkd3jYioQdONbbovX19fxMfHo1evXli+fDny8/Ot8j7nz5/H888/jz/++MMq53MGGo0GS5YswVNPPQW12vgSvLCwEPPnz0e7du3g7++PkJAQdOnSBe+//z4URXFQj+2jdevW6Nu3L+bMmePorlThdQhZG4NuInIq69evx2233YaMjAw88cQTeO211/DAAw8gJycHb7/9tqO7R0REAObNm4cPPvgAb775JiZNmgQAmDx5Mtq2bYuDBw8aPffZZ59FcXGxWec/f/485s6d61ZB93vvvYeKigoMGzbMqD0jIwMdO3bE888/j7Zt2+KVV17B/PnzoVarMXLkSAwbNgyVlZUmvUezZs1QXFyMBx980KI+FhcX49lnn7XotfXxyCOPYOPGjThx4oTd3/tyvA4hW/B0dAeIyL4KCwsREBDg6G7U6vnnn0fr1q3xyy+/wNvb2+jYxYsX7dYPRVFQUlICPz8/u70nEZGr6NOnDzp06FD1/axZs5CcnIx+/frh7rvvxtGjR6s+Pz09PeHpyUvO1atX4+6774avr69R+8iRI3H06FFs3LgRd999d1X7448/jieffBLLli1DYmIinnrqqVrPXVFRAa1WC29v72rnN0d9XlsfPXv2RFhYGNauXYt58+Y5pA86vA4hW+BMN5ETqaiowPz589GiRQv4+PigefPmePrpp1FaWmr0vNr2XDVv3hyjRo2q+l63DHD37t2YMGECoqOj0bhxYwBAfn4+Jk+ejObNm8PHxwfR0dG444478L///a/W/n3++edV57vcW2+9BZVKVbUnKz09HQ899BAaN24MHx8fxMXF4Z577sGpU6eu+Ds4ceIEbrrppmoDHQBER0cbfa/VavHqq6+ibdu28PX1RVRUFHr37m20pN/U32nz5s3Rr18/7NixAx06dICfnx/eeustAEBubi4mT56MJk2awMfHB1dffTWWLFlisz15RESuqHv37pg9ezZOnz6NDz/8sKq9pj3dO3fuROfOnREaGorAwEC0bNkSTz/9NADg+++/x0033QQAeOihh6qWsuv2Kf/444+4//770bRpU/j4+KBJkyaYMmVKtdn0UaNGITAwEGlpaRgwYAACAwMRFRWF6dOnV5s5NmU8AYAPP/wQ7du3h5+fH8LDwzF06FCcPXu2zt/NyZMncfDgQfTs2dOo/ZdffsGOHTswatQoo4BbJykpCddccw2WLFlS9fPp9m0vW7YMr7zyStX4duTIkVr3dK9fvx6tW7eGr68v2rRpg40bN2LUqFFo3ry50fMuv77Q/b/7+++/MWrUKISGhiIkJAQPPfQQioqKjF67evVqdO/eHdHR0fDx8UHr1q3x5ptv1vm7AQAvLy/cfvvt2LRp0xWfx+sQXoe4Kt52JLKxvLw8ZGZmVmsvLy+v1jZmzBisXbsW9913H6ZNm4a9e/ciKSmp6g64pSZMmICoqCjMmTMHhYWFAGQp1+eff47HHnsMrVu3RlZWFvbs2YOjR4/iP//5T43n6du3LwIDA/HZZ5+ha9euRsc+/fRTXH/99VX71AcNGoTDhw9j0qRJaN68OS5evIidO3fizJkz1QZ5Q82aNcOuXbtw7ty5qhsEtRk9ejTWrFmDPn36YMyYMaioqMCPP/6IX375pWoGxpzfaWpqKoYNG4bx48dj7NixaNmyJYqKitC1a1ekpaVh/PjxaNq0KX7++WfMmjULFy5cwCuvvHLFPhIRNSQPPvggnn76aXzzzTcYO3Zsjc85fPgw+vXrh3bt2mHevHnw8fHB33//jZ9++gkAcN1112HevHmYM2cOxo0bhy5dugAAbrnlFgASQBYVFeHRRx9FREQEfv31V7z22ms4d+4c1q9fb/RelZWV6NWrFzp27Ihly5bh22+/xYsvvogWLVrg0UcfrXqeKePJwoULMXv2bAwePBhjxozBpUuX8Nprr+G2227D/v37ERoaWuvv5eeffwaAauPrli1bAAAjRoyo8XWenp7473//i7lz5+Knn34yCtpXr16NkpISjBs3Dj4+PggPD68xCNu2bRuGDBmCtm3bIikpCTk5ORg9ejQaNWpUa38vN3jwYCQkJCApKQn/+9//8M477yA6OhpLliypes6bb76J66+/HnfffTc8PT2xZcsWTJgwAVqtFhMnTqzzPdq3b49NmzZBo9EgODi4xufwOoTXIS5LISKbWL16tQLgil/XX3991fP/+OMPBYAyZswYo/NMnz5dAaAkJydXtQFQnnvuuWrv2axZM2XkyJHV+tC5c2eloqLC6LkhISHKxIkTzf65hg0bpkRHRxud78KFC4parVbmzZunKIqi5OTkKACUF154wezzv/vuuwoAxdvbW+nWrZsye/Zs5ccff1QqKyuNnpecnKwAUB5//PFq59BqtYqimPc7bdasmQJA2b59u9Fz58+frwQEBCjHjh0zap85c6bi4eGhnDlzxuyfkYjIVenGld9++63W54SEhCiJiYlV3z/33HOK4SXnyy+/rABQLl26VOs5fvvtNwWAsnr16mrHioqKqrUlJSUpKpVKOX36dFXbyJEjFQBVY5NOYmKi0r59+6rvTRlPTp06pXh4eCgLFy40On7o0CHF09OzWvvlnn32WQWAkp+fb9Q+YMAABYCSk5NT62s3bNigAFCWL1+uKIqinDx5UgGgBAcHKxcvXjR6ru6Y4e+tbdu2SuPGjY3e+/vvv1cAKM2aNTN6/eXXF7r/dw8//LDR8+69914lIiLCqK2m/y+9evVSrrrqKqO2rl27Kl27dq323HXr1ikAlL1791Y7ZojXIYLXIa6Fy8uJbOyNN97Azp07q321a9fO6HlfffUVAGDq1KlG7dOmTQMgd6otNXbsWHh4eBi1hYaGYu/evTh//rxZ5xoyZAguXryI77//vqrt888/h1arxZAhQwAAfn5+8Pb2xvfff4+cnByzzv/www9j+/btuP3227Fnzx7Mnz8fXbp0wTXXXFM1UwAAX3zxBVQqFZ577rlq59AtYzT3d5qQkIBevXoZta1fvx5dunRBWFgYMjMzq7569uyJyspK/PDDD2b9fERE7i4wMPCKWcx1M8KbNm2yaHms4R7XwsJCZGZm4pZbboGiKNi/f3+15z/yyCNG33fp0gX//PNP1femjCcbNmyAVqvF4MGDjcaC2NhYXHPNNfjuu++u2OesrCx4enoiMDDQqF33ewoKCqr1tbpjGo3GqH3QoEGIioq64vueP38ehw4dwogRI4zeu2vXrmjbtu0VX2uopt9hVlaWUZ8M/7/oVvl17doV//zzD/Ly8up8j7CwMACocXWgIV6H8DrEFTHoJrKxm2++GT179qz2pRtcdE6fPg21Wo2rr77aqD02NhahoaE4ffq0xX1ISEio1rZ06VL8+eefaNKkCW6++WY8//zzRhchtenduzdCQkLw6aefVrV9+umnuPHGG3HttdcCAHx8fLBkyRJ8/fXXiImJwW233YalS5ciPT3dpP726tULO3bsQG5uLn744QdMnDgRp0+fRr9+/aqSmJw4cQLx8fEIDw+v9Tzm/k5r+j0dP34c27dvR1RUlNGXbomfPZOqEBG5goKCgisGkUOGDMGtt96KMWPGICYmBkOHDsVnn31mcgB+5swZjBo1CuHh4VX7tHVLjS8P7nT7bA2FhYUZBWKmjCfHjx+Hoii45pprqo0HR48etXgs0P2ernSTorbAvKYx63K6ce7ycbC2tto0bdrU6HvdNYzh71G3/D0gIAChoaGIioqq2qdvStCt/Fsara6a7rwO4XWIK+KebiInU9dgcyW1lRSpKfPl4MGD0aVLF2zcuBHffPMNXnjhBSxZsgQbNmxAnz59an0PHx8fDBgwABs3bsSKFSuQkZGBn376CYsWLTJ63uTJk9G/f398+eWX2LFjB2bPno2kpCQkJycjMTHRpJ/H398fXbp0QZcuXRAZGYm5c+fi66+/xsiRI016vY6pv9Oafk9arRZ33HEHZsyYUeNrdAM8EREB586dQ15e3hUDOj8/P/zwww/47rvvsG3bNmzfvh2ffvopunfvjm+++abayixDlZWVuOOOO5CdnY2nnnoKrVq1QkBAANLS0jBq1KhqgfuVzmUOrVYLlUqFr7/+usZzXj6DfbmIiAhUVFQgPz/fKHi+7rrr8OWXX+LgwYO47bbbanytrgRb69atjdrtmdW6tt+jLlA+ceIEevTogVatWuGll15CkyZN4O3tja+++govv/yySTdUdAF8ZGTkFZ/H6xBjvA5xDQy6iZxEs2bNoNVqcfz4cVx33XVV7RkZGcjNzUWzZs2q2sLCwpCbm2v0+rKyMly4cMGs94yLi8OECRMwYcIEXLx4Ef/5z3+wcOHCKwbdgMxSrF27Frt27cLRo0ehKErVki5DLVq0wLRp0zBt2jQcP34cN954I1588UWjrLam0iUk0f2MLVq0wI4dO5CdnV3rXWZzfqe1adGiBQoKCqplnCUiouo++OADAKi2RPZyarUaPXr0QI8ePfDSSy9h0aJFeOaZZ/Ddd9+hZ8+etQYphw4dwrFjx7B27Vqj5GM7d+60uM+mjCctWrSAoihISEiwKMhp1aoVAMlibri9rF+/fkhKSsL7779fY9BdWVmJdevWISwsDLfeeqvZ76sb5/7+++9qx2pqs9SWLVtQWlqKzZs3G82K17Xs3tDJkyehVqtN+v3yOoRcDZeXEzmJu+66CwCqZaF86aWXAEjGTp0WLVpU28OzatWqWme6L1dZWVltqVd0dDTi4+OrlbCoSc+ePREeHo5PP/0Un376KW6++WajJVFFRUUoKSkxek2LFi0QFBRU5/l37dpVY7tuX1TLli0ByF42RVEwd+7cas/V3Xk353dam8GDByMlJQU7duyodiw3NxcVFRV1noOIqCFITk7G/PnzkZCQgOHDh9f6vOzs7GptN954IwBUjREBAQEAUO0Gs27GVfc5r/vvV1991eJ+mzKeDBw4EB4eHpg7d67Re+uek5WVdcX36NSpEwBUK0F2yy23oGfPnli9ejW2bt1a7XXPPPMMjh07hhkzZlg0sx0fH482bdrg/fffR0FBQVX77t27cejQIbPPV5ua/r/k5eVh9erVJp9j3759uP766xESElLnc3kdIngd4jo4003kJG644QaMHDkSq1atQm5uLrp27Ypff/0Va9euxYABA9CtW7eq544ZMwaPPPIIBg0ahDvuuAMHDhzAjh076lySpZOfn4/GjRvjvvvuww033IDAwEB8++23+O233/Diiy/W+XovLy8MHDgQn3zyCQoLC7Fs2TKj48eOHUOPHj0wePBgtG7dGp6enti4cSMyMjIwdOjQK577nnvuQUJCAvr3748WLVqgsLAQ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Mw3HuVF999KG6tOnD7Zs2YLly5fjpZdeuuV2I0aMQHp6OgYOHAh3d/dG/V+ILB1buomszMSJE9G9e3e88cYbSEpKwv79+7Fo0SJ8+OGHhm1iY2OxaNEiREdHo7q6GlOmTLnp2+66pKamol+/fggODsbVq1exe/duhIWF1fu6WbNmoaysDP3790dgYCAKCgqwcOFCVFVV4eGHHwYgkvAzZ84gMzMTrVu3houLC7RaLXr16oV58+ahbdu2uHjxIl577TWjfY8ePRrz58/H5MmTMWrUKGRkZNRbNG7GjBl45JFHEBAQgCFDhkCpVOLIkSM4fvw43nzzTcTFxSE4OBjJycl45513UFRUhFdffbXe/ycREVF9KioqkJeXh+rqauTn5+O7775DWloaHnnkETzzzDM3bf/uu+/Cz88PkZGRUCqVWL9+PXx9fQ2zcgQFBWHnzp2IiYmBWq2Gu7s77r77bnz55ZdISEiAQqHA9OnTjeYBb4ihQ4di7ty5GDRoENLS0uDn54fDhw/D398f0dHR9d5LGyMuLg5ff/01EhISoNfrsWjRopu2CQsLw6VLl+Do6NiofRNZA7Z0E1mZrl27Yt26dVizZg06deqEGTNm4PXXX8fw4cMN28yfPx9t2rTBfffdhyeffBKTJk1q0E2suroaKSkpCAsLQ9++fREcHGyUzN/KAw88gP/85z945plnEBoain79+iEvLw/btm0zVCpNTExE37598dBDD8HLywuff/45ADEtyLVr1xAVFYXU1NSbbuQBAQHYsGEDNm3ahC5dumDp0qX1jg2Lj4/H119/jW3btqF79+7o1asXFixYgMDAQACiW/vGjRvx119/oUePHhg1alSjqrESERHdynfffQc/Pz8EBQWhb9++2L17NxYuXIjNmzcbeqRdz8XFBW+//Ta6deuG7t274+zZs/j2228Nrcnz58/H9u3b0aZNG0RGRgIQibq7uzvuueceJCQkID4+Hl27dm1UnCqVCtu2bYO3tzf69++Pzp07Y968eYYY67uXNlZsbCy++eYbrFy5EikpKTcNgwMAT0/Peru9E1kjhVTXFU9ERERERERETcaWbiIiIiIiIiIzYdJNRPUaPXq00ZQh1z9Gjx4td3hERETUjFatWnXLzwXh4eFyh0dkcdi9nIjqdfHiRRQVFdW5ztXVFd7e3s0cEREREcmluLgY+fn5da5zcHC443HfRLaKSTcRERERERGRmbB7OREREREREZGZMOkmIiIiIiIiMhMm3URERERERERmwqSbiIiIiIiIyEyYdBMRERERERGZCZNuIiIiIiIiIjNh0k1ERERERERkJky6iYiIiIiIiMzk/wNV0CbHt0WPHgAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/tmp/ipykernel_247/1059880748.py:19: FutureWarning: \n",
            "\n",
            "Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.\n",
            "\n",
            "  sns.boxplot(x='Distance_Category', y='Exam_Score', data=df.sort_values('Distance_Category'), palette='viridis')\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 800x500 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAArcAAAHWCAYAAABt3aEVAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAVApJREFUeJzt3X18j/X////7a3byWpttiM00ey1nc5pEmpOESYooFZKTiN4lnaAT5XRO0znvkD411jslRLxFyWmEJHrLOZk52xS2GW3Ynr8/+nl9vWxjY/OaY7fr5bLL+30cx/N4Ho/Xcbym+47X83i+bMYYIwAAAMACPNxdAAAAAFBQCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcAAACwDMItAAAALINwCwAAAMsg3AIAAMAyCLcACsSIESNks9ncXQYAoJgj3ALIZvr06bLZbM4fu92u0NBQtW7dWhMnTtSpU6cK5DhHjhzRiBEjtGXLlgLpr6jYt2+fnn76ad16662y2+0KCAhQ48aN9cEHH+jvv//Od3+TJ0/W9OnTC77QYmrNmjVq06aNKlSoILvdrooVK6pdu3aaOXOmu0sDUABsxhjj7iIAFC3Tp0/Xk08+qZiYGEVEROjcuXNKTEzUypUrtXTpUlWsWFELFixQnTp1nPucP39e58+fl91uz/NxfvnlFzVo0ECxsbHq2bNnIbyS62/RokV69NFH5ePjo+7du6tWrVo6e/as1qxZo7lz56pnz56aNm1avvqsVauWbr75Zq1cubJwii5GZs+erU6dOqlu3brq3LmzSpUqpf3792v16tXy8vLSihUr3F0igGvk6e4CABRdbdq0Uf369Z3LgwcP1vLly9W2bVs9+OCD2rFjh3x9fSVJnp6e8vQs3v+k7N+/X507d1Z4eLiWL1+u8uXLO7f169dPe/fu1aJFi9xYYeE6ffq0/Pz83F3GZY0YMUI1atTQ+vXr5e3t7bLt2LFj160OY4zS09Odvz8ACg7DEgDkS4sWLTR06FAdOHBA//nPf5zrcxpzu3TpUjVp0kRBQUHy9/dXtWrV9Prrr0uSVq5cqQYNGkiSnnzySecQiAsfv//444969NFHVbFiRfn4+CgsLEwvvfRSto/1e/bsKX9/fx0+fFgdOnSQv7+/ypYtq0GDBikzM9OlbVZWlj744APVrl1bdrtdZcuW1X333adffvnFpd1//vMf3XHHHfL19VXp0qXVuXNnHTx48IrnZsKECUpLS9Mnn3ziEmwvqFy5sl544QXncmxsrFq0aKFy5crJx8dHNWrU0JQpU1z2cTgc2rZtm1atWuU8R/fcc49ze3Jysl588UWFhYXJx8dHlStX1ptvvqmsrCyXfo4fP65u3bopICBAQUFB6tGjh3777TeXc37B8uXL1bRpU/n5+SkoKEjt27fXjh07XNpcuN7bt2/X448/rlKlSqlJkyaKjY2VzWbT5s2bs73+sWPHqkSJEjp8+HCO52/OnDmy2WxatWpVtm0fffSRbDabfv/9d0lSYmKinnzySd1yyy3y8fFR+fLl1b59e8XHx+fY9wX79u1TgwYNsgVbSSpXrpzLcl7eL+fPn9eoUaNUqVIl+fj4yOFw6PXXX1dGRoZLXw6HQ23bttV3332n+vXry9fXVx999JGkvF9DAHlTvG+zALgq3bp10+uvv67vv/9effr0ybHNtm3b1LZtW9WpU0cxMTHy8fHR3r17tXbtWklS9erVFRMTo2HDhqlv375q2rSpJKlRo0aS/vn4+MyZM3rmmWdUpkwZ/fzzz5o0aZIOHTqk2bNnuxwrMzNTrVu3VsOGDfX222/rhx9+0DvvvKNKlSrpmWeecbbr3bu3pk+frjZt2uipp57S+fPn9eOPP2r9+vXOO9RjxozR0KFD9dhjj+mpp57Sn3/+qUmTJunuu+/W5s2bFRQUlOt5WbhwoW699Vbna7iSKVOmqGbNmnrwwQfl6emphQsX6tlnn1VWVpb69esnSXr//ffVv39/+fv764033pAkBQcHS5LOnDmjZs2a6fDhw3r66adVsWJF/fTTTxo8eLCOHj2q999/X9I/Ia1du3b6+eef9cwzzygyMlLffPONevToka2mH374QW3atNGtt96qESNG6O+//9akSZPUuHFj/frrr3I4HC7tH330UVWpUkVjx46VMUaPPPKI+vXrp88//1y33367S9vPP/9c99xzjypUqJDj+XjggQfk7++vr776Ss2aNXPZNmvWLNWsWVO1atWSJHXs2FHbtm1T//795XA4dOzYMS1dulQJCQnZarxYeHi4li1bpkOHDumWW27JtZ2Ut/fLU089pRkzZuiRRx7RwIEDtWHDBo0bN047duzQvHnzXPrbtWuXunTpoqefflp9+vRRtWrV8nwNAeSDAYBLxMbGGklm48aNubYJDAw0t99+u3N5+PDh5uJ/Ut577z0jyfz555+59rFx40YjycTGxmbbdubMmWzrxo0bZ2w2mzlw4IBzXY8ePYwkExMT49L29ttvN3fccYdzefny5UaSef7557P1m5WVZYwxJj4+3pQoUcKMGTPGZfvWrVuNp6dntvUXS0lJMZJM+/btc21zqZxeY+vWrc2tt97qsq5mzZqmWbNm2dqOGjXK+Pn5md27d7usf+2110yJEiVMQkKCMcaYuXPnGknm/fffd7bJzMw0LVq0yHb+69ata8qVK2eOHz/uXPfbb78ZDw8P0717d+e6C9e7S5cu2erq0qWLCQ0NNZmZmc51v/76a67X+tJ9y5UrZ86fP+9cd/ToUePh4eG8xidPnjSSzFtvvXXZvnLyySefGEnG29vbNG/e3AwdOtT8+OOPLrUak7f3y5YtW4wk89RTT7lsHzRokJFkli9f7lwXHh5uJJklS5a4tM3rNQSQdwxLAHBV/P39LztrwoU7nN98881Vfbx68VjE06dP66+//lKjRo1kjMnxI+9//etfLstNmzbVH3/84VyeO3eubDabhg8fnm3fC8Mpvv76a2VlZemxxx7TX3/95fwJCQlRlSpVLvuwUWpqqiSpZMmSV/UaU1JS9Ndff6lZs2b6448/lJKScsX9Z8+eraZNm6pUqVIu9UZHRyszM1OrV6+WJC1ZskReXl4ud9k9PDycd4cvOHr0qLZs2aKePXuqdOnSzvV16tRRq1at9O2332ar4dLzLkndu3fXkSNHXM7X559/Ll9fX3Xs2PGyr6lTp046duyYy8Nzc+bMUVZWljp16iTpn/Pm7e2tlStX6uTJk5ft71K9evXSkiVLdM8992jNmjUaNWqUmjZtqipVquinn35ytsvL++XC+RgwYIDL9oEDB0pStvHVERERat26tcu6vF5DAHlHuAVwVdLS0i4b5Dp16qTGjRvrqaeeUnBwsDp37qyvvvoqz0E3ISHBGbIujKO98FH1pcHvwnjIi5UqVcol+Ozbt0+hoaEuoe1Se/bskTFGVapUUdmyZV1+duzYcdkHjgICAiQpX9OkrV27VtHR0c6xrWXLlnWOSc5LuN2zZ4+WLFmSrdbo6GhJ/+8BqQMHDqh8+fK66aabXPavXLmyy/KBAwckSdWqVct2rOrVq+uvv/7S6dOnXdZHRERka9uqVSuVL19en3/+uaR/hkV88cUXat++/RXD/3333afAwEDNmjXLuW7WrFmqW7euqlatKkny8fHRm2++qcWLFys4OFh33323JkyYoMTExMv2fUHr1q313XffKTk5WatXr1a/fv104MABtW3b1nnO8vJ+OXDggDw8PLKdx5CQEAUFBTnP5wU5nau8XkMAeceYWwD5dujQIaWkpGT7j/rFfH19tXr1aq1YsUKLFi3SkiVLNGvWLLVo0ULff/+9SpQokeu+mZmZatWqlU6cOKFXX31VkZGR8vPz0+HDh9WzZ89sAflyfeVHVlaWbDabFi9enGOf/v7+ue4bEBCg0NBQ5wNPV7Jv3z61bNlSkZGRevfddxUWFiZvb299++23eu+99/L0R0BWVpZatWqlV155JcftF8JgYcrpaf8SJUro8ccf18cff6zJkydr7dq1OnLkiJ544okr9ufj46MOHTpo3rx5mjx5spKSkrR27VqNHTvWpd2LL76odu3aaf78+fruu+80dOhQjRs3TsuXL8821jc3N910k5o2baqmTZvq5ptv1siRI7V48eIcxyJfTl6/vCSnc1UUriFgNYRbAPn22WefSVK2j1gv5eHhoZYtW6ply5Z69913NXbsWL3xxhtasWKFoqOjcw0FW7du1e7duzVjxgx1797duX7p0qVXXXOlSpX03Xff6cSJE7nejatUqZKMMYqIiLiqUNG2bVtNmzZN69atU1RU1GXbLly4UBkZGVqwYIEqVqzoXJ/T0IfczlOlSpWUlpbmvMuXm/DwcK1YsUJnzpxxuXu7d+/ebO2kfx58utTOnTt1880353mqr+7du+udd97RwoULtXjxYpUtW/aK75cLOnXqpBkzZmjZsmXasWOHjDHOIQkXq1SpkgYOHKiBAwdqz549qlu3rt555x2XWTzy6sIDYkePHnX2faX3S3h4uLKysrRnzx5Vr17duT4pKUnJycnO83k5eb2GAPKOYQkA8mX58uUaNWqUIiIi1LVr11zbnThxItu6unXrSpJzmqQLQSk5Odml3YW7puai75gxxuiDDz646ro7duwoY4xGjhyZbduF4zz88MMqUaKERo4c6XLsC22OHz9+2WO88sor8vPz01NPPaWkpKRs2/ft2+d8DTm9xpSUFMXGxmbbz8/PL9s5kqTHHntM69at03ff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btu1a6f58+dr7NixmjNnjnx9fVWnTh398MMPatasWWHUCAAAAOTJVU0F9sADD7jMWgAAAAAUBVc9z630z1eGzpo1S6dPn1arVq1UpUqVgqoLKHJMehofi7uZSU9zdwkAgCIuz+F2wIABOnfunCZNmiRJOnv2rO666y5t375dN910k1555RUtXbqU+WVhWecPbHZ3CQAA4AryHG6///57jR071rn8+eefKyEhQXv27FHFihXVq1cvjR49WosWLSqUQgF3Y7YE9zPpafyRAQC4rDyH24SEBJepvr7//ns98sgjCg8PlyS98MILuv/++wu+QqCIYLYE92NYCADgSvL89bseHh4uc82uX7/e5VvAgoKCdPLkyYKtDgAAAMiHPIfb6tWra+HChZKkbdu2KSEhQc2bN3duP3DggIKDgwu+QgAAACCP8jws4ZVXXlHnzp21aNEibdu2Tffff78iIiKc27/99lvdeeedhVIkAAAAkBd5vnP70EMP6dtvv1WdOnX00ksvadasWS7bb7rpJj377LMFXiAAAACQV/ma57Zly5Zq2bJljtuGDx9eIAUBRRXz3Lof89wCAK7kmr7EASgOgoKC5O3to7NMQVUkeHv7KCgoyN1lAACKKMItcAUhISGaM2e2kpOT3V3KNYmPj9ewYcMUExMjh8Ph7nKuWlBQkEJCQtxdBgCgiCLcAnkQEhJimUDlcDgUGRnp7jIAACgUeX6gDAAAACjqCLcAAACwjHwPSzh+/LiGDRumFStW6NixY8rKcn1+/MSJEwVWHAAAAJAf+Q633bp10969e9W7d28FBwfLZrMVRl0AAABAvuU73P74449as2aNbrvttsKoBwAAALhq+Q63kZGR+vvvvwujFqDYSk9PV3x8fKEe40L/hX0ch8Mhu91eqMcAACA3NmOMyc8OGzdu1GuvvaZhw4apVq1a8vLyctkeEBBQoAVeD6mpqQoMDFRKSsoNWT9ufDt37lT37t3dXUaBiIuLY6oxAECBy2tey/ed26CgIKWmpqpFixYu640xstlsyszMzH+1QDHncDgUFxfn7jIKxI38BREAgBtfvsNt165d5eXlpZkzZ/JAGVBA7HY7dzsBACgA+Q63v//+uzZv3qxq1aoVRj0AAADAVcv3lzjUr19fBw8eLIxaAAAAgGuS7zu3/fv31wsvvKCXX35ZtWvXzvZAWZ06dQqsOAAAACA/8j1bgodH9pu9Npvthn6gjNkSAAAAirZCmy1h//7911QYAAAAUFjyHW7Dw8MLow4AAADgmuU73F6wfft2JSQk6OzZsy7rH3zwwWsuCgAAALga+Q63f/zxhx566CFt3brVOdZWknO+2xtxzC0AAACsId9Tgb3wwguKiIjQsWPHdNNNN2nbtm1avXq16tevr5UrVxZCiQAAAEDe5PvO7bp167R8+XLdfPPN8vDwkIeHh5o0aaJx48bp+eef1+bNmwujTgAAAOCK8n3nNjMzUyVLlpQk3XzzzTpy5Iikfx4027VrV8FWBwAAAORDvu/c1qpVS7/99psiIiLUsGFDTZgwQd7e3po2bZpuvfXWwqgRAAAAyJN8h9shQ4bo9OnTkqSYmBi1bdtWTZs2VZkyZTRr1qwCLxAAAADIq3x/Q1lOTpw4oVKlSjlnTLjR8A1lAAAARVte81q+x9z++eef2daVLl1aNptNW7duzW93AAAAQIHJd7itXbu2Fi1alG3922+/rTvvvLNAigIAAACuRr7D7YABA9SxY0c988wz+vvvv3X48GG1bNlSEyZM0MyZMwujRgAAACBPrmrM7ebNm9WtWzdlZGToxIkTatiwoT799FOFhIQURo2FjjG3AAAARVuhjbmVpMqVK6tWrVqKj49XamqqOnXqdMMGWwAAAFhHvsPt2rVrVadOHe3Zs0f/+9//NGXKFPXv31+dOnXSyZMnC6NGAAAAIE/yHW5btGihTp06af369apevbqeeuopbd68WQkJCapdu3aBF3j48GE98cQTKlOmjHx9fVW7dm398ssvzu3GGA0bNkzly5eXr6+voqOjtWfPngKvAwAAAEVfvsPt999/r/Hjx8vLy8u5rlKlSlq7dq2efvrpAi3u5MmTaty4sby8vLR48WJt375d77zzjkqVKuVsM2HCBE2cOFFTp07Vhg0b5Ofnp9atWys9Pb1AawEAAEDRVyBf4lBYXnvtNa1du1Y//vhjjtuNMQoNDdXAgQM1aNAgSVJKSoqCg4M1ffp0de7cOU/H4YEyAACAoi2veS3PX797//3364svvlBgYKAkafz48frXv/6loKAgSdLx48fVtGlTbd++/doqv8iCBQvUunVrPfroo1q1apUqVKigZ599Vn369JEk7d+/X4mJiYqOjnbuExgYqIYNG2rdunW5htuM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vn//fiPJbN68OV/9xsbGmsDAwKs65uX2OXnypGnUqJGpU6eOOXr0aLY2Q4YMMQEBAebMmTPO9a1atTJDhw41ksyKFSvydVyrye3cd+rUyTzxxBPX3P+ECRNMRESEc3ny5MmmVKlSJiMjw7nu1VdfNdWqVcu1j4vfcwkJCaZatWqmRYsW5tSpU9naDBkyxFSpUsVl/6pVqzqv9/79+6/5NVnBiBEjTHBwsBk4cKD53//+l+/9ua4Fgzu3N6CtW7dq4MCBuuWWW9S9e3eVLVtWK1as0G233ebSbvz48SpTpoxuv/12vfXWW3n6azItLU0PPPCAtm/frrVr16patWrZ2gQHB6t169aaMWOGpH/u7syaNUu9evUqmBdYjCxYsED169fXo48+qnLlyun222/Xxx9/fM39rlu3TkFBQapfv75zXXR0tDw8PLRhw4bL7st7wFr69eunm2++WXfeeac+/fRTmTxMbZ6YmKhmzZpJklatWqWQkJBsbe644w45HA7NnTtXkpSQkKDVq1erW7duBfsCLCQrK0uLFi1S1apV1bp1a5UrV04NGzbMdfjC5aSkpKh06dLO5XXr1unuu++Wt7e3c13r1q21a9cunTx58rJ97dq1S40bN1aNGjX07bff5nj3/cEHH9TJkye1Zs0aSdKaNWt08uRJtWvXLt+1W9mrr76qDz74QDt27FC9evVUr149TZw4UX/++Wee9ue6FgzC7Q3i+PHj+uCDD1SvXj3Vr19ff/zxhyZPnqyjR49q8uTJioqKcmn//PPP68svv9SKFSv09NNPa+zYsXrllVeueJxRo0Zpy5Yt+vHHHxUWFpZru169emn69OkyxmjOnDmqVKmS6tate60vs9j5448/NGXKFFWpUkXfffednnnmGT3//PPO0Hi1EhMTVa5cOZd1np6eKl26tBITEy+7L+8B64iJidFXX32lpUuXqmPHjnr22Wc1adKkK+73wgsv6OzZs1q6dGmu43mlf94Dn376qSRp+vTpuv/++1W2bNmCKt9yjh07prS0NI0fP1733Xefvv/+ez300EN6+OGHtWrVqjz3s3fvXk2aNElPP/20c11iYqKCg4Nd2l1YvtLvfPfu3VW5cmXNnj0712+88vLy0hNPPOG83p9++qmeeOIJeXl55bnu4sBut6tTp05atGiRDh8+rO7du2v69OmqUKGCOnTo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u3btMrGxscbX19fExsa6nLuAgADz9ttvm71795q9e/eazz//3JQqVcqkp6c727377rvG4XCYrKysK75mANZEuAVgWZcLt506dTLVq1d3Ll8cbvv3729atGiRa0C6uO3l1KxZ00yaNMm5HB4ebp544gnnclZWlilXrpyZMmWKMcaYjz76yJQsWdIcP348x/5atmxpxo4d67Lus88+M+XLl8+xfVJSkpFk3n333SvWeqmNGzcaSebUqVPGGGNWrFjhEkiNMSY9Pd3cdNNNzvB9Qe/evU2XLl2MMca8+uqrplatWi7b33jjDZe+Hn/8cdOqVSuXNi+//LKpUaOGczk8PNx06NDBpc3ff/9tSpUqZWbNmuVcV6dOnVzDPoDigWEJAIolY4xsNluO23r27KktW7aoWrVqev755/X9999fsb+0tDQNGjRI1atXV1BQkPz9/bVjxw4lJCS4tKtTp47z/9tsNoWEhDg//t+yZYtuv/12lS5dOsdj/Pbbb4qJiZG/v7/zp0+fPjp69KjOnDmT42vMq02bNqldu3aqWLGiSpYsqWbNmklStvovtnfvXp05c0atWrVyqSkuLk779u2TJO3atUsNGjRw2e/OO+90Wd6xY4caN27ssq5x48bas2ePMjMznevq16/v0sZut6tbt2769NNPJUm//vqrfv/9d+eQCQDFEw+UASiWduzYoYiIiBy31atXT/v379fixYv1ww8/6LHHHlN0dHS2saQXGzRokJYuXaq3335blStXlq+vrx555BGdPXvWpZ2Xl5fLss1mc87c4Ovre9ma09LSNHLkSD388MPZttnt9mzrypYtq6CgIO3cufOy/Z4+fVqtW7dW69at9fnnn6ts2bJKSEhQ69ats9V/aT2StGjRIlWoUMFlm4+Pz2WPeTX8/PyyrXvqqadUt25dHTp0SLGxsWrRooXCw8ML/NgAbhyEWwDFzvLly7V161a99NJLubYJCAhQp06d1KlTJz3yyCO67777dOLECZUuXVpeXl4udxQlae3aterZs6ceeughSf8Ev/j4+HzVVadOHf3f//2f8ziXqlevnnbt2qXKlSvnqT8PDw917txZn332mYYPH67Q0FCX7WlpabLb7dq5c6eOHz+u8ePHKywsTJL0yy+/uLS9MDPBxa+7Ro0a8vHxUUJCgvNO76WqVaumb7/91mXdxo0bXZarV6+utWvXuqxbu3atqlatqhIlSlz2NdauXVv169fXxx9/rJkzZ+rf//73ZdsDsD6GJQCwtIyMDCUmJurw4cP69ddfNXbsWLVv315t27ZV9+7dc9zn3Xff1RdffKGdO3dq9+7dmj17tkJCQpxTZjkcDi1btkyJiYk6efKkJKlKlSr6+uuvtWXLFv322296/PHHs82leyVdunRRSEiIOnTooLVr1+qPP/7Q3LlztW7dOknSsGHDFBcXp5EjR2rbtm3asWOHvvzySw0ZMiTXPseMGaOwsDA1bNhQcXFx2r59u/bs2aNPP/1Ut99+u9LS0lSxYkV5e3tr0qRJ+uOPP7RgwQKNGjXKpZ/w8HDZbDb997//1Z9//qm0tDSVLFlSgwYN0ksvvaQZM2Zo3759+vXXXzVp0iTNmDFDkvT0009r586devXVV7V792599dVXzlkqLgwLGThwoJYtW6ZRo0Zp9+7dmjFjhv79739r0KBBeTpvTz31lMaPHy9jjPOPCwDFmJvH/AJAoenRo4eRZCQZT09PU7ZsWRMdHW0+/fRTk5mZ6dJWFz0kNm3aNFO3bl3j5+dnAgICTMuWLc2vv/7qbLtgwQJTuXJl4+npacLDw40xxuzfv980b97c+Pr6mrCwMPPvf//bNGvWzLzwwgvO/cLDw817773nctzbbrvNDB8+3LkcHx9vOnbsaAICAsxNN91k6tevbzZs2ODcvmTJEtOoUSPj6+trAgICzJ133mmmTZt22fOQnJxsXnvtNVOlShXj7e1tgoODTXR0tJk3b57zobmZM2cah8NhfHx8TFRUlFmwYIGRZDZv3uzsJyYmxoSEhBibzWZ69OhhjPnnobj333/fVKtWzXh5eZmyZcua1q1bm1WrVjn3++abb0zlypWNj4+Pueeee8yUKVOMJPP3338728yZM8fUqFHDeHl5mYoVK5q33nrL5TXkdO4uOHXqlLnpppvMs88+e9nzAKB4sBmTjycOAAC4RmPGjNHUqVN18ODBAukvPj5elSpV0saNG1WvXr0C6RPAjYsxtwCAQjV58mQ1aNBAZcqU0dq1a/XWW29l+/KFq3Hu3DkdP35cQ4YM0V133UWwBSCJcAsAKGR79uzR6NGjdeLECVWsWFEDBw7U4MGDr7nftWvXqnnz5qpateplZ7IAULwwLAEAAACWwWwJAAAAsAzCLQAAACyDcAsAAADLINwCAADAMgi3AAAAsAzCLQAAACyDcAsAAADLINwCAADAMv4/UtdQSjQYIUMAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ce6c2d58"
      },
      "source": [
        "### 9. ลองนำไปใช้งาน (Prediction for a New Student)\n",
        "\n",
        "ส่วนนี้สาธิตการนำโมเดลที่ฝึกแล้วไปใช้ทำนายคะแนนของนักเรียนคนใหม่:\n",
        "\n",
        "*   **`preprocess_new_distance` function**: เป็นฟังก์ชันที่สร้างขึ้นมาเพื่อแปลงค่าระยะทาง (`distance_km`) ของนักเรียนคนใหม่ให้เป็น One-Hot Encoding ในรูปแบบเดียวกับข้อมูลที่ใช้ฝึกโมเดล เพื่อให้สอดคล้องกับคุณลักษณะ `X_train`\n",
        "*   **`new_student_hours = 6`, `new_student_distance_km = 15`**: กำหนดข้อมูลของนักเรียนคนใหม่ (อ่าน 6 ชม. บ้านไกล 15 กม.)\n",
        "*   **`new_distance_encoded = preprocess_new_distance(...)`**: แปลงระยะทางของนักเรียนคนใหม่เป็น One-Hot Encoding\n",
        "*   **`new_student_features = pd.DataFrame(...)`**: สร้าง DataFrame ของคุณลักษณะสำหรับนักเรียนคนใหม่ โดยรวม `Hours_Studied` กับ `new_distance_encoded`\n",
        "*   **`new_student_features = new_student_features.reindex(...)`**: ปรับลำดับคอลัมน์ของคุณลักษณะนักเรียนใหม่ให้ตรงกับ `X_train` ซึ่งเป็นสิ่งสำคัญเพื่อให้โมเดลสามารถทำนายได้อย่างถูกต้อง\n",
        "*   **`predicted_score = model.predict(new_student_features)`**: ใช้โมเดลที่ฝึกไว้ทำนายคะแนนสอบของนักเรียนคนใหม่\n",
        "*   **`print(...)`**: แสดงผลคะแนนที่ทำนายได้ออกมา"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 8. ลองนำไปใช้งาน (Prediction)\n",
        "# สมมติว่านักเรียน A: อ่านหนังสือ 6 ชม. และบ้านไกล 15 กม.\n",
        "# ฟังก์ชันช่วยแปลงระยะทางเป็น One-Hot Encoding สำหรับการทำนาย\n",
        "def preprocess_new_distance(distance_km, bins, labels, all_distance_categories):\n",
        "    category = pd.cut([distance_km], bins=bins, labels=labels, right=True, include_lowest=True)[0]\n",
        "    # Create a dummy dataframe with all possible categories, then fill the one for the new distance\n",
        "    dummy_df = pd.DataFrame(0, index=[0], columns=[f'Distance_{cat}' for cat in all_distance_categories])\n",
        "    dummy_df[f'Distance_{category}'] = 1\n",
        "    return dummy_df\n",
        "\n",
        "# Get all possible categories from the training data for consistent encoding\n",
        "all_distance_categories = sorted(distance_dummies.columns.str.replace('Distance_', '').tolist())\n",
        "\n",
        "new_student_hours = 6\n",
        "new_student_distance_km = 15\n",
        "\n",
        "# Preprocess the new student's distance\n",
        "new_distance_encoded = preprocess_new_distance(new_student_distance_km, bins, labels, all_distance_categories)\n",
        "\n",
        "# Prepare the feature array for the new student\n",
        "# Ensure the order of columns matches X_train\n",
        "new_student_features = pd.DataFrame([[new_student_hours]], columns=['Hours_Studied'])\n",
        "new_student_features = pd.concat([new_student_features, new_distance_encoded], axis=1)\n",
        "\n",
        "# Reindex to ensure column order consistency with X_train\n",
        "new_student_features = new_student_features.reindex(columns=X_train.columns, fill_value=0)\n",
        "\n",
        "predicted_score = model.predict(new_student_features)\n",
        "print(f\"\\nทำนายคะแนนของนักเรียน A: {predicted_score[0]:.2f} คะแนน\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "64atSeheBu_-",
        "outputId": "ff12da03-d30b-4dec-d0fb-79775a0168d8"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "ทำนายคะแนนของนักเรียน A: 73.39 คะแนน\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**ผู้จัดทำ**\n",
        "\n",
        "จิตราภรณ์ วรรณวิลัย\n",
        "\n",
        "ชลิดา ทิตศานติกุล\n",
        "\n",
        "จิรัชยา โชคกำเนิด\n",
        "\n",
        "นรีกานต์ นึกสม"
      ],
      "metadata": {
        "id": "qVLG8tXL7vFt"
      }
    }
  ]
}
