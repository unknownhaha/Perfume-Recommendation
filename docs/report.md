# ระบบแนะนำน้ำหอมด้วยการกรองตามเนื้อหา
# Content-Based Perfume Recommendation System

**ผู้จัดทำ / Authors**

---

## บทคัดย่อ

โครงงานนี้นำเสนอระบบแนะนำน้ำหอมโดยอาศัยการกรองตามเนื้อหา (Content-Based Filtering) สำหรับชุดข้อมูลน้ำหอมขนาดประมาณ 26,000 รายการ โดยใช้อัลกอริทึม K-Nearest Neighbors (KNN) ร่วมกับ Cosine Similarity บนเวกเตอร์คุณลักษณะแบบ Sparse ที่สร้างจากส่วนผสม (Ingredients) กลุ่มกลิ่น (Scent Family) และเพศ (Gender) ผู้ใช้งานสามารถระบุส่วนผสมที่ชอบ กลุ่มกลิ่น และเพศที่ต้องการ ระบบจะคืนผลรายชื่อน้ำหอมที่มีความคล้ายคลึงสูงสุดพร้อมคะแนนความคล้ายคลึง ระบบนี้ไม่ต้องการข้อมูลคะแนนจากผู้ใช้ (User Ratings) และทำงานได้อย่างรวดเร็วในระยะเวลาน้อยกว่า 5 มิลลิวินาทีต่อการค้นหาหนึ่งครั้ง นำเสนอผ่านเว็บแอปพลิเคชันที่สร้างด้วย Streamlit

**คำสำคัญ:** ระบบแนะนำ, การกรองตามเนื้อหา, KNN, Cosine Similarity, น้ำหอม

## Abstract

This project presents a content-based perfume recommendation system for a dataset of approximately 26,000 perfumes. The system employs K-Nearest Neighbors (KNN) combined with cosine similarity on sparse feature vectors constructed from ingredients, scent family, and gender. Users specify preferred ingredients, scent family, and gender; the system returns the most similar perfumes ranked by similarity score. No user rating data is required. Queries execute in under 5 ms and results are served through a Streamlit web application.

**Keywords:** Recommendation System, Content-Based Filtering, KNN, Cosine Similarity, Perfume

---

## 1. บทนำ

### 1.1 ที่มาและความสำคัญ

ปัจจุบันตลาดน้ำหอมมีผลิตภัณฑ์หลากหลายนับหมื่นรายการ การค้นหาน้ำหอมที่ตรงกับความชอบส่วนตัวเป็นเรื่องยากสำหรับผู้บริโภคทั่วไป เนื่องจากต้องอาศัยความรู้เฉพาะทางด้านส่วนผสม กลุ่มกลิ่น และการผสมผสานระหว่างโน้ตต่างๆ ระบบแนะนำที่อาศัยความคล้ายคลึงของคุณลักษณะสามารถช่วยผู้ใช้ค้นพบน้ำหอมใหม่ที่สอดคล้องกับรสนิยมได้อย่างมีประสิทธิภาพ โดยไม่ต้องพึ่งพาข้อมูลประวัติการให้คะแนนจากผู้ใช้ (Collaborative Filtering) ซึ่งมักต้องการข้อมูลจำนวนมากและเผชิญกับปัญหา Cold Start

โครงงานนี้จึงได้พัฒนาระบบแนะนำน้ำหอมแบบ Content-Based Filtering โดยใช้ชุดข้อมูล doevent/perfume จาก HuggingFace ซึ่งประกอบด้วยข้อมูลน้ำหอมกว่า 26,000 รายการ พร้อมรายละเอียดส่วนผสม กลุ่มกลิ่น แบรนด์ และเพศ

### 1.2 วัตถุประสงค์

1. เพื่อพัฒนาระบบแนะนำน้ำหอมที่ไม่ต้องการข้อมูลคะแนนจากผู้ใช้
2. เพื่อเปรียบเทียบแนวทางการสร้างเวกเตอร์คุณลักษณะหลายแบบและเลือกแนวทางที่ให้ผลดีที่สุด
3. เพื่อพัฒนาเว็บแอปพลิเคชันที่ผู้ใช้สามารถค้นหาน้ำหอมได้อย่างสะดวก
4. เพื่อให้ระบบตอบสนองได้รวดเร็ว (< 5 ms ต่อการค้นหา)

### 1.3 ทฤษฎีและเทคโนโลยีที่เกี่ยวข้อง

#### 1.3.1 Content-Based Filtering

Content-Based Filtering เป็นแนวทางของระบบแนะนำที่วิเคราะห์คุณลักษณะของสินค้าเอง โดยไม่จำเป็นต้องมีข้อมูลพฤติกรรมของผู้ใช้คนอื่น ระบบสร้างโปรไฟล์ของสินค้าจากคุณลักษณะต่างๆ และเปรียบเทียบความคล้ายคลึงระหว่างโปรไฟล์ที่ผู้ใช้ระบุกับสินค้าในคลัง วิธีนี้เหมาะกับปัญหาที่ไม่มีข้อมูล Rating และแก้ปัญหา Cold Start ได้ดี

#### 1.3.2 K-Nearest Neighbors (KNN)

K-Nearest Neighbors เป็นอัลกอริทึมที่หาจุดข้อมูลที่ใกล้ที่สุด k จุดในพื้นที่ Feature Space โดยในโครงงานนี้ใช้ `NearestNeighbors` จาก scikit-learn พร้อม metric=`cosine` และ algorithm=`brute` ซึ่งเหมาะกับ Sparse Vector ขนาดใหญ่

#### 1.3.3 Cosine Similarity

Cosine Similarity วัดความคล้ายคลึงระหว่างเวกเตอร์สองตัวโดยคำนวณค่า cosine ของมุมระหว่างเวกเตอร์ ให้ค่าในช่วง [0, 1] โดยค่า 1 หมายถึงเหมือนกันทุกประการ และ 0 หมายถึงไม่มีความคล้ายคลึงกัน วิธีนี้เหมาะกับเวกเตอร์ที่ sparse เนื่องจากไม่ขึ้นกับขนาด (magnitude) ของเวกเตอร์

#### 1.3.4 MultiLabelBinarizer

`MultiLabelBinarizer` จาก scikit-learn แปลงรายการส่วนผสม (list of strings) ให้เป็น Binary Vector โดยแต่ละมิติแทนส่วนผสมหนึ่งชนิด ใช้สำหรับ Encoding ส่วนผสมของน้ำหอม

#### 1.3.5 OneHotEncoder

`OneHotEncoder` จาก scikit-learn แปลงตัวแปรแบบ Categorical เช่น กลุ่มกลิ่น (family) กลุ่มกลิ่นย่อย (subfamily) และเพศ (gender) ให้เป็น Binary Vector

#### 1.3.6 Streamlit

Streamlit เป็น Python Framework ที่ช่วยให้นักพัฒนาสามารถสร้างเว็บแอปพลิเคชัน Data Science และ Machine Learning ได้อย่างรวดเร็ว โดยเขียนด้วย Python ล้วนๆ ไม่ต้องมีความรู้ด้าน Frontend เหมาะสำหรับการสาธิตและต้นแบบระบบ

### 1.4 ขอบเขตของโครงงาน

1. ชุดข้อมูล: doevent/perfume จาก HuggingFace (~26,000 รายการ)
2. ระบบแนะนำแบบ Content-Based เท่านั้น ไม่รวม Collaborative Filtering
3. เว็บแอปพลิเคชันที่ใช้งานผ่าน Streamlit บนเครื่อง Local

**กลุ่มเป้าหมาย:** ผู้ที่สนใจค้นหาน้ำหอมและต้องการคำแนะนำตามรสนิยมส่วนตัว

### 1.5 ประโยชน์ที่คาดว่าจะได้รับ

1. ผู้ใช้งานสามารถค้นหาน้ำหอมที่เหมาะกับรสนิยมได้อย่างรวดเร็ว
2. ระบบสามารถใช้งานได้ทันทีโดยไม่ต้องมีประวัติการใช้งาน (No Cold Start)
3. ได้ต้นแบบระบบแนะนำที่สามารถขยายไปยังโดเมนสินค้าอื่นได้

---

## 2. วิธีการดำเนินโครงงาน

### 2.1 การรวบรวมข้อมูล (Data Collection)

ชุดข้อมูลที่ใช้คือ **doevent/perfume** จาก HuggingFace Hub ซึ่งสามารถดาวน์โหลดอัตโนมัติผ่าน `huggingface_hub` ข้อมูลถูกจัดเก็บในรูปแบบ JSON (`data/perfumes.json`) ประกอบด้วยคอลัมน์หลัก ได้แก่ `brand`, `name_perfume`, `family`, `subfamily`, `fragrances`, `ingredients` (list), `origin`, `gender`, `years`, `description`, `image_name`

นอกจากนี้ยังมีรูปภาพผลิตภัณฑ์ขนาดประมาณ 835 MB ที่สามารถดาวน์โหลดเสริมได้ผ่านปุ่มใน Sidebar ของแอปพลิเคชัน

### 2.2 การสำรวจข้อมูลเบื้องต้น (Exploratory Data Analysis)

ดำเนินการใน `notebooks/01_eda.ipynb` วิเคราะห์การกระจายตัวของข้อมูล เช่น จำนวนน้ำหอมต่อแบรนด์, การกระจายของกลุ่มกลิ่น, ส่วนผสมที่พบบ่อย และการกระจายตามเพศ เพื่อทำความเข้าใจโครงสร้างของชุดข้อมูลก่อนเข้าสู่กระบวนการ Preprocessing

### 2.3 การเตรียมข้อมูลและสร้าง Feature Vector (Preprocessing)

ดำเนินการใน `notebooks/02_preprocessing.ipynb` ประกอบด้วย:

1. **ทำความสะอาดข้อมูล:** แปลงตัวแปร Categorical ให้เป็น Uppercase และส่วนผสมให้เป็น Title-case ด้วย `clean_ingredients()`
2. **Encoding ส่วนผสม:** ใช้ `MultiLabelBinarizer` สร้าง Binary Vector จากรายการส่วนผสม
3. **Encoding หมวดหมู่:** ใช้ `OneHotEncoder` สร้าง Vector จาก `family`, `subfamily`, `gender`
4. **TF-IDF Encoding:** ใช้ `TfidfVectorizer` (max_features=500) สร้าง Vector จาก `description`
5. **สร้าง Feature Matrix** 4 แบบ (Approach A–D) บันทึกเป็น Sparse Matrix (.npz)

| Approach | Feature Vector |
|----------|----------------|
| A | MultiLabelBinarizer(ingredients) เท่านั้น |
| B | ingredients + OneHotEncoder(family, subfamily, gender) |
| C | B + TfidfVectorizer(description) |
| D | ingredients × 2 + categories (ให้น้ำหนักส่วนผสมมากขึ้น) |

### 2.4 การพัฒนา Recommendation Engine

ดำเนินการใน `notebooks/03_recommendation_engine.ipynb`:

1. เปรียบเทียบ Approach A–D ด้วย Evaluation Metric ที่เหมาะสม
2. เลือก Approach ที่ดีที่สุด (ผล: **Approach D**)
3. บันทึก KNN Model, Best Approach และ Encoders เป็น Artifact ใน `models/`
4. ทดสอบผล Recommendation จาก Query ตัวอย่าง

**Core inference flow:**
```
Query (ingredients + family + gender)
    → _build_query_vector() → Sparse Query Vector
    → Gender Pre-filter (กรอง DataFrame และ Matrix)
    → NearestNeighbors.fit(filtered_matrix)
    → kneighbors(query) → Top-N indices
    → Return DataFrame พร้อม similarity score
```

### 2.5 การพัฒนาเว็บแอปพลิเคชัน

สร้างเว็บแอปพลิเคชันด้วย Streamlit (`src/app.py`) มีฟีเจอร์หลัก ได้แก่:

- **ตัวกรองฝั่ง Sidebar:** เลือก Gender, Scent Family, Subfamily
- **Multiselect ส่วนผสม:** เลือกส่วนผสมที่ชอบจาก Top-200 ที่พบบ่อย
- **แสดงผล:** ชื่อน้ำหอม แบรนด์ ส่วนผสม กลุ่มกลิ่น คะแนนความคล้ายคลึง และรูปภาพ (ถ้ามี)
- **ดาวน์โหลดรูปภาพ:** ปุ่มใน Sidebar สำหรับดาวน์โหลด Product Image

---

## 3. ผลการดำเนินงาน

### 3.1 ผลการพัฒนาระบบ

#### 3.1.1 ชุดข้อมูลและ Artifact

หลังผ่านกระบวนการ Preprocessing ได้ผลลัพธ์ดังนี้:

- `perfume_df.pkl` — DataFrame ที่ผ่านการทำความสะอาดแล้ว (~26,000 แถว)
- `mlb_ingredients.pkl` — MultiLabelBinarizer สำหรับ Encode ส่วนผสม
- `ohe_categories.pkl` — OneHotEncoder สำหรับ family, subfamily, gender
- `tfidf_description.pkl` — TfidfVectorizer สำหรับ description
- `matrix_A.npz` – `matrix_D.npz` — Sparse Feature Matrix 4 แบบ
- `best_approach.pkl` — ระบุ Approach ที่ดีที่สุด (ค่า: "D")
- `knn_model.pkl` — KNN Model ที่ Fit บน Full Matrix

#### 3.1.2 ผลการเปรียบเทียบ Approach

ผลการเปรียบเทียบแนวทางการสร้าง Feature Vector บันทึกใน `models/model_comparison.csv` โดย **Approach D** (ingredients × 2 + categories) ให้ผลดีที่สุด เนื่องจากการให้น้ำหนักส่วนผสมเป็นสองเท่าช่วยเพิ่มความสำคัญของโน้ตกลิ่นซึ่งเป็นปัจจัยหลักในการเลือกน้ำหอม

#### 3.1.3 เว็บแอปพลิเคชัน

เว็บแอปพลิเคชัน Streamlit สามารถทำงานได้ครบทุกฟีเจอร์:
1. ระบบกรองตาม Gender, Family, Subfamily
2. ระบบ Multiselect ส่วนผสมจาก Top-200 ingredients
3. การแสดงผลน้ำหอมแนะนำพร้อมคะแนนความคล้ายคลึง
4. การแสดงรูปภาพผลิตภัณฑ์ (เมื่อมีการดาวน์โหลด image dataset)
5. ระบบ Admin จัดการการดาวน์โหลด Image Dataset

### 3.2 การประเมินประสิทธิภาพ

#### 3.2.1 ความเร็วในการตอบสนอง

ระบบ KNN แบบ Brute-force บน In-memory Sparse Matrix ขนาด ~26,000 รายการ ให้เวลาตอบสนองน้อยกว่า **5 มิลลิวินาที** ต่อการค้นหาหนึ่งครั้ง ซึ่งเหมาะสมสำหรับการใช้งานจริง

#### 3.2.2 ผลการทดสอบเชิงคุณภาพ

ผลการแนะนำจาก Query ตัวอย่าง เช่น `liked_ingredients=["Rose", "Jasmine"], family="FLORAL", gender="FEMALE"` ให้ผลน้ำหอมที่มีส่วนผสมและกลุ่มกลิ่นใกล้เคียงกัน โดยมีค่า Cosine Similarity สูง ซึ่งสอดคล้องกับความคาดหวัง

### 3.3 ปัญหาที่พบ

1. **Out-of-Vocabulary Notes:** เมื่อผู้ใช้ระบุส่วนผสมที่ไม่มีใน Training Data เช่น คำสะกดผิด ระบบจะสร้าง Query Vector ที่มีค่าศูนย์สำหรับโน้ตนั้น ทำให้ผลลัพธ์อาจไม่ตรงตามความต้องการ
2. **ขนาดของ Image Dataset:** รูปภาพผลิตภัณฑ์มีขนาด ~835 MB ทำให้ไม่เหมาะกับการรวมใน Repository หลัก ต้องดาวน์โหลดแยกต่างหาก
3. **Cold Start สำหรับน้ำหอมใหม่:** น้ำหอมที่ไม่อยู่ในชุดข้อมูลไม่สามารถแนะนำได้โดยตรง

---

## 4. สรุปผลการดำเนินโครงงาน

ระบบแนะนำน้ำหอมแบบ Content-Based Filtering ที่พัฒนาขึ้นสามารถแนะนำน้ำหอมที่มีความคล้ายคลึงตามส่วนผสม กลุ่มกลิ่น และเพศได้อย่างมีประสิทธิภาพ โดยไม่ต้องการข้อมูล User Rating และตอบสนองได้รวดเร็วในระดับมิลลิวินาที การเลือกใช้ Approach D ซึ่งให้น้ำหนักส่วนผสมเป็นสองเท่าพิสูจน์ว่าให้ผลการแนะนำดีที่สุดในบรรดาแนวทางที่ทดสอบ ระบบนำเสนอผ่านเว็บแอปพลิเคชัน Streamlit ที่ใช้งานง่ายและรองรับการขยายฟีเจอร์เพิ่มเติมในอนาคต

**ข้อเสนอแนะสำหรับการพัฒนาในอนาคต:**
- นำ Sentence-Transformers มาใช้ Encode Description เพื่อเพิ่มความแม่นยำของ Semantic Similarity
- พัฒนา Approach E ที่รวม Free-text Search จากผู้ใช้
- เพิ่มระบบ Feedback เพื่อเก็บข้อมูล Implicit Rating สำหรับพัฒนา Hybrid Recommender ในอนาคต
- พัฒนา Mobile Application เพื่อให้เข้าถึงได้ทุกที่

---

## 5. เอกสารอ้างอิง

[1] HuggingFace, "doevent/perfume dataset," [ระบบออนไลน์], สืบค้นจาก: https://huggingface.co/datasets/doevent/perfume

[2] scikit-learn developers, "sklearn.neighbors.NearestNeighbors," [ระบบออนไลน์], สืบค้นจาก: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html

[3] scikit-learn developers, "sklearn.preprocessing.MultiLabelBinarizer," [ระบบออนไลน์], สืบค้นจาก: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html

[4] scikit-learn developers, "sklearn.preprocessing.OneHotEncoder," [ระบบออนไลน์], สืบค้นจาก: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

[5] scikit-learn developers, "sklearn.feature_extraction.text.TfidfVectorizer," [ระบบออนไลน์], สืบค้นจาก: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

[6] Streamlit Inc., "Streamlit — A faster way to build and share data apps," [ระบบออนไลน์], สืบค้นจาก: https://streamlit.io/

[7] scipy.sparse, "Sparse matrix operations," [ระบบออนไลน์], สืบค้นจาก: https://docs.scipy.org/doc/scipy/reference/sparse.html

[8] Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based Recommender Systems: State of the Art and Trends. In *Recommender Systems Handbook* (pp. 73–105). Springer.

[9] Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513–523.

---

## 6. ลิงก์ที่เกี่ยวข้อง

- **GitHub Repository:** https://github.com/[your-username]/Perfume-Recommendation
- **Dataset:** https://huggingface.co/datasets/doevent/perfume
