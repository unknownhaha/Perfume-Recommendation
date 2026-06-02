# ระบบแนะนำน้ำหอมแบบ Content-Based Hybrid
# Content-Based Hybrid Perfume Recommendation System

**ผู้จัดทำ / Authors:** ทีมพัฒนาโครงการ Perfume Recommendation

---

## บทคัดย่อ

โครงงานนี้นำเสนอระบบแนะนำน้ำหอมด้วยแนวคิดการกรองตามเนื้อหา (Content-Based Filtering) สำหรับชุดข้อมูลน้ำหอมประมาณ 26,000 รายการ โดยสร้างเวกเตอร์คุณลักษณะจากส่วนผสม (Ingredients) กลุ่มกลิ่น (Scent Family) กลุ่มกลิ่นย่อย (Subfamily) และเพศ (Gender) จากนั้นคำนวณคะแนนความเกี่ยวข้องแบบผสม (Hybrid Score) ระหว่าง **Cosine Similarity** และ **Jaccard Note Overlap** เพื่อจัดอันดับผลลัพธ์ ระบบรองรับการกรองตามเพศและแบรนด์ รวมถึงแสดงโน้ตที่ตรงกับสิ่งที่ผู้ใช้ระบุเพื่อเพิ่มความอธิบายได้ของคำแนะนำ (Explainability) ระบบไม่ต้องใช้ข้อมูลคะแนนจากผู้ใช้ (User Ratings) และให้ผลลัพธ์ได้รวดเร็วบนคลังข้อมูลขนาดใหญ่ โดยแสดงผลผ่านเว็บแอปพลิเคชันที่พัฒนาด้วย Streamlit

**คำสำคัญ:** ระบบแนะนำ, การกรองตามเนื้อหา, Cosine Similarity, Jaccard Similarity, น้ำหอม

## Abstract

This project presents a content-based perfume recommendation system for a dataset of approximately 26,000 perfumes. The system constructs sparse feature vectors from ingredients, scent family, subfamily, and gender, then ranks candidates using a hybrid relevance score that combines cosine similarity with Jaccard note overlap. It supports gender and brand filtering and exposes matched-note explanations to improve interpretability. The approach does not require user rating data and is delivered through a Streamlit web application.

**Keywords:** Recommendation System, Content-Based Filtering, Cosine Similarity, Jaccard Similarity, Perfume

---

## 1. บทนำ

### 1.1 ที่มาและความสำคัญ

ปัจจุบันตลาดน้ำหอมมีผลิตภัณฑ์หลากหลายนับหมื่นรายการ การค้นหาน้ำหอมที่ตรงกับความชอบส่วนตัวเป็นเรื่องยากสำหรับผู้บริโภคทั่วไป เนื่องจากต้องอาศัยความรู้เฉพาะทางด้านส่วนผสม กลุ่มกลิ่น และการผสมผสานระหว่างโน้ตต่างๆ ระบบแนะนำที่อาศัยความคล้ายคลึงของคุณลักษณะสามารถช่วยผู้ใช้ค้นพบน้ำหอมใหม่ที่สอดคล้องกับรสนิยมได้อย่างมีประสิทธิภาพ โดยไม่ต้องพึ่งพาข้อมูลประวัติการให้คะแนนจากผู้ใช้ (Collaborative Filtering) ซึ่งมักต้องการข้อมูลจำนวนมากและเผชิญกับปัญหา Cold Start

โครงงานนี้จึงได้พัฒนาระบบแนะนำน้ำหอมแบบ Content-Based Filtering โดยใช้ชุดข้อมูล doevent/perfume จาก HuggingFace ซึ่งประกอบด้วยข้อมูลน้ำหอมกว่า 26,000 รายการ พร้อมรายละเอียดส่วนผสม กลุ่มกลิ่น แบรนด์ และเพศ

### 1.2 วัตถุประสงค์

1. เพื่อพัฒนาระบบแนะนำน้ำหอมที่ไม่ต้องการข้อมูลคะแนนจากผู้ใช้
2. เพื่อเปรียบเทียบแนวทางการสร้างเวกเตอร์คุณลักษณะหลายแบบ (Approach A-D) และเลือกแนวทางที่เหมาะสมที่สุด
3. เพื่อพัฒนากลไกจัดอันดับแบบ Hybrid (Cosine + Jaccard) ที่ให้ผลแนะนำแม่นยำและอธิบายได้
4. เพื่อพัฒนาเว็บแอปพลิเคชันที่ผู้ใช้สามารถค้นหาและกรองคำแนะนำได้สะดวก

### 1.3 ทฤษฎีและเทคโนโลยีที่เกี่ยวข้อง

#### 1.3.1 Content-Based Filtering

Content-Based Filtering เป็นแนวทางของระบบแนะนำที่วิเคราะห์คุณลักษณะของสินค้าเอง โดยไม่จำเป็นต้องมีข้อมูลพฤติกรรมของผู้ใช้คนอื่น ระบบสร้างโปรไฟล์ของสินค้าจากคุณลักษณะต่างๆ และเปรียบเทียบความคล้ายคลึงระหว่างโปรไฟล์ที่ผู้ใช้ระบุกับสินค้าในคลัง วิธีนี้เหมาะกับปัญหาที่ไม่มีข้อมูล Rating และแก้ปัญหา Cold Start ได้ดี

#### 1.3.2 Cosine Similarity ใน Sparse Vector Space

Cosine Similarity ใช้วัดความใกล้เคียงของเวกเตอร์ระหว่างความต้องการผู้ใช้กับรายการน้ำหอมทั้งหมดในคลังข้อมูล โดยวิธีนี้เหมาะกับข้อมูลแบบ sparse และมีประสิทธิภาพดีเมื่อจำนวนมิติของเวกเตอร์สูง ระบบจะใช้ค่านี้เป็นคะแนนหลักก่อนเข้าสู่ขั้นตอน re-ranking

#### 1.3.3 Jaccard Similarity

Jaccard Similarity ใช้วัดอัตราส่วนระหว่างจำนวนโน้ตที่ซ้ำกันกับจำนวนโน้ตรวมกันของสองเซต โดยในระบบนี้ใช้วัดการ overlap ของ `liked_ingredients` กับ `ingredients` ของน้ำหอมแต่ละรายการ เพื่อนำมาเป็นคะแนนเสริมใน Hybrid Ranking ช่วยให้ผลลัพธ์สะท้อนการจับคู่โน้ตจริงได้ชัดเจนขึ้น

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
4. ไม่ครอบคลุมระบบรีวิว/เรตติ้งจากผู้ใช้จริง และยังไม่รองรับ online learning

**กลุ่มเป้าหมาย:** ผู้ที่สนใจค้นหาน้ำหอมและต้องการคำแนะนำตามรสนิยมส่วนตัว

### 1.5 ประโยชน์ที่คาดว่าจะได้รับ

1. ผู้ใช้งานสามารถค้นหาน้ำหอมที่เหมาะกับรสนิยมได้อย่างรวดเร็ว
2. ระบบสามารถใช้งานได้ทันทีโดยไม่ต้องมีประวัติการใช้งาน (No Cold Start)
3. ได้ระบบที่อธิบายเหตุผลคำแนะนำได้ผ่าน matched notes
4. ได้ต้นแบบที่สามารถขยายไปยังโดเมนสินค้าอื่นได้

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
4. **TF-IDF Encoding:** ใช้ `TfidfVectorizer` (max_features=500) สร้าง Vector จาก `description` (ใน Approach C)
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
3. บันทึก Best Approach และ Encoders เป็น Artifact ใน `models/`
4. สร้างกลไกจัดอันดับแบบ Hybrid โดยใช้ `cosine_similarity` + Jaccard overlap
5. ทดสอบผล Recommendation จาก Query ตัวอย่าง

**Core inference flow:**
```
Query (ingredients + family + subfamily + gender + optional brand)
    → _build_query_vector() → Sparse Query Vector
    → Gender/Brand Pre-filter (กรอง DataFrame และ Matrix)
    → cosine_similarity(query, filtered_matrix)
    → Shortlist candidates
    → Re-rank with Jaccard note overlap (Hybrid score)
    → Return DataFrame พร้อม similarity/hybrid/matched notes
```

### 2.5 การพัฒนาเว็บแอปพลิเคชัน

สร้างเว็บแอปพลิเคชันด้วย Streamlit (`src/app.py`) มีฟีเจอร์หลัก ได้แก่:

- **ตัวกรองฝั่ง Sidebar:** เลือก Gender, Scent Family, Subfamily และ Brand
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
- `model_comparison.csv` — ตารางสรุปผลเปรียบเทียบแนวทาง

#### 3.1.2 ผลการเปรียบเทียบ Approach

ผลการเปรียบเทียบแนวทางการสร้าง Feature Vector บันทึกใน `models/model_comparison.csv` โดย **Approach D** (ingredients × 2 + categories) ให้ผลดีที่สุด เนื่องจากการให้น้ำหนักส่วนผสมเป็นสองเท่าช่วยเพิ่มความสำคัญของโน้ตกลิ่นซึ่งเป็นปัจจัยหลักในการเลือกน้ำหอม และเมื่อนำมาผสานกับ Jaccard overlap ทำให้ลำดับผลลัพธ์สะท้อนความต้องการผู้ใช้ได้ดีขึ้น

#### 3.1.3 เว็บแอปพลิเคชัน

เว็บแอปพลิเคชัน Streamlit สามารถทำงานได้ครบทุกฟีเจอร์:
1. ระบบกรองตาม Gender, Family, Subfamily และ Brand
2. ระบบ Multiselect ส่วนผสมจาก Top-200 ingredients
3. การแสดงผลน้ำหอมแนะนำพร้อมคะแนนความคล้ายคลึงและ Hybrid Score
4. การแสดงรูปภาพผลิตภัณฑ์ (เมื่อมีการดาวน์โหลด image dataset)
5. แสดง `matched_notes` เพื่ออธิบายเหตุผลของคำแนะนำ

### 3.2 การประเมินประสิทธิภาพ

#### 3.2.1 ความเร็วในการตอบสนอง

ระบบใช้การคำนวณ cosine similarity บน In-memory Sparse Matrix ขนาด ~26,000 รายการร่วมกับการ re-rank ระยะสั้นด้วย Jaccard overlap ทำให้ตอบสนองได้รวดเร็วและเหมาะกับการใช้งานเชิงโต้ตอบบนเว็บแอป

#### 3.2.2 ผลการทดสอบเชิงคุณภาพ

ผลการแนะนำจาก Query ตัวอย่าง เช่น `liked_ingredients=["Rose", "Jasmine"], family="FLORAL", gender="FEMALE"` ให้ผลน้ำหอมที่มีส่วนผสมและกลุ่มกลิ่นใกล้เคียงกัน โดยมีค่า Cosine Similarity สูง ซึ่งสอดคล้องกับความคาดหวัง

### 3.3 ปัญหาที่พบ

1. **Out-of-Vocabulary Notes:** เมื่อผู้ใช้ระบุส่วนผสมที่ไม่มีใน Training Data เช่น คำสะกดผิด ระบบจะสร้าง Query Vector ที่มีค่าศูนย์สำหรับโน้ตนั้น ทำให้ผลลัพธ์อาจไม่ตรงตามความต้องการ
2. **ความไม่สม่ำเสมอของ Metadata:** บางรายการมีข้อมูล `family/subfamily/gender` ไม่ครบถ้วน ทำให้คุณภาพคำแนะนำผันผวนในบางเงื่อนไขกรอง
3. **ขนาดของ Image Dataset:** รูปภาพผลิตภัณฑ์มีขนาด ~835 MB ทำให้ไม่เหมาะกับการรวมใน Repository หลัก ต้องดาวน์โหลดแยกต่างหาก
4. **Cold Start สำหรับน้ำหอมใหม่:** น้ำหอมที่ไม่อยู่ในชุดข้อมูลไม่สามารถแนะนำได้โดยตรง

---

## 4. สรุปผลการดำเนินโครงงาน

ระบบแนะนำน้ำหอมแบบ Content-Based ที่พัฒนาขึ้นสามารถแนะนำรายการที่สอดคล้องกับความชอบผู้ใช้จากส่วนผสม กลุ่มกลิ่น กลุ่มกลิ่นย่อย และเพศได้อย่างมีประสิทธิภาพ โดยไม่ต้องอาศัยข้อมูล User Rating แนวทางที่เลือกใช้คือ Approach D ร่วมกับการจัดอันดับแบบ Hybrid (Cosine + Jaccard) ซึ่งให้ผลลัพธ์มีความแม่นยำและอธิบายได้ดีขึ้นผ่าน matched notes ระบบถูกบรรจุในเว็บแอป Streamlit ที่ใช้งานง่ายและพร้อมต่อยอดในเชิงผลิตภัณฑ์

**ข้อเสนอแนะสำหรับการพัฒนาในอนาคต:**
- นำ Sentence-Transformers มาใช้ encode คำบรรยายเพื่อเพิ่มความแม่นยำเชิงความหมาย
- เพิ่มระบบ feedback loop เพื่อเก็บ implicit feedback จากการคลิก/บันทึกรายการ
- พัฒนา personalization รายบุคคลจากประวัติการใช้งานจริง
- เพิ่มระบบ benchmark อัตโนมัติและ regression test สำหรับคุณภาพคำแนะนำ

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

- **GitHub Repository:** (ใส่ URL ของ repository ปัจจุบันหลังจาก deploy/เผยแพร่)
- **Dataset:** https://huggingface.co/datasets/doevent/perfume
