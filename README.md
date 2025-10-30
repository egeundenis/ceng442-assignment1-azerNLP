# CENG442 Assignment 1: Azerbaijani Text Preprocessing + Word Embeddings

**Group Members:**
- Ege Ündeniş
- Emine Sena Top
- Rümeysa Yavuzkanat

## 1. Data & Goal

We processed five Azerbaijani sentiment datasets (124,128 total rows after cleaning). We kept neutral as 0.5 to preserve fine-grained sentiment distinctions from 3-class datasets, which helps train more nuanced models.

| Dataset | Rows | Labels |
|---------|------|--------|
| labeled-sentiment | 2,955 | tri-class → 0.0/0.5/1.0 |
| test__1_ | 4,198 | binary → 0.0/1.0 |
| train__3_ | 19,557 | binary → 0.0/1.0 |
| train-00000-of-00001 | 41,756 | tri-class → 0.0/0.5/1.0 |
| merged_dataset_CSV__1_ | 55,662 | binary → 0.0/1.0 |

## 2. Preprocessing

We applied Azerbaijani-aware lowercasing (İ→i, I→ı), replaced URLs/emails/phones/mentions with tokens (URL, EMAIL, PHONE, USER), stripped HTML, collapsed repeated characters (≥3→2), converted digits to **\<NUM\>**, split camelCase hashtags, mapped emojis to EMO_POS/EMO_NEG, applied deasciify (cox→çox, yaxsi→yaxşı), removed punctuation while preserving Azerbaijani letters (ə,ğ,ı,ö,ü,ç,ş,x,q), and dropped duplicates/empties.

**Examples:**

**Before:** "Bu telefon ÇOX yaxsi!! 😍 #TechReview"  
**After:** "bu telefon çox yaxşı EMO_POS tech review"

**Before:** "@user heç yaxşı deyil 😡"  
**After:** "USER heç yaxşı_NEG deyil_NEG EMO_NEG"

**Stats:** Removed ~3-5% duplicates and ~1-2% empty rows per dataset.

## 3. Mini Challenges

**Hashtag Splitting:** Used `re.sub('([a-z])([A-Z])', r'\1 \2')` to split camelCase. Successfully split ~8% of social samples (e.g., #QarabagIsBack → "qarabag is back").

**Emoji Mapping:** Extended dictionary to 12 emojis including 🔥→EMO_POS and 😭→EMO_NEG. Emojis appeared in ~15% of social texts.

**Stopwords:** We compared Turkish, English, and Russian lists, then proposed: **və, ilə, bu, bir, də, da, ki, mən, sən, o, amma, ancaq, həm, orada, bütün, hər, çox, az, üçün**. Never removed negators (yox, deyil, heç, qətiyyən, yoxdur). Removing these reduced vocabulary by ~12%.

**Negation Scope:** Marked next 3 tokens with **_NEG** after negators. Example: "heç yaxşı deyil" → "heç yaxşı_NEG deyil_NEG". Added double negation detection for "yox deyil" patterns. Negated words showed different nearest neighbors (more negative context).

**Deasciify:** Small map: **{slm→salam, tmm→tamam, sagol→sağol, cox→çox, yaxsi→yaxşı}**. Changed ~4% of tokens, mostly in social domain.

## 4. Domain-Aware

**Detection:** 4 classes (news/social/reviews/general) using regex:

- **NEWS_HINTS:** "apa|trend|azertac|reuters|bloomberg|dha|aa|bbc|cnn"
- **SOCIAL_HINTS:** "@|#|emojis"
- **REV_HINTS:** "azn|manat|qiymət|aldım|ulduz|çox yaxşı|çox pis"

**Distribution:** News 8.3%, Social 31.2%, Reviews 24.5%, General 36.0%.

**Reviews Normalization:** 
- `\d+ azn` → **\<PRICE\>**
- `4 ulduz` → **\<STARS_4\>**
- `çox yaxşı` → **\<RATING_POS\>**

**Corpus Tags:** Each line in **corpus_all.txt** prefixed with domain token: `domnews bakının mərkəzində...`

## 5. Embeddings

**Training:** Word2Vec and FastText on 124,353 sentences.

| Parameter | Value |
|-----------|-------|
| vector_size | 300 |
| window | 5 |
| min_count | 3 |
| sg | 1 |
| epochs | 10 |
| FastText min_n/max_n | 3/6 |

**Coverage:**

| Dataset | Word2Vec | FastText |
|---------|----------|----------|
| labeled-sentiment | 93.0% | 93.0% |
| test__1_ | 98.6% | 98.6% |
| train__3_ | 98.9% | 98.9% |
| train-00000 | 94.3% | 94.3% |
| merged_dataset | 94.9% | 94.9% |

**Similarity (Synonyms vs Antonyms):**

| Type | Word2Vec | FastText |
|------|----------|----------|
| Synonyms (yaxşı-əla, bahalı-qiymətli, ucuz-sərfəli) | 0.341 | 0.433 |
| Antonyms (yaxşı-pis, bahalı-ucuz) | 0.358 | 0.419 |
| **Separation** | **0.017** | **0.014** |

**Nearest Neighbors:**

- **"yaxşı"** — W2V: **iyi, \<RATING_POS\>, yaxshi, awsome** | FT: **yaxşıı, yaxşıkı, yaxşıca**
- **"pis"** — W2V: **günd, yaxşıdır_NEG** | FT: **piis, pisdii, pisik**
- **"\<RATING_POS\>"** — W2V: **süper, uygulama, deneyin** | FT: **\<RATING_NEG\>, süperr, süper**

**Analysis:** W2V has cleaner semantic neighbors. FT captures morphology but is noisier. Domain tokens learned sentiment associations.

## 6. Reproducibility

**Environment:** Python 3.10, pandas 2.0.3, gensim 4.3.2, openpyxl 3.1.2, ftfy 6.1.1, Google Colab (12GB RAM, Intel Xeon 2.20GHz).

**Run:**
```bash
git clone https://github.com/<org>/ceng442-assignment1-<group>
pip install -r requirements.txt
# Place datasets in sample_data/
python homework.ipynb  # or run in Jupyter/Colab
```

**Outputs:** 5 `*_2col.xlsx`, `corpus_all.txt` (124,353 lines), `embeddings/*.model`

## 7. Conclusions

**Which model worked better?** Word2Vec produced cleaner semantic neighbors and better sentiment context distinction. FastText had superior OOV handling via subwords and captured morphological richness but with noisier neighbors.

**Why?** Low syn/ant separation (0.02) due to limited data (~124k sentences) and high domain diversity. FastText's character n-grams capture morphology (yaxşı→yaxşıca) but dilute semantic signal.

**Next steps:**
1. Expand corpus to ~10M sentences
2. Test sentiment-specific training objectives
3. Parameter tuning — experiment with vector_size=200, min_count=5, window=3 and update results
4. Try lemmatization if tools become available

**Repository:** [https://github.com/egeundenis/ceng442-assignment1-azerNLP](https://github.com/egeundenis/ceng442-assignment1-azerNLP)
