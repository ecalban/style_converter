# Kuşaklararası Köprü

Modern Türkçe bir metni eski edebi Türkçe veya Anadolu ozanı üslubuna dönüştüren RAG + Gemini tabanlı Yapay Zeka / NLP projesi.

Projeyi kod üzerinden anlamak ve öğretmen sorularına hazırlanmak için önce `PROJE_REHBERI.md` dosyasını okuyun.

## Nasıl Çalıştırılır?

VS Code ile `kusaklararasi-kopru` klasörünü açın.

Proje klasöründe `.env` adlı bir dosya oluşturup Gemini API bilgilerini yazın:

```text
GEMINI_API_KEY=google_ai_studio_keyiniz
GEMINI_MODEL=gemini-2.5-flash
```

429 kota/hız sınırı hatası alırsanız birkaç dakika bekleyip tekrar deneyin. Sunum sırasında daha hafif bir Gemini modeli kullanmak isterseniz `.env` içinde şu satırı geçici olarak değiştirebilirsiniz:

```text
GEMINI_MODEL=gemini-2.5-flash-lite
```

Sonra terminalde çalıştırın:

```bash
python3 -m src.server --port 8001
```

Tarayıcıdan açın:

```text
http://127.0.0.1:8001
```

8001 doluysa farklı bir port verebilirsiniz.

## Proje Hangi Problemi Çözer?

Günümüzde kullanılan sade Türkçe ile eski edebi Türkçe ve halk edebiyatı dili arasında kopukluk vardır. Bu proje, kullanıcının yazdığı modern bir metni seçilen üsluba göre yeniden yazar.

Hedef kitle:

- Edebiyat derslerinde eski metinleri daha anlaşılır ve ilgi çekici hale getirmek isteyen öğrenciler
- Türkçe dil ve üslup çalışmaları yapan kişiler
- Dijital kültürel miras projeleri

## Veri Seti Tanıtımı

Projede kullanıcının verdiği iki CSV veri seti kullanılmıştır.

| Veri seti | Sınıf / Persona | CSV satırı | Kullanılan örnek |
|---|---:|---:|---:|
| `genisletilmis_eski_edebi_metinler.csv` | Eski Edebi Metin | 1721 | 1721 |
| `anadolu_ozanlari_veri_seti.csv` | Anadolu Ozanı | 471 | 386 |
| Toplam | 2 sınıf | 2192 | 2107 |

CSV kolonları:

- `yazar_adi`
- `metin_pasaji`
- `kaynak_link`
- `donem`

Veri formatı metindir. Çok kısa olan bazı Anadolu ozanı başlık satırları eğitim dışında bırakılmıştır.

## Veri Ön İşleme

Modelden önce şu işlemler yapılmıştır:

1. CSV dosyaları okunmuştur.
2. `metin_pasaji` alanı ana metin olarak seçilmiştir.
3. Boş veya çok kısa metinler filtrelenmiştir.
4. Metinler küçük harfe çevrilmiştir.
5. Türkçe karakter uyumu için Unicode normalizasyonu yapılmıştır.
6. Metinler tokenlara ayrılmıştır.
7. Çok sık ve tek başına ayırt edici olmayan stop-word kelimeleri çıkarılmıştır.
8. Naive Bayes değerlendirme modeli için eğitim/test ayrımı yapılmıştır.

## Kullanılan Yapay Zeka Yöntemi

Bu projedeki ana üretim yaklaşımı:

```text
RAG + Gemini 2.5 Flash
```

Çalışma sırası:

1. Kullanıcı modern Türkçe bir metin girer.
2. Seçilen personaya ait CSV pasajları TF-IDF vektörlerine çevrilir.
3. Kullanıcı metni ile veri seti pasajları kosinüs benzerliğiyle karşılaştırılır.
4. En benzer pasajlar RAG bağlamı olarak seçilir.
5. Bu örnekler Gemini promptuna eklenir.
6. Gemini, kullanıcının metnini hedef üslupta yeniden yazar.
7. Naive Bayes modeli, üretilen çıktının hangi persona sınıfına benzediğini ölçer.

## Neden Gemini 2.5 Flash Seçildi?

Bu proje sadece sınıflandırma problemi değildir; asıl amaç metin üretmektir. Naive Bayes gibi klasik modeller sınıf tahmini yapabilir ama doğal ve çeşitli metin üretmekte yeterli değildir.

Gemini 2.5 Flash seçilme sebepleri:

- Türkçe metin üretimi için yeterli kalite sunar.
- Bulutta çalıştığı için bilgisayarda ağır yerel model çalıştırmak gerekmez.
- RAG promptundaki örnekleri takip edebilir.
- Proje demosu için hızlı ve pratiktir.

## Hazır Model Mi Kullanıldı?

Evet. Metin üretimi için hazır **Gemini 2.5 Flash** modeli kullanılmıştır. Bu projede fine-tuning yapılmaz.

Projede sıfırdan yazılan kısımlar:

- CSV okuma ve veri temizleme
- TF-IDF vektörleme
- Kosinüs benzerliğiyle RAG örneği getirme
- Gemini promptu oluşturma
- Naive Bayes çıktı değerlendirme modeli
- Web API ve arayüz

## Kullanılan Kütüphaneler

Harici Python paketi zorunlu değildir.

Python standart kütüphanesi:

- `csv`
- `json`
- `re`
- `math`
- `os`
- `time`
- `unicodedata`
- `urllib`
- `pathlib`
- `collections`
- `dataclasses`
- `http.server`
- `unittest`

Ön yüz:

- HTML
- CSS
- Vanilla JavaScript

LLM sağlayıcısı:

- Google Gemini API

## Eğitim ve Test Sonuçları

Gemini tarafında klasik epoch ile eğitim yapılmaz. Gemini bu projede yeniden eğitilmez; veri setinden getirilen örnekler prompt içinde bağlam olarak kullanılır.

Projede ayrıca Naive Bayes değerlendirme modeli eğitilmiştir. Bu model üretilen çıktının hangi persona sınıfına benzediğini ölçmek için kullanılır.

| Metrik | Sonuç |
|---|---:|
| Accuracy | 98.3 |
| Macro F1 | 97.3 |

Sınıf bazında:

| Sınıf | Precision | Recall | F1 | Test örneği |
|---|---:|---:|---:|---:|
| Eski Edebi Metin | 100.0 | 98.0 | 99.0 | 345 |
| Anadolu Ozanı | 91.8 | 100.0 | 95.7 | 78 |

## Demo Özellikleri

Web arayüzünde:

- Kullanıcı metin girebilir.
- Persona seçebilir.
- Sistem RAG + Gemini ile üslup dönüşümü yapar.
- Veri setinden getirilen benzer örnek pasajlar gösterilir.
- Eski kelimeler için sözlükçe gösterilir.
- Gemini üretim modu görünür.
- Naive Bayes çıktı persona tahmini ve test başarımı ekranda görünür.
- Veri seti özeti ve eğitim/test bilgileri ekranda gösterilir.

## Testleri Çalıştırma

```bash
python3 -m unittest discover -s tests
```

## Dosya Yapısı

```text
kusaklararasi-kopru/
  data/source/
    anadolu_ozanlari_veri_seti.csv
    genisletilmis_eski_edebi_metinler.csv
  public/
    index.html
    styles.css
    app.js
  src/
    server.py
    style_engine.py
  tests/
    test_style_engine.py
```

## Geliştirme Önerileri

Proje geliştirilecek olursa:

- Daha fazla yazar ve dönem eklenebilir.
- Persona sayısı artırılabilir.
- Retrieval için embedding tabanlı arama kullanılabilir.
- Daha büyük paralel veri seti oluşturulursa fine-tuning denenebilir.
- İnsan değerlendirmesiyle üslup başarısı daha güvenilir ölçülebilir.
