# Proje Rehberi

Bu dosya, projeyi hocaya anlatırken elinin altında olacak ana rehberdir. Amaç sadece "hangi dosya ne işe yarıyor" demek değil; projeyi gerçekten anlamanı, gelen sorulara rahat cevap vermeni ve sunumda yanlış iddia kurmamanı sağlamaktır.

## 1. Projenin Son Hali

Projenin adı:

```text
Kuşaklararası Köprü
```

Proje, kullanıcının yazdığı modern Türkçe bir metni seçilen tarihsel/edebi üsluba dönüştürür.

Güncel yapay zeka yaklaşımı:

```text
RAG + Gemini LLM
```

Seçilen LLM:

```text
Google Gemini 2.5 Flash
```

Ek değerlendirme modeli:

```text
TF-IDF + Multinomial Naive Bayes
```

Önemli ayrım:

```text
Gemini metni üretir.
RAG, Gemini'ye veri setinden örnek sağlar.
Naive Bayes ise üretilen çıktının hangi üsluba benzediğini değerlendirir.
```

## 2. Tek Cümlelik Özet

Sunumda en kısa haliyle şöyle diyebilirsin:

```text
Bu projede modern Türkçe bir metni eski edebi metin veya Anadolu ozanı üslubuna dönüştürmek için RAG + Gemini 2.5 Flash kullandım; veri setindeki benzer örnekleri TF-IDF ve kosinüs benzerliğiyle getirip Gemini'ye bağlam olarak veriyorum, ardından Naive Bayes modeliyle üretilen çıktının hedef üsluba ne kadar benzediğini değerlendiriyorum.
```

## 3. Proje Hangi Problemi Çözüyor?

Problem:

Modern Türkçeyle yazılmış günlük cümleleri, eski edebi metinlere veya Anadolu ozanı söyleyişine benzeyen bir dile dönüştürmek.

Neden önemli?

- Eski metinleri anlamak ve onlarla bağ kurmak öğrenciler için zor olabiliyor.
- Edebi üslup farklarını göstermek için etkileşimli bir araç faydalı oluyor.
- Aynı modern cümlenin farklı edebi/persona üsluplarında nasıl söylenebileceği görülebiliyor.

Hedef kitle:

- Edebiyat ve Türkçe dersindeki öğrenciler
- Eski metinlere ilgi duyan kullanıcılar
- Dil, üslup ve kültürel miras çalışmaları yapan kişiler

## 4. Kullanılan Veri Setleri

Veri setleri proje klasöründe burada:

```text
data/source/anadolu_ozanlari_veri_seti.csv
data/source/genisletilmis_eski_edebi_metinler.csv
```

CSV kolonları:

- `yazar_adi`
- `metin_pasaji`
- `kaynak_link`
- `donem`

Projede iki persona/sınıf var:

| Persona | Veri seti | İçerik |
|---|---|---|
| Eski Edebi Metin | `genisletilmis_eski_edebi_metinler.csv` | Ömer Seyfettin, Halit Ziya gibi edebi pasajlar |
| Anadolu Ozanı | `anadolu_ozanlari_veri_seti.csv` | Yunus Emre, Karacaoğlan gibi ozan metinleri |

Kullanılan veri sayıları:

| Bilgi | Değer |
|---|---:|
| Toplam CSV satırı | 2192 |
| Kullanılan metin örneği | 2107 |
| Naive Bayes eğitim örneği | 1684 |
| Naive Bayes test örneği | 423 |
| Sınıf/persona sayısı | 2 |

Burada çok önemli bir nokta var:

```text
Bu veri setleri Gemini'yi eğitmek için kullanılmıyor.
Veri setleri RAG bağlamı ve Naive Bayes değerlendirme modeli için kullanılıyor.
```

Yani bu projede fine-tuning yoktur.

## 5. Veri Ön İşleme

Kodda veri ön işleme işlemleri `src/style_engine.py` içinde yapılır.

Yapılan işlemler:

1. CSV dosyaları okunur.
2. `metin_pasaji` alanı ana metin olarak alınır.
3. Boş veya çok kısa satırlar filtrelenir.
4. Metinler küçük harfe çevrilir.
5. Türkçe karakter uyumu için Unicode normalizasyonu yapılır.
6. Metinler tokenlara ayrılır.
7. Çok sık geçen ve tek başına ayırt edici olmayan stop-word kelimeler çıkarılır.
8. Naive Bayes değerlendirme modeli için veri yaklaşık yüzde 80 eğitim, yüzde 20 test olarak ayrılır.

Kodda bakılacak yerler:

| Fonksiyon | Görevi |
|---|---|
| `normalize()` | Metindeki harf/Unicode farklarını azaltır |
| `tokenize()` | Metni kelimelere ayırır |
| `_load_corpus()` | CSV dosyalarını okur |
| `_split_train_test()` | Eğitim/test ayrımı yapar |

## 6. Kullanılan Model ve Yöntem

Projede tek bir model yok; üç parçalı bir yapı var.

| Parça | Ne işe yarıyor? |
|---|---|
| RAG | Veri setinden benzer örnek pasajları bulup Gemini'ye bağlam sağlar |
| Gemini 2.5 Flash | Kullanıcının metnini hedef üslupta yeniden üretir |
| Naive Bayes | Üretilen çıktının hangi persona sınıfına benzediğini ölçer |

### RAG Nedir?

RAG, "Retrieval Augmented Generation" anlamına gelir.

Bu projedeki karşılığı:

```text
Önce veri setinden benzer örnekleri getir, sonra LLM'e bu örneklerle birlikte metin üretmesini söyle.
```

Bizim projede retrieval kısmı şu şekilde yapılır:

1. Kullanıcının metni TF-IDF vektörüne çevrilir.
2. Seçilen personaya ait veri seti pasajları TF-IDF vektörüne çevrilir.
3. Kullanıcı metni ile veri seti pasajları kosinüs benzerliğiyle karşılaştırılır.
4. En benzer 4 pasaj seçilir.
5. Bu pasajlar Gemini promptuna örnek olarak eklenir.

Kodda bakılacak yerler:

| Fonksiyon | Görevi |
|---|---|
| `retrieve()` | En benzer pasajları bulur |
| `_build_idf()` | TF-IDF için IDF değerlerini hesaplar |
| `_vectorize()` | Metni TF-IDF vektörüne çevirir |
| `_cosine()` | İki metin vektörü arasındaki benzerliği hesaplar |

`_cosine()` içinde karşılaştırılan iki metin:

- Birinci metin: kullanıcının yazdığı modern metin
- İkinci metin: seçilen personaya ait veri setindeki bir pasaj

## 7. Neden Gemini 2.5 Flash Seçildi?

Bu proje metin üretimi yaptığı için sadece klasik sınıflandırma modeli yeterli değildir. Kural tabanlı dönüşüm önceki sürümde çok yapay sonuçlar verebiliyordu.

Gemini 2.5 Flash seçilme sebepleri:

- Bulutta çalıştığı için bilgisayarın RAM'i düşük olsa bile yerel model yüklemek gerekmez.
- Türkçe metin üretimi için yeterlidir.
- RAG promptundaki örnekleri takip edebilir.
- Proje demosu için hızlı cevap verir.
- `gemini-2.5-pro` modeline göre bu proje için daha pratik ve hafiftir.

Doğru anlatım:

```text
Metin üretimi için hazır Gemini 2.5 Flash LLM kullandım.
Bu modeli sıfırdan eğitmedim ve fine-tuning yapmadım.
Kendi veri setimi RAG bağlamı olarak kullandım.
```

Yanlış anlatım:

```text
Gemini'yi kendi veri setimle eğittim.
```

Bu yanlış olur, çünkü projede fine-tuning yapılmadı.

## 8. Naive Bayes Bu Projede Ne Yapıyor?

Naive Bayes metni dönüştüren model değildir.

Bu projedeki görevi:

```text
Gemini tarafından üretilen çıktının Eski Edebi Metin mi yoksa Anadolu Ozanı mı sınıfına daha çok benzediğini tahmin etmek.
```

Kodda bakılacak yerler:

| Fonksiyon | Görevi |
|---|---|
| `_train_classifier()` | Naive Bayes modelini eğitim verisiyle eğitir |
| `_class_scores()` | Bir metin için sınıf log-olasılıklarını hesaplar |
| `predict_persona()` | En yüksek skorlu sınıfı tahmin eder |
| `_evaluate_classifier()` | Test setinde accuracy, precision, recall ve F1 hesaplar |

Naive Bayes sonuçları:

| Metrik | Sonuç |
|---|---:|
| Accuracy | 98.3 |
| Macro F1 | 97.3 |

Bu sonuçların anlamı:

```text
Naive Bayes değerlendirme modeli, veri setindeki test metinlerini iki persona sınıfına ayırmada yüksek başarı göstermiştir.
```

Bu sonuçların anlamı olmayan şey:

```text
Gemini'nin her dönüşüm çıktısı yüzde 98.3 doğrudur.
```

Bu ikinci cümleyi söyleme. Accuracy, Gemini çıktısının kalite yüzdesi değil; Naive Bayes sınıflandırıcısının test başarımıdır.

## 9. Kodda Ana Akış Nerede?

Ana dosya:

```text
src/style_engine.py
```

Ana sınıflar ve fonksiyonlar:

| Kod | Ne işe yarıyor? |
|---|---|
| `StyleEngine` | Projenin ana motoru |
| `StyleEngine.transform()` | Kullanıcı Dönüştür butonuna basınca çalışan ana işlem |
| `LLMClient` | Gemini bağlantısını ve LLM üretimini yönetir |
| `LLMClient._build_prompt()` | RAG örnekleriyle LLM promptunu hazırlar |
| `LLMClient._generate_gemini()` | Google Gemini API çağrısını yapar |
| `retrieve()` | Veri setinden benzer pasajları getirir |
| `predict_persona()` | Üretilen çıktının persona tahminini yapar |

Sunumda "model nerede?" denirse:

```text
Metni üreten ana yapı LLMClient sınıfındaki Gemini çağrısıdır.
RAG tarafı retrieve, _vectorize ve _cosine fonksiyonlarında çalışır.
Naive Bayes değerlendirme modeli _train_classifier ve predict_persona fonksiyonlarında bulunur.
Ana akış ise StyleEngine.transform fonksiyonundadır.
```

## 10. Kullanıcı Butona Basınca Ne Oluyor?

Örnek:

```text
Kullanıcı şunu yazıyor:
Telefonumun şarjı bittiği için sana geç cevap verdim.

Persona:
Eski Edebi Metin
```

Sistem arka planda şu adımları izler:

1. `public/app.js` içindeki `transformText()` çalışır.
2. Kullanıcının metni, seçilen persona ve yoğunluk değeri JSON olarak `/api/transform` adresine gönderilir.
3. `src/server.py` içindeki `do_POST()` isteği alır.
4. `StyleEngine.transform()` çağrılır.
5. Metin boş mu, persona geçerli mi diye kontrol edilir.
6. `retrieve()` fonksiyonu seçilen personaya ait veri setinden en benzer 4 pasajı bulur.
7. `LLMClient._build_prompt()` bu pasajları prompt içine ekler.
8. `LLMClient._generate_gemini()` Gemini 2.5 Flash modeline istek atar.
9. Gemini dönüştürülmüş metni üretir.
10. `predict_persona()` üretilen çıktının hangi personaya benzediğini Naive Bayes ile tahmin eder.
11. Sözlükçe, uyarılar, RAG örnekleri ve metrikler hazırlanır.
12. Sonuç JSON olarak arayüze döner.
13. `renderResult()` sonucu ekranda gösterir.

Kısa akış:

```text
Kullanıcı metni -> API -> RAG örnekleri -> Gemini promptu -> Gemini çıktısı -> Naive Bayes değerlendirme -> arayüz
```

## 11. Prompt İçinde Ne Var?

Prompt `LLMClient._build_prompt()` içinde hazırlanır.

Promptun içinde şunlar vardır:

- Hedef üslup adı
- Hedef üslup açıklaması
- Üslup yoğunluğu
- RAG ile getirilen örnek pasajlar
- Anlamı koruma kuralları
- Modern teknoloji kelimeleri için uyarı
- Kullanıcının modern Türkçe metni

Gemini'ye verilen temel yönerge şu mantıktadır:

```text
Aşağıdaki hedef üslup örneklerini incele ve kullanıcının modern Türkçe cümlesini bu üsluba göre yeniden yaz. Anlamı değiştirme, yeni bilgi ekleme, sadece dönüştürülmüş metni yaz.
```

Bu yüzden çıktı daha doğal olur. Model sadece kelime değiştirme yapmaz; örnekleri ve bağlamı kullanarak cümleyi yeniden kurar.

## 12. Arayüzde Neyi Göstermelisin?

Demo sırasında şu alanları özellikle göster:

1. Sol taraftaki metin kutusu: kullanıcı modern Türkçe metin giriyor.
2. Persona seçimi: Eski Edebi Metin veya Anadolu Ozanı.
3. Dönüştür butonu.
4. Sağdaki çıktı: Gemini'nin ürettiği dönüşüm.
5. RAG örnekleri sekmesi: veri setinden getirilen benzer pasajlar.
6. Sözlükçe sekmesi: eski kelimelerin açıklamaları.
7. Ölçüm sekmesi: üretim modu, Naive Bayes persona tahmini, test accuracy.

Ölçüm sekmesinde özellikle kontrol et:

```text
Üretim Modu
gemini / gemini-2.5-flash
```

Gemini API anahtarı eksikse veya API çağrısı başarısız olursa proje kural tabanlı üretime dönmez; ekranda hata gösterir.

## 13. Örnek Demo Cümleleri

Gemini'nin kural tabanlı sisteme göre daha iyi çalıştığını göstermek için teknoloji veya günlük dil içeren cümleler iyi olur.

Eski Edebi Metin için:

```text
Telefonumun şarjı bittiği için sana geç cevap verdim.
```

```text
Bugün okula gitmem gerekiyor ama çok yorgunum.
```

```text
İnternette araştırma yaparken eski bir şiire rastladım.
```

Anadolu Ozanı için:

```text
Uzun zamandır seni görmedim, biraz konuşmak iyi gelir.
```

```text
Akşam eve dönerken içimde garip bir hüzün vardı.
```

```text
Arkadaşımla kahve içip biraz dertleşmek istiyorum.
```

Sunum için en iyi örnek:

```text
Telefonumun şarjı bittiği için sana geç cevap verdim.
```

Çünkü bu cümlede modern teknoloji kelimesi var. Kural tabanlı sistem bunu daha yapay çevirebilirken Gemini anlamı yorumlayıp daha doğal bir edebi karşılık üretebilir.

## 14. Yönergedeki Başlıklara Göre Cevaplar

### Proje Tanıtımı

Projenin adı Kuşaklararası Köprü. Proje, modern Türkçe cümleleri eski edebi metin veya Anadolu ozanı üslubuna dönüştürmeyi amaçlar. Bu problem önemlidir çünkü öğrencilerin eski metinlerle bağ kurmasını ve farklı üslup özelliklerini uygulamalı olarak görmesini sağlar.

### Veri Seti Tanıtımı

Projede iki CSV veri seti kullanıldı. Birincisi eski edebi metinlerden, ikincisi Anadolu ozanları metinlerinden oluşuyor. Toplam 2192 CSV satırı var; çok kısa satırlar çıkarıldıktan sonra 2107 kullanılabilir metin örneği elde edildi. Veriler metin formatındadır.

### Veri Ön İşleme

CSV dosyaları okundu, boş/kısa metinler temizlendi, metinler normalize edildi, tokenlara ayrıldı ve stop-word kelimeler çıkarıldı. Naive Bayes değerlendirme modeli için veri yaklaşık yüzde 80 eğitim ve yüzde 20 test olarak ayrıldı.

### Kullanılan Yapay Zeka Yöntemi

Ana yöntem RAG + Gemini 2.5 Flash LLM'dir. RAG tarafında TF-IDF ve kosinüs benzerliği kullanılarak veri setinden benzer pasajlar getirilir. Gemini bu örnekleri bağlam olarak alıp metni hedef üslupta üretir. Ayrıca TF-IDF + Multinomial Naive Bayes modeli çıktı değerlendirmesi için kullanılır.

### Eğitim Süreci ve Deneyler

Gemini tarafında fine-tuning yapılmadı, bu yüzden epoch yoktur. RAG yaklaşımında veri seti her istek sırasında bağlam olarak kullanılır. Naive Bayes değerlendirme modeli ise veri setiyle eğitildi ve test edildi. Test sonucunda accuracy 98.3, macro F1 97.3 bulundu.

### Demo Gösterimi

Kullanıcı metin girer, persona seçer ve Dönüştür butonuna basar. Sistem RAG ile benzer örnekleri getirir, Gemini'ye prompt gönderir ve dönüşüm sonucunu ekranda gösterir. RAG örnekleri, sözlükçe ve ölçüm sekmeleriyle süreç açıklanabilir hale getirilir.

### Sonuç ve Değerlendirme

Proje hedefe ulaştı. Modern Türkçe bir metin seçilen üsluba dönüştürülebiliyor. En güçlü tarafı, sadece kural tabanlı kelime değiştirmek yerine RAG + Gemini ile daha doğal metin üretmesi. Geliştirme olarak daha büyük veri seti, embedding tabanlı arama ve insan değerlendirmesi eklenebilir.

## 15. Öğretmenin Sorabileceği Sorular

**Hangi algoritma veya model kullanıldı?**

RAG + Gemini 2.5 Flash LLM kullanıldı. RAG kısmında TF-IDF ve kosinüs benzerliği var. Ek olarak çıktı değerlendirmesi için TF-IDF + Multinomial Naive Bayes modeli kullanıldı.

**Neden Gemini 2.5 Flash seçildi?**

Çünkü proje metin üretimi yapıyor ve yerel model çalıştırmak RAM açısından zor olabilir. Gemini bulutta çalıştığı için bilgisayarı yormaz, Türkçe üretim için yeterlidir ve RAG örneklerini kullanarak doğal sonuçlar verebilir.

**Modelin temel çalışma mantığı nedir?**

Önce kullanıcının metni veri setindeki pasajlarla karşılaştırılır. En benzer pasajlar seçilir ve Gemini promptuna eklenir. Gemini bu bağlama göre metni hedef üslupta yeniden yazar. Sonra Naive Bayes modeli çıktının hangi personaya benzediğini ölçer.

**Hazır model mi kullanıldı, sıfırdan mı geliştirildi?**

Metin üretiminde hazır Gemini 2.5 Flash modeli kullanıldı. Model sıfırdan eğitilmedi ve fine-tuning yapılmadı. Ancak RAG sistemi, TF-IDF/kosinüs benzerliği, prompt hazırlama, Naive Bayes değerlendirme modeli, API ve arayüz proje içinde geliştirildi.

**Veri seti Gemini'yi eğitiyor mu?**

Hayır. Veri seti Gemini'yi eğitmez. Veri seti, RAG sırasında Gemini'ye örnek göstermek için kullanılır. Ayrıca Naive Bayes değerlendirme modelinin eğitim/test sürecinde kullanılır.

**Accuracy hangi modele ait?**

Accuracy, Naive Bayes değerlendirme modelinin test seti başarımıdır. Gemini çıktısının yüzde 98.3 doğru olduğu anlamına gelmez.

**RAG örneklerinde benzerlik neden bazen düşük görünüyor?**

Çünkü eski edebi metinlerle modern kullanıcı cümlesi birebir aynı kelimeleri kullanmayabilir. TF-IDF/kosinüs benzerliği kelime izlerine bakar. Yine de en yakın örnekleri seçerek Gemini'ye hedef üslup hakkında bağlam sağlar.

**Modern teknoloji uyarısı neden çıkıyor?**

Telefon, internet, uygulama gibi kelimelerin eski dönem metinlerinde doğrudan karşılığı zayıf olduğu için sistem uyarı verir. Bu hata değildir; sadece modelin bu kelimeleri yorumlayarak dönüştürmesi gerektiğini gösterir.

**Projede TensorFlow, PyTorch veya Scikit-learn var mı?**

Hayır. Bu proje Python standart kütüphaneleriyle yazıldı. TF-IDF, kosinüs benzerliği ve Naive Bayes mantığı proje içinde kodlandı. Gemini için API çağrısı `urllib` ile yapıldı.

**Projede kural tabanlı metin üretimi kaldı mı?**

Hayır. Önceki kural tabanlı üretim ve yedek üretici kodları kaldırıldı. Metni üreten yapı Gemini'dir. Gemini API çalışmazsa proje eski kurallarla çıktı uydurmaz, kullanıcıya hata gösterir.

## 16. Kullanılan Kütüphaneler

Python standart kütüphaneleri:

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

## 17. Dosya Dosya Ne İşe Yarıyor?

| Dosya | Açıklama |
|---|---|
| `src/style_engine.py` | Ana yapay zeka motoru. Veri seti okuma, RAG, Gemini çağrısı, Naive Bayes değerlendirme burada |
| `src/server.py` | Yerel web sunucusu. `/api/personas`, `/api/dataset`, `/api/transform` API uçlarını yönetir |
| `public/app.js` | Arayüzün davranışı. Kullanıcı metnini API'ye gönderir ve sonucu ekrana basar |
| `public/index.html` | Sayfa iskeleti |
| `public/styles.css` | Görsel tasarım |
| `data/source/*.csv` | Projenin gerçek veri setleri |
| `.env` | Gemini API key ve model ayarları |
| `.env.example` | `.env` dosyası için örnek |
| `README.md` | Genel proje açıklaması |
| `tests/test_style_engine.py` | Temel testler |

Zamanın azsa en çok şu dosyalara odaklan:

1. `src/style_engine.py`
2. `data/source/`
3. `public/app.js`
4. Çalışan web arayüzü

## 18. Çalıştırma

Proje klasörü:

```bash
cd /Users/erencalban/Desktop/kusaklararasi-kopru
```

`.env` dosyasında şu bilgiler olmalı:

```text
GEMINI_API_KEY=google_ai_studio_keyin
GEMINI_MODEL=gemini-2.5-flash
```

429 Too Many Requests hatası alırsan bu genellikle Gemini tarafındaki hız/kota sınırıdır. Birkaç dakika bekleyip tekrar deneyebilirsin. Sunumda daha hafif model kullanman gerekirse `.env` içindeki model satırını geçici olarak şöyle değiştirebilirsin:

```text
GEMINI_MODEL=gemini-2.5-flash-lite
```

Sunucuyu başlat:

```bash
python3 -m src.server --port 8001
```

Tarayıcıda aç:

```text
http://127.0.0.1:8001
```

Port doluysa:

```bash
python3 -m src.server --port 8002
```

## 19. Sunum İçin 5-10 Dakikalık Akış

1. Kendini ve proje adını söyle.
2. Problemi anlat: modern Türkçeyi eski edebi/ozan üslubuna dönüştürmek.
3. Veri setlerini göster: `data/source` klasöründe iki CSV var.
4. Kullanılan yöntemi söyle: RAG + Gemini 2.5 Flash.
5. RAG'i açıkla: TF-IDF ve kosinüs benzerliğiyle benzer pasajlar getiriliyor.
6. Gemini'yi açıkla: Bu örnekleri kullanarak metni yeniden yazıyor.
7. Naive Bayes'i açıkla: Çıktının hangi personaya benzediğini değerlendiriyor.
8. Arayüzde demo yap.
9. RAG örnekleri sekmesini göster.
10. Ölçüm sekmesinde `gemini / gemini-2.5-flash` yazdığını göster.
11. Accuracy ve F1 değerlerinin Naive Bayes değerlendirme modeline ait olduğunu söyle.
12. Sonuç ve geliştirme önerileriyle bitir.

## 20. Kısa Konuşma Metni

Şunu doğal şekilde okuyup kendi cümlelerinle anlatabilirsin:

```text
Bu projede Kuşaklararası Köprü adında bir metin dönüştürme uygulaması geliştirdim. Amacım, modern Türkçe ile yazılan bir cümleyi eski edebi metin veya Anadolu ozanı üslubuna dönüştürmekti. Bunun için RAG + Gemini 2.5 Flash yaklaşımını kullandım.

Kullanıcı bir metin yazdığında sistem önce seçilen personaya ait veri setindeki metinlerle bu girdiği metni karşılaştırıyor. Bu karşılaştırmada TF-IDF vektörleme ve kosinüs benzerliği kullanılıyor. En benzer dört pasaj seçiliyor ve Gemini'ye gönderilen promptun içine örnek olarak ekleniyor. Böylece Gemini sadece genel bilgisini değil, benim veri setimdeki üslup örneklerini de dikkate alarak çıktı üretiyor.

Ek olarak projede TF-IDF + Multinomial Naive Bayes modeli de var. Bu model metni üretmiyor; üretilen çıktının Eski Edebi Metin mi yoksa Anadolu Ozanı mı sınıfına daha çok benzediğini değerlendirmek için kullanılıyor. Bu model test setinde 98.3 accuracy ve 97.3 macro F1 sonucuna ulaştı.

Projede fine-tuning yapmadım. Gemini hazır bir büyük dil modeli olarak kullanıldı. Benim kendi katkım; veri setini hazırlayıp kullanmak, RAG getirme mekanizmasını kurmak, promptu oluşturmak, Naive Bayes değerlendirme modelini yazmak ve web arayüzünü geliştirmek oldu.
```

## 21. Yanlış Söylememen Gerekenler

Şunları söyleme:

```text
Gemini'yi ben eğittim.
```

```text
Bu projede fine-tuning yaptım.
```

```text
Naive Bayes metni dönüştürüyor.
```

```text
Accuracy 98.3 olduğu için Gemini çıktıları yüzde 98.3 doğru.
```

Doğru halleri:

```text
Gemini hazır model olarak kullanıldı.
```

```text
Fine-tuning yapılmadı, RAG kullanıldı.
```

```text
Metni Gemini üretiyor; Naive Bayes çıktıyı değerlendiriyor.
```

```text
Accuracy, Naive Bayes değerlendirme modelinin test seti başarımıdır.
```

## 22. Geliştirme Önerileri

Projeyi geliştirmek isteseydim şunları ekleyebilirdim:

- TF-IDF yerine embedding tabanlı arama kullanmak
- Daha fazla yazar ve dönem eklemek
- Persona sayısını artırmak
- İnsan değerlendirmesiyle çıktı kalitesini ölçmek
- Daha büyük paralel veri seti oluşturup fine-tuning denemek
- RAG örneklerini daha dengeli seçmek

## 23. En Son Kontrol Listesi

Sunumdan önce şunları kontrol et:

- `.env` dosyasında `GEMINI_API_KEY` var mı?
- Sunucu hatasız açılıyor mu?
- Arayüzde dönüşüm yapınca çıktı geliyor mu?
- Ölçüm sekmesinde `gemini / gemini-2.5-flash` görünüyor mu?
- RAG örnekleri sekmesinde veri setinden örnekler geliyor mu?
- Sunumda fine-tuning yaptığını söylemiyorsun, RAG yaptığını söylüyorsun.
