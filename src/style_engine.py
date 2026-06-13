from __future__ import annotations

# Bu dosya projenin ana "yapay zeka motoru"dur.
# Veri setini okuma, metin ön işleme, RAG için benzer örnek getirme,
# LLM promptu oluşturma, LLM ile metin üretme ve üretilen çıktıyı
# Naive Bayes modeliyle değerlendirme işlemlerinin tamamı burada yapılır.

import csv
import json
import math
import os
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ENV_PATH = PROJECT_ROOT / ".env"


def load_local_env() -> None:
    # Proje klasöründeki .env dosyasını okur.
    # Böylece kullanıcı API key'i terminalde her seferinde export etmek yerine
    # .env dosyasında saklayabilir. Terminalden verilen değerler varsa onları ezmez.
    if not LOCAL_ENV_PATH.exists():
        return

    for raw_line in LOCAL_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


# Metinleri kelimelere ayırmak için kullanılan düzenli ifade.
# Türkçe karakterleri de kapsayacak şekilde hazırlanmıştır.
TOKEN_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+", re.UNICODE)

# Stop words: Sınıflandırmada çok ayırt edici olmayan sık kelimeler.
# Örneğin "ve", "ile", "ama" gibi kelimeler metnin üslubunu tek başına
# çok iyi göstermediği için token listesinden çıkarılır.
STOP_WORDS = {
    "acaba",
    "ama",
    "ancak",
    "ben",
    "bir",
    "bu",
    "da",
    "de",
    "diye",
    "gibi",
    "icin",
    "için",
    "ile",
    "ise",
    "mi",
    "mı",
    "mu",
    "mü",
    "ne",
    "o",
    "ve",
    "veya",
    "ya",
}

# Çıktıda geçen eski/edebi kelimelerin kullanıcıya açıklanması için kullanılır.
# Arayüzde "Sözlükçe" sekmesini bu sözlük besler.
GLOSSARY = {
    "afiyet": "Sağlık, iyi olma hali.",
    "ahbap": "Yakın arkadaş, dost.",
    "azimet": "Yola çıkma, gitme.",
    "bendeniz": "Ben anlamında alçakgönüllü hitap.",
    "dem": "An, vakit.",
    "diyar": "Memleket, yer, bölge.",
    "dost": "Yakın arkadaş, sevilen kişi.",
    "efendim": "Saygılı hitap sözü.",
    "gönül": "Kalp, iç dünya, duygu merkezi.",
    "hane": "Ev.",
    "kelam": "Söz.",
    "latif": "Hoş, zarif, güzel.",
    "mahzun": "Hüzünlü.",
    "mektep": "Okul.",
    "meclis": "Topluluk, sohbet ortamı.",
    "menzil": "Varılacak yer, durak.",
    "meşgale": "Uğraş, iş.",
    "meşgul": "Bir işle uğraşan.",
    "muhabbet": "Sevgiyle sohbet, yakınlık.",
    "münasip": "Uygun, yerinde.",
    "müsaade": "İzin.",
    "müteşekkir": "Teşekkür eden, minnettar.",
    "nida": "Sesleniş.",
    "seda": "Ses, yankı.",
    "sevda": "Derin sevgi.",
    "sıhhat": "Sağlık.",
    "vakit": "Zaman.",
    "ziyadesiyle": "Fazlasıyla, çok.",
}

# Eski metinlerde doğrudan karşılığı zayıf olabilecek modern teknoloji kelimeleri.
# Bu kelimeler girilirse arayüzde uyarı gösterilir.
TECH_TERMS = {
    "akıllı telefon",
    "bilgisayar",
    "internet",
    "mail",
    "mesaj",
    "sosyal medya",
    "telefon",
    "uygulama",
}


@dataclass(frozen=True)
class PersonaProfile:
    # Her üslup/persona için arayüz etiketi, açıklama
    # ve o üslubu temsil eden anahtar kelimeler bu yapıda tutulur.
    key: str
    label: str
    short_label: str
    description: str
    style_markers: tuple[str, ...]


@dataclass(frozen=True)
class LLMResult:
    # LLM üretiminden dönen sonucu tek yerde toplar.
    # provider: Bu projede Gemini kaynak bilgisidir.
    # used_llm: Gerçek bir LLM çağrısı başarılı oldu mu?
    # prompt: LLM'e gönderilen RAG promptu. Rehberlik ve açıklama için saklanır.
    text: str
    provider: str
    model: str
    used_llm: bool
    prompt: str


# Projede kullanılan iki sınıf/persona burada tanımlıdır.
# Model bu iki sınıf arasında tahmin yapar:
# 1. Eski Edebi Metin
# 2. Anadolu Ozanı
PERSONAS: dict[str, PersonaProfile] = {
    "istanbul_beyefendisi": PersonaProfile(
        key="istanbul_beyefendisi",
        label="Eski Edebi Metin",
        short_label="Edebi",
        description="Ömer Seyfettin ve Halit Ziya metinlerinden beslenen tarihsel edebi anlatım.",
        style_markers=(
            "efendim",
            "bendeniz",
            "müsaadenizle",
            "vakit",
            "münasip",
            "sıhhat",
            "ziyadesiyle",
            "hane",
            "mektep",
        ),
    ),
    "anadolu_ozani": PersonaProfile(
        key="anadolu_ozani",
        label="Anadolu Ozanı",
        short_label="Ozan",
        description="Dost, gönül, yol ve sevda imgeleriyle daha şiirsel bir söyleyiş.",
        style_markers=(
            "dost",
            "gönül",
            "yol",
            "sevda",
            "diyar",
            "seda",
            "menzil",
            "ocak",
        ),
    ),
}

# Hangi CSV dosyasının hangi sınıfa/personaya ait olduğunu gösteren yapı.
# Veri seti buradan okunur ve her satıra ilgili sınıf etiketi verilir.
DATASET_SPECS = (
    {
        "file": "genisletilmis_eski_edebi_metinler.csv",
        "persona": "istanbul_beyefendisi",
        "name": "Genişletilmiş Eski Edebi Metinler",
        "format": "CSV / metin pasajı",
    },
    {
        "file": "anadolu_ozanlari_veri_seti.csv",
        "persona": "anadolu_ozani",
        "name": "Anadolu Ozanları Veri Seti",
        "format": "CSV / şiir ve halk edebiyatı pasajı",
    },
)


def normalize(text: str) -> str:
    # Türkçedeki büyük İ harfi gibi Unicode farklarını azaltır.
    # Amaç aynı kelimenin farklı yazım biçimlerini model için aynı hale getirmektir.
    folded = text.casefold().replace("i\u0307", "i")
    return unicodedata.normalize("NFC", folded)


def tokenize(text: str) -> list[str]:
    # Metni önce normalize eder, sonra kelimelere ayırır.
    # Stop-word olan ve çok kısa olan tokenlar çıkarılır.
    return [
        token
        for token in TOKEN_RE.findall(normalize(text))
        if token and token not in STOP_WORDS and len(token) > 1
    ]


def _clean_spaces(text: str) -> str:
    # CSV'den gelen fazla boşlukları temizler.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class LLMClient:
    # RAG + LLM üretim katmanı.
    # Bu sınıf veri setinden gelen benzer örnekleri prompt içine koyar
    # ve Gemini'den dönüştürülmüş metin ister.
    def __init__(self) -> None:
        load_local_env()
        self.provider = "gemini"
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        self.temperature = self._read_temperature()

    @staticmethod
    def _read_temperature() -> float:
        # LLM_TEMPERATURE hatalı girilirse sunucu açılışta çökmesin diye
        # güvenli şekilde varsayılan değere döner.
        try:
            return float(os.getenv("LLM_TEMPERATURE", "0.35"))
        except ValueError:
            return 0.35

    def generate(
        self,
        source_text: str,
        profile: PersonaProfile,
        retrieved_examples: list[dict[str, Any]],
        intensity: float,
    ) -> LLMResult:
        # LLM'e gidecek prompt önce RAG örnekleriyle hazırlanır.
        prompt = self._build_prompt(source_text, profile, retrieved_examples, intensity)
        return self._generate_gemini(prompt)

    def _build_prompt(
        self,
        source_text: str,
        profile: PersonaProfile,
        retrieved_examples: list[dict[str, Any]],
        intensity: float,
    ) -> str:
        # RAG'in asıl kısmı burasıdır:
        # Veri setinden bulunan benzer örnekler LLM'e bağlam olarak verilir.
        examples = "\n\n".join(
            f"Örnek {index} ({example['author']}):\n{example['text']}"
            for index, example in enumerate(retrieved_examples, start=1)
        )
        intensity_label = "hafif" if intensity < 0.45 else "orta" if intensity < 0.75 else "güçlü"
        return f"""Aşağıdaki hedef üslup örneklerini incele ve kullanıcının modern Türkçe cümlesini bu üsluba göre yeniden yaz.
Hedef üslup: {profile.label}
Üslup açıklaması: {profile.description}
Üslup yoğunluğu: {intensity_label}

Hedef üslup örnekleri:
{examples}

Kurallar:
- Anlamı değiştirme.
- Yeni olay, kişi veya bilgi ekleme.
- Çok yapay, aşırı süslü veya tek kalıp bir cümle kurma.
- Modern teknoloji kelimesi varsa uydurma tarihsel karşılık üretme; anlamı koru.
- Sadece dönüştürülmüş metni yaz, açıklama ekleme.

Kullanıcının modern Türkçe metni:
{source_text}
"""

    def _generate_gemini(
        self,
        prompt: str,
    ) -> LLMResult:
        # Google AI Studio'dan alınan Gemini API key ile çalışan bölüm.
        # Gemini'nin native generateContent uç noktasına istek gönderilir.
        api_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY bulunamadı. .env dosyasına Gemini API anahtarını ekleyin.")

        payload = {
            "systemInstruction": {
                "parts": [{"text": "Sen Türkçe edebi üslup dönüşümü yapan dikkatli bir dil modelisin."}]
            },
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        retryable_statuses = {429, 500, 502, 503, 504}
        last_error: Exception | None = None
        model = urllib.parse.quote(self.gemini_model, safe="")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        for attempt in range(3):
            request = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    data = json.loads(response.read().decode("utf-8"))
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                return LLMResult(
                    text=text,
                    provider="gemini",
                    model=self.gemini_model,
                    used_llm=True,
                    prompt=prompt,
                )
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in retryable_statuses or attempt == 2:
                    break
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt == 2:
                    break
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = exc
                break

            # 503 gibi geçici yoğunluk hatalarında kısa bekleyip tekrar dener.
            time.sleep(1.5 * (attempt + 1))

        raise ValueError(f"Gemini LLM çağrısı başarısız oldu: {last_error}")


class StyleEngine:
    # Uygulamanın ana sınıfıdır.
    # Sunucu açıldığında bir kez oluşturulur; veri setini okur, RAG indeksini hazırlar,
    # değerlendirme modelini eğitir ve API isteklerinde Gemini üretimini yönetir.
    def __init__(self, data_path: Path):
        # 1. CSV veri setlerini oku.
        self.data_path = Path(data_path)
        self.corpus, self._source_stats = self._load_corpus(self.data_path)

        # 2. Benzer pasajları bulmak için TF-IDF değerlerini hazırla.
        self._idf = self._build_idf()

        # 3. Veri setini eğitim ve test olarak ayır.
        self.train_set, self.test_set = self._split_train_test()

        # 4. Naive Bayes sınıflandırıcıyı eğitim verisiyle eğit.
        # Bu model artık metni üreten ana model değil; Gemini çıktısının seçilen
        # üsluba benzerliğini ölçmek ve proje metriklerini göstermek için kullanılır.
        self._classifier = self._train_classifier()

        # 5. Test verisiyle accuracy, precision, recall ve F1 hesapla.
        self.training_report = self._evaluate_classifier()

        # 6. RAG örnekleriyle prompt kuran ve Gemini üretimini yapan katman.
        self.llm_client = LLMClient()

    def personas(self) -> list[dict[str, str]]:
        # Arayüzde gösterilecek persona seçeneklerini döndürür.
        return [
            {
                "key": profile.key,
                "label": profile.label,
                "shortLabel": profile.short_label,
                "description": profile.description,
            }
            for profile in PERSONAS.values()
        ]

    def dataset_summary(self) -> dict[str, Any]:
        # Arayüzdeki "Veri Seti ve Eğitim" bölümünü besler.
        # Kaç CSV satırı var, kaç örnek eğitim/test için kullanıldı gibi bilgiler burada hazırlanır.
        by_persona: dict[str, dict[str, Any]] = {}
        for key, profile in PERSONAS.items():
            entries = [entry for entry in self.corpus if entry["persona"] == key]
            authors = Counter(entry["author"] for entry in entries).most_common(6)
            by_persona[key] = {
                "label": profile.label,
                "examples": len(entries),
                "trainExamples": len([entry for entry in self.train_set if entry["persona"] == key]),
                "testExamples": len([entry for entry in self.test_set if entry["persona"] == key]),
                "topAuthors": [{"name": name, "count": count} for name, count in authors],
            }

        return {
            "totalRowsInCsv": sum(item["sourceRows"] for item in self._source_stats),
            "usableExamples": len(self.corpus),
            "trainExamples": len(self.train_set),
            "testExamples": len(self.test_set),
            "split": "Her sınıf/persona içinde deterministik 80/20 eğitim-test ayrımı",
            "files": self._source_stats,
            "byPersona": by_persona,
        }

    def model_report(self) -> dict[str, Any]:
        # Arayüzde ve README'de görülen model açıklaması ve metrikleri burada hazırlanır.
        return {
            "algorithm": "RAG + Gemini 2.5 Flash metin üretimi; TF-IDF + Naive Bayes çıktı değerlendirme modeli",
            "generationApproach": "TF-IDF/kosinüs ile benzer veri seti pasajları getirilir, bu örnekler Gemini promptuna bağlam olarak eklenir ve metin Gemini tarafından üretilir.",
            "trainingType": "Gemini hazır LLM olarak kullanılır; proje içinde eğitilen Naive Bayes modeli çıktı üslubunu değerlendirir.",
            "epochs": "RAG ve Naive Bayes tarafında epoch yoktur. Naive Bayes eğitim verisini bir kez tarar; Gemini tarafında bu projede yeniden eğitim yapılmaz.",
            "llmProvider": self.llm_client.provider,
            "libraries": [
                "Python standart kütüphanesi: csv, json, re, math, pathlib, collections, urllib, http.server",
                "Ön yüz: HTML, CSS, Vanilla JavaScript",
                "LLM sağlayıcısı: Google Gemini API",
            ],
            "metrics": self.training_report,
        }

    def transform(self, text: str, persona: str, intensity: float = 0.65) -> dict[str, Any]:
        # Kullanıcı "Dönüştür" butonuna bastığında çalışan ana fonksiyon.
        # RAG örneklerini bulur, Gemini'ye prompt hazırlar, çıktı üretir
        # ve üretilen çıktıyı Naive Bayes modeliyle değerlendirir.
        text = text.strip()
        if not text:
            raise ValueError("Metin boş olamaz.")
        if persona not in PERSONAS:
            raise ValueError("Bilinmeyen persona seçildi.")

        intensity = min(1.0, max(0.15, float(intensity)))
        profile = PERSONAS[persona]

        # Seçilen persona için veri setindeki en benzer pasajları getirir.
        # Bu pasajlar Gemini promptunda bağlam olarak kullanılacağı için RAG'in
        # "retrieval" yani getirme adımıdır.
        retrieved = self.retrieve(text, persona, top_n=4)

        # Dönüştürülmüş metni Gemini üretir. Eğer GEMINI_API_KEY ayarlı değilse
        # kural tabanlı üretime düşmeden açık hata döndürülür.
        generation = self.llm_client.generate(text, profile, retrieved, intensity)
        transformed = generation.text

        # Çıktıdaki eski kelimelerin anlamlarını bulur.
        glossary = self.explain_terms(transformed)

        # Dönüşüm çıktısı için sezgisel kalite puanları üretir.
        metrics = self.evaluate(text, transformed, profile, intensity)

        # Modern teknoloji kelimeleri varsa uyarı oluşturur.
        warnings = self._warnings(text)

        # Naive Bayes modeliyle üretilen çıktının hangi sınıfa benzediğini tahmin eder.
        # Böylece model, dönüşüm sonucunun hedef personaya yaklaşıp yaklaşmadığını
        # ölçmek için kullanılır.
        predicted = self.predict_persona(transformed)

        return {
            "input": text,
            "persona": {
                "key": profile.key,
                "label": profile.label,
                "description": profile.description,
            },
            "output": transformed,
            "retrievedExamples": retrieved,
            "glossary": glossary,
            "metrics": metrics,
            "predictedPersona": predicted,
            "llm": {
                "provider": generation.provider,
                "model": generation.model,
                "usedLLM": generation.used_llm,
                "prompt": generation.prompt,
            },
            "trainingReport": self.training_report,
            "warnings": warnings,
            "pipeline": [
                "Metin temizleme ve tokenization",
                "RAG getirme: TF-IDF ve kosinüs benzerliğiyle persona havuzundan benzer pasaj seçme",
                "Benzer pasajları Gemini promptuna bağlam olarak ekleme",
                "Gemini ile hedef üslupta metin üretme",
                "Naive Bayes modeliyle üretilen çıktının persona benzerliğini ölçme",
                "Sözlükçe, uyarı ve kalite ölçümü hazırlama",
            ],
        }

    def retrieve(self, text: str, persona: str, top_n: int = 3) -> list[dict[str, Any]]:
        # RAG örnek getirme bölümüdür.
        # Kullanıcı metni ve veri setindeki pasajlar TF-IDF vektörüne çevrilir.
        # Sonra kosinüs benzerliği ile en yakın pasajlar seçilir.
        query_vector = self._vectorize(text)
        scored = []
        for entry in self.corpus:
            if entry["persona"] != persona:
                continue
            score = self._cosine(query_vector, self._vectorize(entry["text"]))
            scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "id": entry["id"],
                "title": entry["title"],
                "author": entry["author"],
                "text": entry["text"],
                "score": round(score, 3),
            }
            for score, entry in scored[:top_n]
        ]

    def predict_persona(self, text: str) -> dict[str, Any]:
        # Naive Bayes modelinin verilen metin için sınıf tahmini yaptığı yer.
        # Bu proje akışında çoğunlukla Gemini'nin ürettiği çıktı burada değerlendirilir.
        # _class_scores her sınıf için log-olasılık üretir; en yüksek skor tahmin edilir.
        scores = self._class_scores(text)
        predicted_key = max(scores, key=scores.get)
        max_score = scores[predicted_key]
        exp_scores = {key: math.exp(value - max_score) for key, value in scores.items()}
        total = sum(exp_scores.values()) or 1.0
        probabilities = {
            key: round(value / total, 3)
            for key, value in sorted(exp_scores.items(), key=lambda item: item[1], reverse=True)
        }
        profile = PERSONAS[predicted_key]
        return {
            "key": predicted_key,
            "label": profile.label,
            "confidence": probabilities[predicted_key],
            "probabilities": probabilities,
        }

    def explain_terms(self, text: str) -> list[dict[str, str]]:
        # Çıktıdaki eski/edebi kelimeleri GLOSSARY içinde arar ve anlamlarını döndürür.
        normalized_output = normalize(text)
        found = []
        for term, meaning in GLOSSARY.items():
            if normalize(term) in normalized_output:
                found.append({"term": term, "meaning": meaning})
        return found

    def evaluate(
        self, input_text: str, output_text: str, profile: PersonaProfile, intensity: float
    ) -> dict[str, dict[str, Any]]:
        # Dönüştürülen cümle için basit değerlendirme puanları üretir.
        # Bunlar sınıflandırma metriği değil, çıktı kalitesini kullanıcıya göstermek içindir.
        input_tokens = set(tokenize(input_text))
        output_tokens = set(tokenize(output_text))
        overlap = len(input_tokens & output_tokens) / max(1, len(input_tokens))
        marker_count = sum(1 for marker in profile.style_markers if normalize(marker) in normalize(output_text))
        terminal_ok = output_text[-1] in ".!?"
        word_count = len(tokenize(output_text))

        semantic = round(min(100, 55 + overlap * 45))
        style = round(min(100, 35 + marker_count * 10 + intensity * 25))
        fluency = round(min(100, 62 + (12 if terminal_ok else 0) + min(word_count, 26)))

        return {
            "semanticFidelity": {
                "label": "Anlam Sadakati",
                "score": semantic,
                "note": "Ortak anahtar kelime ve anlam izi korunumu üzerinden sezgisel hesaplanır.",
            },
            "styleStrength": {
                "label": "Üslup Başarısı",
                "score": style,
                "note": "Persona kelime işaretleri ve seçilen yoğunluk birlikte değerlendirilir.",
            },
            "fluency": {
                "label": "Akıcılık",
                "score": fluency,
                "note": "Cümle sonu, uzunluk ve okunabilirlik için basit ölçüm.",
            },
        }

    def _warnings(self, text: str) -> list[str]:
        # Kullanıcı modern teknoloji kelimeleri yazarsa uyarı üretir.
        # Örneğin "internet" eski dönem metinlerinde doğrudan geçmediği için riskli kabul edilir.
        normalized_text = normalize(text)
        text_tokens = tokenize(text)
        terms = sorted(
            term
            for term in TECH_TERMS
            if normalize(term) in normalized_text
            or any(token.startswith(normalize(term)) for token in text_tokens)
        )
        if not terms:
            return []
        joined = ", ".join(terms)
        return [
            f"Tarihsel karşılığı sınırlı olabilecek modern kavramlar bulundu: {joined}. Çıktı bu kelimeleri yorumlayabilir."
        ]

    def _load_corpus(self, data_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        # CSV veri setlerini okuyan fonksiyon.
        # Her satırdan metin, yazar, kaynak ve dönem bilgisi alınır.
        # Ayrıca hangi dosyadan geldiyse ona göre persona etiketi eklenir.
        if data_path.is_file() and data_path.suffix == ".json":
            entries = json.loads(data_path.read_text(encoding="utf-8"))
            stats = [
                {
                    "name": "Eski demo JSON havuzu",
                    "file": data_path.name,
                    "persona": "mixed",
                    "sourceRows": len(entries),
                    "usableExamples": len(entries),
                    "format": "JSON / metin pasajı",
                }
            ]
            return entries, stats

        source_dir = data_path if data_path.is_dir() else data_path / "source"
        entries: list[dict[str, Any]] = []
        stats: list[dict[str, Any]] = []

        for spec in DATASET_SPECS:
            csv_path = source_dir / spec["file"]
            if not csv_path.exists():
                continue

            source_rows = 0
            usable_rows = 0
            with csv_path.open(newline="", encoding="utf-8-sig") as file:
                reader = csv.DictReader(file)
                for row_index, row in enumerate(reader, start=1):
                    source_rows += 1
                    text = _clean_spaces(row.get("metin_pasaji", ""))

                    # Çok kısa satırlar genellikle başlık olduğu için modele dahil edilmez.
                    if len(text) < 40:
                        continue
                    usable_rows += 1
                    author = _clean_spaces(row.get("yazar_adi", "Bilinmeyen"))
                    entries.append(
                        {
                            "id": f"{spec['persona']}-{usable_rows:04d}",
                            "persona": spec["persona"],
                            "title": f"{author} pasajı",
                            "author": author,
                            "text": text,
                            "source": row.get("kaynak_link", ""),
                            "period": row.get("donem", ""),
                            "sourceFile": spec["file"],
                        }
                    )

            stats.append(
                {
                    "name": spec["name"],
                    "file": spec["file"],
                    "persona": spec["persona"],
                    "sourceRows": source_rows,
                    "usableExamples": usable_rows,
                    "format": spec["format"],
                }
            )

        if not entries:
            raise FileNotFoundError("CSV veri setleri data/source klasöründe bulunamadı.")
        return entries, stats

    def _split_train_test(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        # Veri setini sınıf bazında yaklaşık %80 eğitim, %20 test olacak şekilde ayırır.
        # index % 5 == 0 olanlar test, diğerleri eğitim verisi yapılır.
        train: list[dict[str, Any]] = []
        test: list[dict[str, Any]] = []
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in self.corpus:
            grouped[entry["persona"]].append(entry)

        for entries in grouped.values():
            for index, entry in enumerate(entries):
                if index % 5 == 0:
                    test.append(entry)
                else:
                    train.append(entry)
        return train, test

    def _train_classifier(self) -> dict[str, Any]:
        # MODELİN EĞİTİLDİĞİ ASIL BÖLÜM BURASIDIR.
        # Multinomial Naive Bayes için her sınıfta geçen kelimelerin frekansı sayılır.
        # classDocCounts: Her sınıfta kaç eğitim metni var?
        # tokenCounts: Her sınıfta hangi kelime kaç kez geçti?
        # tokenTotals: Her sınıftaki toplam kelime sayısı.
        # vocabulary: Eğitimde görülen tüm farklı kelimeler.
        class_doc_counts = Counter(entry["persona"] for entry in self.train_set)
        token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        token_totals: Counter[str] = Counter()
        vocabulary: set[str] = set()

        for entry in self.train_set:
            persona = entry["persona"]
            tokens = tokenize(entry["text"])

            # Bu satırlar Naive Bayes modelinin "öğrenme" kısmıdır:
            # kelimeler sınıflara göre sayılır.
            token_counts[persona].update(tokens)
            token_totals[persona] += len(tokens)
            vocabulary.update(tokens)

        return {
            "classDocCounts": class_doc_counts,
            "tokenCounts": token_counts,
            "tokenTotals": token_totals,
            "vocabulary": vocabulary,
            "totalDocs": len(self.train_set),
        }

    def _class_scores(self, text: str) -> dict[str, float]:
        # MODELİN TAHMİN HESABI BURADADIR.
        # Kullanıcının metnindeki her kelime için:
        # P(sınıf) ve P(kelime | sınıf) olasılıkları log olarak toplanır.
        # En yüksek skorlu sınıf modelin tahmini olur.
        tokens = tokenize(text)
        vocabulary = self._classifier["vocabulary"]
        vocab_size = max(1, len(vocabulary))
        total_docs = max(1, self._classifier["totalDocs"])
        scores: dict[str, float] = {}

        for persona in PERSONAS:
            class_docs = self._classifier["classDocCounts"].get(persona, 0)

            # Prior: Sınıfın eğitim verisinde görülme olasılığı.
            prior = (class_docs + 1) / (total_docs + len(PERSONAS))
            score = math.log(prior)
            token_total = self._classifier["tokenTotals"].get(persona, 0)
            denominator = token_total + vocab_size
            class_counts = self._classifier["tokenCounts"].get(persona, Counter())
            for token in tokens:
                # Laplace smoothing: Eğitimde hiç görülmeyen kelimeler sıfır olasılık yapmasın diye +1 eklenir.
                score += math.log((class_counts.get(token, 0) + 1) / denominator)
            scores[persona] = score
        return scores

    def _evaluate_classifier(self) -> dict[str, Any]:
        # Test verisi üzerinde model başarısını ölçen bölüm.
        # Accuracy, precision, recall, F1-score ve confusion matrix burada hesaplanır.
        classes = list(PERSONAS.keys())
        confusion: dict[str, Counter[str]] = {key: Counter() for key in classes}
        for entry in self.test_set:
            # Test verisindeki her metin için model tahmini alınır.
            predicted = self.predict_persona(entry["text"])["key"]
            confusion[entry["persona"]][predicted] += 1

        total = len(self.test_set)
        correct = sum(confusion[key][key] for key in classes)
        per_class = []
        macro_f1_values = []

        for key in classes:
            # TP, FP, FN değerlerinden precision, recall ve F1 hesaplanır.
            tp = confusion[key][key]
            fp = sum(confusion[actual][key] for actual in classes if actual != key)
            fn = sum(confusion[key][predicted] for predicted in classes if predicted != key)
            support = sum(confusion[key].values())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            macro_f1_values.append(f1)
            per_class.append(
                {
                    "key": key,
                    "label": PERSONAS[key].label,
                    "precision": round(precision * 100, 1),
                    "recall": round(recall * 100, 1),
                    "f1": round(f1 * 100, 1),
                    "support": support,
                }
            )

        return {
            "trainSize": len(self.train_set),
            "testSize": len(self.test_set),
            "accuracy": round((correct / total) * 100, 1) if total else 0.0,
            "macroF1": round((sum(macro_f1_values) / len(macro_f1_values)) * 100, 1)
            if macro_f1_values
            else 0.0,
            "perClass": per_class,
            "confusionMatrix": [
                {
                    "actual": PERSONAS[actual].label,
                    "predicted": {PERSONAS[predicted].label: confusion[actual][predicted] for predicted in classes},
                }
                for actual in classes
            ],
        }

    def _build_idf(self) -> dict[str, float]:
        # TF-IDF içindeki IDF kısmını hesaplar.
        # Nadir geçen kelimelere daha yüksek ağırlık verir.
        # Bu değerler benzer pasaj bulma aşamasında kullanılır.
        documents = [set(tokenize(entry["text"])) for entry in self.corpus]
        doc_count = max(1, len(documents))
        idf: dict[str, float] = {}
        for document in documents:
            for token in document:
                idf[token] = idf.get(token, 0.0) + 1.0
        return {token: math.log((doc_count + 1) / (count + 1)) + 1 for token, count in idf.items()}

    def _vectorize(self, text: str) -> dict[str, float]:
        # Bir metni TF-IDF vektörüne çevirir.
        # Bu vektörler kosinüs benzerliği için kullanılır.
        tokens = tokenize(text)
        if not tokens:
            return {}
        vector: dict[str, float] = {}
        for token in tokens:
            vector[token] = vector.get(token, 0.0) + 1.0
        for token, count in list(vector.items()):
            vector[token] = count * self._idf.get(token, 1.0)
        return vector

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        # İki TF-IDF vektörü arasındaki kosinüs benzerliğini hesaplar.
        # Sonuç 1'e yaklaştıkça metinler daha benzerdir.
        if not left or not right:
            return 0.0
        shared = set(left) & set(right)
        numerator = sum(left[token] * right[token] for token in shared)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)
