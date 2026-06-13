from pathlib import Path
import unittest

from src.style_engine import LLMResult, StyleEngine, tokenize


# Test dosyasının amacı:
# Projenin ana RAG + LLM yapay zeka motoru bozulmadan çalışıyor mu diye kontrol etmektir.
# Öğretmen "projenizi test ettiniz mi?" derse bu dosyayı gösterebilirsin.

ROOT = Path(__file__).resolve().parents[1]


class FakeGeminiClient:
    # Testler gerçek Gemini API'ye gitmeden transform akışını doğrulasın diye
    # sabit bir Gemini benzeri cevap döndürür.
    provider = "gemini"

    def generate(self, source_text, profile, retrieved_examples, intensity):
        return LLMResult(
            text="Efendim, bugün vakit içinde dostane bir kelam ederim.",
            provider="gemini",
            model="gemini-2.5-flash-test",
            used_llm=True,
            prompt="test prompt",
        )


class StyleEngineTest(unittest.TestCase):
    # Her testten önce StyleEngine yeniden oluşturulur.
    # Bu sırada CSV veri seti okunur, RAG değerleri hazırlanır,
    # eğitim/test ayrımı yapılır ve Naive Bayes değerlendirme modeli eğitilir.
    def setUp(self) -> None:
        self.engine = StyleEngine(ROOT / "data" / "source")
        self.engine.llm_client = FakeGeminiClient()

    def test_tokenize_keeps_turkish_letters(self) -> None:
        # Türkçe karakterlerin tokenization sırasında bozulmadığını kontrol eder.
        # Örneğin "Nasılsın" kelimesindeki ı harfi korunmalıdır.
        tokens = tokenize("Nasılsın, bugün mektebe gidiyor musun?")
        self.assertIn("nasılsın", tokens)
        self.assertIn("bugün", tokens)
        self.assertIn("mektebe", tokens)

    def test_transform_returns_output_and_glossary(self) -> None:
        # Dönüştürme fonksiyonunun çıktı, RAG örnekleri, LLM bilgisi ve sözlükçe ürettiğini kontrol eder.
        result = self.engine.transform(
            "Nasılsın, neler yapıyorsun?",
            "istanbul_beyefendisi",
            0.65,
        )
        self.assertIn("output", result)
        self.assertIn("llm", result)
        self.assertIn("usedLLM", result["llm"])
        self.assertTrue(result["llm"]["usedLLM"])
        self.assertGreater(len(result["output"]), 10)
        self.assertGreaterEqual(len(result["retrievedExamples"]), 1)
        self.assertGreaterEqual(len(result["glossary"]), 1)

    def test_modern_technology_warning(self) -> None:
        # "İnternet" gibi eski dönemlerde olmayan modern kelimeler için uyarı üretildiğini kontrol eder.
        result = self.engine.transform(
            "İnternette güzel bir yazı buldum.",
            "istanbul_beyefendisi",
            0.65,
        )
        self.assertTrue(result["warnings"])

    def test_all_personas_are_available(self) -> None:
        # Arayüzde ve modelde beklenen iki sınıf/persona var mı diye bakar.
        personas = self.engine.personas()
        keys = {persona["key"] for persona in personas}
        self.assertEqual(keys, {"istanbul_beyefendisi", "anadolu_ozani"})

    def test_dataset_summary_and_training_report_exist(self) -> None:
        # Veri seti özeti ve değerlendirme modeli başarı raporu gerçekten üretiliyor mu diye kontrol eder.
        # Accuracy'nin 70'in üstünde olması modelin rastgele çalışmadığını gösteren basit bir eşiğimizdir.
        summary = self.engine.dataset_summary()
        report = self.engine.model_report()["metrics"]
        self.assertGreater(summary["usableExamples"], 1000)
        self.assertGreater(report["trainSize"], 1000)
        self.assertGreaterEqual(report["accuracy"], 70)


if __name__ == "__main__":
    # Dosya doğrudan çalıştırılırsa testleri başlatır.
    unittest.main()
