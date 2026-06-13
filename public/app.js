// Bu dosya web arayüzünün davranışlarını yönetir.
// Kullanıcı metin girince API'ye istek gönderir, gelen sonucu ekrana basar,
// sekmeleri çalıştırır ve veri seti/model bilgilerini gösterir.

// Sayfa genelinde kullanılacak küçük durum bilgisi.
const state = {
  personas: [],
  selectedPersona: "istanbul_beyefendisi",
  lastResult: null,
};

// HTML'deki önemli alanlar JavaScript değişkenlerine bağlanır.
// Böylece bu alanların yazısı, içeriği veya tıklanma davranışı değiştirilebilir.
const form = document.querySelector("#transformForm");
const personaGrid = document.querySelector("#personaGrid");
const sourceText = document.querySelector("#sourceText");
const intensity = document.querySelector("#intensity");
const intensityValue = document.querySelector("#intensityValue");
const statusPill = document.querySelector("#statusPill");
const outputText = document.querySelector("#outputText");
const personaTitle = document.querySelector("#personaTitle");
const warningList = document.querySelector("#warningList");
const examplesPanel = document.querySelector("#examplesPanel");
const glossaryPanel = document.querySelector("#glossaryPanel");
const metricsPanel = document.querySelector("#metricsPanel");
const copyButton = document.querySelector("#copyButton");
const datasetList = document.querySelector("#datasetList");

// "Model ve Algoritma Cevapları" bölümünde gösterilen kısa açıklamalar.
// Öğretmenin sorabileceği temel soruların cevapları arayüzde buradan görünür.
const methodAnswers = [
  {
    question: "Hangi algoritma veya model kullanıldı?",
    answer:
      "Metin üretimi için RAG + Gemini 2.5 Flash yaklaşımı kullanıldı. Önce TF-IDF ve kosinüs benzerliğiyle veri setinden hedef üsluba yakın örnekler getiriliyor, sonra bu örnekler Gemini promptuna eklenerek çıktı üretiliyor. Naive Bayes modeli ise üretilen çıktının hangi üsluba benzediğini ölçüyor.",
  },
  {
    question: "Neden bu model seçildi?",
    answer:
      "Metin dönüştürme sadece sınıflandırma ile iyi çözülemediği için Gemini kullanıldı. RAG eklenmesinin sebebi, modelin kendi başına rastgele bir üslup uydurması yerine projedeki gerçek CSV örneklerinden yararlanmasını sağlamak.",
  },
  {
    question: "Modelin temel çalışma mantığı nedir?",
    answer:
      "Kullanıcı metni temizlenir, seçilen personaya ait CSV pasajlarıyla TF-IDF vektörleri üzerinden karşılaştırılır ve en benzer örnekler seçilir. Bu örnekler prompt içine konur, Gemini metni hedef üslupta yeniden yazar, Naive Bayes de çıktıyı değerlendirir.",
  },
  {
    question: "Hazır model mi, sıfırdan mı?",
    answer:
      "Gemini 2.5 Flash hazır bir dil modeli olarak kullanılır; bu projede fine-tuning yapılmaz. RAG getirme, prompt hazırlama, TF-IDF/kosinüs benzerliği ve Naive Bayes değerlendirme kodları proje içinde yazılmıştır.",
  },
  {
    question: "Kullanılan kütüphaneler nelerdir?",
    answer:
      "Python 3 standart kütüphaneleri kullanıldı: http.server, urllib, json, csv, re, math, pathlib, dataclasses. Ön yüzde HTML, CSS ve Vanilla JavaScript var. LLM sağlayıcısı olarak yalnızca Google Gemini API kullanıldı.",
  },
];

function setStatus(text) {
  // Sağ üstteki küçük durum etiketini günceller.
  // Örnek: Hazır, Çalışıyor, Tamamlandı, Hata.
  statusPill.textContent = text;
}

function formatIntensity(value) {
  // Slider değerini kullanıcıya daha anlaşılır bir metin olarak gösterir.
  const numeric = Number(value);
  if (numeric < 0.45) return "Hafif";
  if (numeric < 0.75) return "Orta";
  return "Güçlü";
}

function escapeHtml(value) {
  // Veri setinden gelen metinler doğrudan HTML'e basılırsa güvenlik sorunu olabilir.
  // Bu fonksiyon özel HTML karakterlerini kaçırarak güvenli basım sağlar.
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderPersonas(personas) {
  // API'den gelen persona listesini arayüzde radyo kartları olarak gösterir.
  personaGrid.innerHTML = personas
    .map(
      (persona, index) => `
        <label class="persona-option">
          <input
            type="radio"
            name="persona"
            value="${persona.key}"
            ${index === 0 ? "checked" : ""}
          />
          <span class="persona-card">
            <strong>${escapeHtml(persona.label)}</strong>
            <span>${escapeHtml(persona.description)}</span>
          </span>
        </label>
      `,
    )
    .join("");

  personaGrid.querySelectorAll("input").forEach((input) => {
    input.addEventListener("change", () => {
      // Kullanıcı persona değiştirince seçili persona state içine yazılır.
      state.selectedPersona = input.value;
    });
  });
}

function renderWarnings(warnings) {
  // Modern teknoloji kelimesi veya LLM bağlantı durumu gibi uyarıları arayüzde gösterir.
  warningList.innerHTML = warnings
    .map((warning) => `<div class="warning-item">${escapeHtml(warning)}</div>`)
    .join("");
}

function renderExamples(examples) {
  // RAG örnek getirme sonuçlarını ekrana basar.
  // Bu örnekler veri setinden kosinüs benzerliği ile seçilir.
  if (!examples.length) {
    examplesPanel.innerHTML = `<div class="empty-state">Bu persona için örnek bulunamadı.</div>`;
    return;
  }

  examplesPanel.innerHTML = examples
    .map(
      (example) => `
        <article class="detail-item">
          <header>
            <strong>${escapeHtml(example.title)}</strong>
            <small>Benzerlik: ${Math.round(example.score * 100)}%</small>
          </header>
          <p>${escapeHtml(example.text)}</p>
        </article>
      `,
    )
    .join("");
}

function renderGlossary(items) {
  // Çıktıda geçen eski/edebi kelimelerin açıklamalarını "Sözlükçe" sekmesine yazar.
  if (!items.length) {
    glossaryPanel.innerHTML = `<div class="empty-state">Çıktıda açıklanacak eski kelime bulunamadı.</div>`;
    return;
  }

  glossaryPanel.innerHTML = items
    .map(
      (item) => `
        <article class="detail-item">
          <header>
            <strong>${escapeHtml(item.term)}</strong>
          </header>
          <p>${escapeHtml(item.meaning)}</p>
        </article>
      `,
    )
    .join("");
}

function renderMetrics(metrics) {
  // Ölçüm sekmesini hazırlar.
  // Hem dönüşüm kalitesi puanları hem de LLM/Naive Bayes/Test kartları burada gösterilir.
  const modelCards = state.lastResult
    ? [
        {
          label: "Üretim Modu",
          score: state.lastResult.llm.usedLLM ? 100 : 45,
          note: state.lastResult.llm.usedLLM
            ? `Çıktı ${state.lastResult.llm.provider} sağlayıcısındaki ${state.lastResult.llm.model} modeliyle üretildi.`
            : "Gemini çağrısı kullanılmadığı için çıktı üretilemedi.",
        },
        {
          label: "Çıktı Persona Tahmini",
          score: Math.round(state.lastResult.predictedPersona.confidence * 100),
          note: `Naive Bayes modeline göre üretilen çıktı en çok "${state.lastResult.predictedPersona.label}" sınıfına benziyor.`,
        },
        {
          label: "Test Accuracy",
          score: state.lastResult.trainingReport.accuracy,
          note: `Eğitim/test ayrımı: ${state.lastResult.trainingReport.trainSize} eğitim, ${state.lastResult.trainingReport.testSize} test örneği. Macro F1: ${state.lastResult.trainingReport.macroF1}.`,
        },
      ]
    : [];

  metricsPanel.innerHTML = [...Object.values(metrics), ...modelCards]
    .map(
      (metric) => `
        <article class="detail-item">
          <header>
            <strong>${escapeHtml(metric.label)}</strong>
            <small>${metric.score}/100</small>
          </header>
          <p>${escapeHtml(metric.note)}</p>
          <div class="metric-bar" aria-hidden="true">
            <span style="width: ${metric.score}%"></span>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderResult(result) {
  // /api/transform cevabı geldikten sonra tüm sonuç panellerini günceller.
  state.lastResult = result;
  outputText.textContent = result.output;
  personaTitle.textContent = result.persona.label;
  renderWarnings(result.warnings);
  renderExamples(result.retrievedExamples);
  renderGlossary(result.glossary);
  renderMetrics(result.metrics);
}

async function transformText() {
  // Kullanıcı "Dönüştür" butonuna bastığında çalışan ana ön yüz fonksiyonu.
  // Metni, seçilen personayı ve yoğunluk değerini API'ye gönderir.
  setStatus("Çalışıyor");
  const payload = {
    text: sourceText.value,
    persona: state.selectedPersona,
    intensity: Number(intensity.value),
  };

  const response = await fetch("/api/transform", {
    // Backend tarafında src/server.py içindeki do_POST bu isteği karşılar.
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();

  if (!response.ok) {
    // API hata döndürürse kullanıcıya hata metni gösterilir.
    throw new Error(result.error || "Dönüştürme başarısız oldu.");
  }

  // API başarılıysa sonuç ekrana işlenir.
  renderResult(result);
  setStatus("Tamamlandı");
}

function setupTabs() {
  // RAG örnekleri, Sözlükçe ve Ölçüm sekmeleri arasında geçişi sağlar.
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab-button").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      document.querySelector(`#${button.dataset.tab}Panel`).classList.add("active");
    });
  });
}

function setupSamples() {
  // Okul, Teknoloji, Akşam gibi hazır örnek butonlarını çalıştırır.
  document.querySelectorAll("[data-sample]").forEach((button) => {
    button.addEventListener("click", () => {
      sourceText.value = button.dataset.sample;
      sourceText.focus();
    });
  });
}

function setupMethodology() {
  // Model ve algoritma cevaplarını arayüzdeki teknik açıklama bölümüne basar.
  const list = document.querySelector("#methodList");
  list.innerHTML = methodAnswers
    .map(
      (item) => `
        <article class="method-item">
          <h3>${escapeHtml(item.question)}</h3>
          <p>${escapeHtml(item.answer)}</p>
        </article>
      `,
    )
    .join("");
}

function renderDataset(payload) {
  // /api/dataset cevabını kullanarak veri seti ve eğitim sonucu kartlarını oluşturur.
  const dataset = payload.dataset;
  const model = payload.model;
  const fileCards = dataset.files
    .map(
      (file) => `
        <article class="method-item">
          <h3>${escapeHtml(file.name)}</h3>
          <p>${escapeHtml(file.file)}</p>
          <p><strong>${file.sourceRows}</strong> CSV satırı, <strong>${file.usableExamples}</strong> kullanılabilir metin örneği.</p>
        </article>
      `,
    )
    .join("");

  const classCards = Object.values(dataset.byPersona)
    .map(
      (item) => `
        <article class="method-item">
          <h3>${escapeHtml(item.label)}</h3>
          <p><strong>${item.examples}</strong> örnek: ${item.trainExamples} eğitim, ${item.testExamples} test.</p>
          <p>Öne çıkan yazarlar: ${escapeHtml(item.topAuthors.map((author) => `${author.name} (${author.count})`).join(", "))}</p>
        </article>
      `,
    )
    .join("");

  const metricCards = `
    <!-- Modelin eğitim/test sonucu arayüzde özellikle bu kartta gösterilir. -->
    <article class="method-item highlight-item">
      <h3>Eğitim ve Test Sonucu</h3>
      <p>Algoritma: ${escapeHtml(model.algorithm)}</p>
      <p>LLM sağlayıcısı: <strong>${escapeHtml(model.llmProvider)}</strong></p>
      <p>Accuracy: <strong>${model.metrics.accuracy}</strong> / Macro F1: <strong>${model.metrics.macroF1}</strong></p>
      <p>${escapeHtml(model.epochs)}</p>
    </article>
    <article class="method-item highlight-item">
      <h3>Toplam Veri</h3>
      <p><strong>${dataset.totalRowsInCsv}</strong> CSV satırı, <strong>${dataset.usableExamples}</strong> kullanılabilir metin örneği.</p>
      <p>${escapeHtml(dataset.split)}</p>
    </article>
  `;

  datasetList.innerHTML = `${metricCards}${fileCards}${classCards}`;
}

async function bootstrap() {
  // Sayfa ilk açıldığında çalışan başlangıç fonksiyonu.
  // Sekmeleri, örnek butonları, model açıklamalarını ve ilk dönüşümü hazırlar.
  setupTabs();
  setupSamples();
  setupMethodology();
  intensity.addEventListener("input", () => {
    intensityValue.textContent = formatIntensity(intensity.value);
  });
  intensityValue.textContent = formatIntensity(intensity.value);

  copyButton.addEventListener("click", async () => {
    // Kopyala butonu çıktı metnini panoya alır.
    await navigator.clipboard.writeText(outputText.textContent);
    setStatus("Kopyalandı");
  });

  form.addEventListener("submit", async (event) => {
    // Form gönderilince sayfanın yenilenmesini engeller ve API dönüşümünü çalıştırır.
    event.preventDefault();
    try {
      await transformText();
    } catch (error) {
      setStatus("Hata");
      outputText.textContent = error.message;
    }
  });

  // Persona seçeneklerini backend API'den alır.
  const response = await fetch("/api/personas");
  const payload = await response.json();
  state.personas = payload.personas;
  renderPersonas(payload.personas);

  // Veri seti özeti ve model metriklerini backend API'den alır.
  const datasetResponse = await fetch("/api/dataset");
  renderDataset(await datasetResponse.json());

  // Sayfa açıldığında örnek metin için ilk dönüşümü otomatik yapar.
  await transformText();
}

// Uygulama başlatılır.
bootstrap();
