from __future__ import annotations

# Bu dosya yerel web sunucusudur.
# Python'un standart http.server modülüyle çalışır.
# Hem public klasöründeki HTML/CSS/JS dosyalarını tarayıcıya verir,
# hem de /api/personas, /api/dataset ve /api/transform API uçlarını yönetir.

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .style_engine import StyleEngine


# Proje klasörlerinin merkezi tanımları.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = PROJECT_ROOT / "public"
DATA_PATH = PROJECT_ROOT / "data" / "source"


class AppHandler(BaseHTTPRequestHandler):
    # StyleEngine uygulama başlarken bir kere oluşturulur.
    # Bu satır çalışınca veri seti okunur, eğitim/test ayrımı yapılır,
    # RAG indeksi hazırlanır, Naive Bayes değerlendirme modeli eğitilir
    # ve LLM istemcisi ayarlanır.
    engine = StyleEngine(DATA_PATH)

    def do_OPTIONS(self) -> None:
        # Tarayıcıdan gelen CORS ön kontrol isteklerine basit cevap verir.
        self._send_json({"ok": True})

    def do_GET(self) -> None:
        # GET istekleri iki amaçla kullanılır:
        # 1. API bilgisi döndürmek
        # 2. HTML/CSS/JS gibi statik dosyaları servis etmek
        path = urlparse(self.path).path
        if path == "/api/personas":
            # Arayüzdeki persona seçeneklerini döndürür.
            self._send_json({"personas": self.engine.personas()})
            return
        if path == "/api/dataset":
            # Veri seti özeti, RAG + LLM bilgisi ve değerlendirme metriklerini döndürür.
            self._send_json(
                {
                    "dataset": self.engine.dataset_summary(),
                    "model": self.engine.model_report(),
                }
            )
            return

        self._serve_static(path)

    def do_POST(self) -> None:
        # POST istekleri kullanıcıdan gelen metni dönüştürmek için kullanılır.
        path = urlparse(self.path).path
        if path != "/api/transform":
            self._send_json({"error": "Bilinmeyen API yolu."}, HTTPStatus.NOT_FOUND)
            return

        try:
            # Tarayıcıdan gelen JSON gövdesi okunur.
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))

            # StyleEngine.transform çağrılarak RAG örnekleri, LLM çıktısı,
            # Naive Bayes değerlendirmesi, sözlükçe ve uyarılar hazırlanır.
            result = self.engine.transform(
                text=str(payload.get("text", "")),
                persona=str(payload.get("persona", "istanbul_beyefendisi")),
                intensity=float(payload.get("intensity", 0.65)),
            )
            self._send_json(result)
        except ValueError as exc:
            # Boş metin veya bilinmeyen persona gibi kullanıcı hataları.
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except json.JSONDecodeError:
            # Geçersiz JSON gelirse kullanıcıya hata döndürülür.
            self._send_json({"error": "Geçersiz JSON gönderildi."}, HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt: str, *args: object) -> None:
        # Terminalde kısa sunucu logu gösterir.
        print(f"[server] {self.address_string()} - {fmt % args}")

    def _serve_static(self, path: str) -> None:
        # public klasöründeki dosyaları tarayıcıya gönderir.
        # "/" gelirse ana sayfa olan index.html açılır.
        if path == "/":
            target = PUBLIC_DIR / "index.html"
        else:
            relative = unquote(path).lstrip("/")
            target = (PUBLIC_DIR / relative).resolve()

        try:
            # Güvenlik kontrolü: Kullanıcı public klasörü dışındaki dosyalara erişemesin.
            target.relative_to(PUBLIC_DIR.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return

        if not target.exists() or target.is_dir():
            # Bulunamayan yollar ana sayfaya düşer.
            target = PUBLIC_DIR / "index.html"

        # Dosya içeriği ve içerik tipi hazırlanıp HTTP cevabı olarak gönderilir.
        content = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        # API cevaplarını JSON formatında döndüren yardımcı fonksiyon.
        # ensure_ascii=False Türkçe karakterlerin bozulmadan gitmesini sağlar.
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    # Terminalden verilen --host ve --port ayarlarını okur.
    # Örnek: python3 -m src.server --port 8001
    parser = argparse.ArgumentParser(description="Kuşaklararası Köprü yerel web sunucusu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    # ThreadingHTTPServer aynı anda birden fazla isteğe cevap verebilir.
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Kuşaklararası Köprü çalışıyor: http://{args.host}:{args.port}")
    print("Durdurmak için Ctrl+C")
    server.serve_forever()


if __name__ == "__main__":
    # Dosya doğrudan çalıştırıldığında sunucuyu başlatır.
    main()
