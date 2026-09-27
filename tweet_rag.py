# -*- coding: utf-8 -*-
# ============================================================================
# tweet_rag.py — Kişisel Tweet Arşivi RAG Sistemi
# ============================================================================
#
# Kurulum:
#   pip install -r requirements.txt
#
# Kullanım:
#   Windows (PowerShell):
#     $env:GOOGLE_API_KEY="BURAYA_API_ANAHTARINIZI_YAZIN"
#     python tweet_rag.py
#
#   Linux / macOS:
#     export GOOGLE_API_KEY="BURAYA_API_ANAHTARINIZI_YAZIN"
#     python tweet_rag.py
#
# Not: tweets.js dosyası bu script ile aynı dizinde olmalıdır.
#      Farklı yoldaysa TWEETS_FILE değişkenini güncelleyin.
#
# Güncel model adını kontrol edin:
#   https://ai.google.dev/gemini-api/docs/models
# ============================================================================

"""
Kişisel Tweet Arşivi RAG (Retrieval-Augmented Generation) Sistemi.

Twitter/X veri arşivindeki (tweets.js) tweet'leri parse edip,
sentence-transformers ile vektörleştirerek semantik arama yapar
ve Google Gemini LLM ile doğal dilde cevaplar üretir.

Modüller:
    1. Parse & Temizleme   — tweets.js dosyasını okuyup tweet'leri çıkarır.
    2. Gruplama (Chunking) — Zaman damgasına göre thread'leri birleştirir.
    3. Embedding & İndeks  — Vektör deposu oluşturur / diskten yükler.
    4. Sorgu & LLM         — Kosinüs benzerliği ile arama + Gemini cevabı.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

# tweets.js dosya yolu — script ile aynı dizin varsayılır.
SCRIPT_DIR = Path(__file__).resolve().parent
TWEETS_FILE = SCRIPT_DIR / "tweets.js"

# İndeks (kalıcılık) dosyaları
INDEX_FILE = SCRIPT_DIR / "tweet_index.npy"
CHUNKS_FILE = SCRIPT_DIR / "tweet_chunks.json"

# Embedding modeli
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64  # RAM dostu batch boyutu

# Gruplama eşiği: ardışık tweet'ler arası maksimum süre (dakika)
GROUP_GAP_MINUTES = 10

# Grup metin uzunluğu uyarı eşiği
GROUP_CHAR_WARN = 4000

# Retrieval parametreleri
DEFAULT_TOP_K = 5
MAX_CONTEXT_CHARS = 30_000  # LLM'e gönderilecek maksimum bağlam uzunluğu

# LLM ayarları
LLM_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
LLM_MODEL = "gemini-3.8-flash"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1024

# LLM retry ayarları
LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF = [1.0, 3.0, 7.0]  # Her deneme için bekleme (saniye)
LLM_RETRYABLE_CODES = {429, 500, 502, 503}

# Sistem prompt'u — LLM'e gönderilecek talimat
SYSTEM_PROMPT = (
    "Sen kişisel tweet arşivi asistanısın. Sana verilen TWEET METİNLERİNE "
    "SADECE dayanarak cevap ver. Kurallar:\n"
    "- Tweet'lerde olmayan bilgi ekleme, tahmin yapma, yorum yapma, genel "
    "bilgiden cevap üretme.\n"
    "- Tweet'lerde cevap yoksa birebir şunu söyle: \"Arşivinizde bu konuda "
    "tweet bulamadım.\"\n"
    "- Cevabın sonunda hangi tarihteki tweet'lere dayandığını parantez içinde "
    "belirt.\n"
    "- Tweet'ler kronolojik olarak gruplandırılmıştır; grup başlangıç tarihini "
    "kaynak olarak kullan."
)

# Twitter tarih formatı
TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"


# ============================================================================
# 1. PARSE & TEMİZLEME
# ============================================================================


def load_tweets_js(filepath: Path) -> list[dict]:
    """
    tweets.js dosyasını okuyup JSON dizisine dönüştürür.

    Twitter arşiv dosyası 'window.YTD.tweets.partN = [...]' biçiminde
    başlayabilir. Bu prefix temizlenip saf JSON olarak parse edilir.

    Args:
        filepath: tweets.js dosyasının yolu.

    Returns:
        Tweet nesnelerinin listesi (ham JSON dict'ler).

    Raises:
        FileNotFoundError: Dosya bulunamazsa.
        json.JSONDecodeError: JSON parse edilemezse.
    """
    if not filepath.exists():
        print(f"[HATA] Dosya bulunamadı: {filepath}")
        sys.exit(1)

    print(f"[BİLGİ] tweets.js okunuyor: {filepath}")
    with open(filepath, encoding="utf-8") as f:
        raw = f.read()

    # Twitter arşiv prefix'ini temizle: "window.YTD.tweets.partN = "
    # Bu satır, dosyanın başındaki JavaScript değişken atamasını kaldırır.
    raw = re.sub(r"^window\.YTD\.tweets\.\w+\s*=\s*", "", raw, count=1)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[HATA] tweets.js JSON olarak parse edilemedi: {exc}")
        sys.exit(1)

    if not isinstance(data, list):
        print("[HATA] tweets.js içeriği bir JSON dizisi (array) değil.")
        sys.exit(1)

    print(f"[BİLGİ] {len(data):,} ham kayıt okundu.")
    return data


def parse_and_clean(raw_items: list[dict]) -> list[dict]:
    """
    Ham tweet kayıtlarını parse eder, temizler ve kronolojik sıralar.

    İşlemler:
        - 'tweet' sarmalayıcısı olan/olmayan kayıtları tolere eder.
        - full_text yoksa text alanına düşer; ikisi de yoksa atlar.
        - "RT @" ile başlayan retweet'leri atar.
        - created_at tarihini parse eder; parse edilemeyeni atlar.
        - id_str'yi int'e çevirip ona göre kronolojik sıralar.

    Args:
        raw_items: load_tweets_js'den dönen ham kayıt listesi.

    Returns:
        Sıralanmış, temizlenmiş tweet dict'lerinin listesi.
        Her dict: {
            "text": str,
            "date_str": "YYYY-MM-DD",
            "datetime": datetime,
            "id_int": int,
            "id_str": str,
        }
    """
    tweets: list[dict] = []
    skipped = 0
    retweets = 0

    for item in raw_items:
        # "tweet" sarmalayıcısı olan/olmayan kayıtları tolere et
        tweet_obj = item.get("tweet", item)

        # Tweet metni: önce full_text, yoksa text
        text = tweet_obj.get("full_text") or tweet_obj.get("text")
        if not text:
            skipped += 1
            continue

        # Retweet'leri atla
        if text.startswith("RT @"):
            retweets += 1
            continue

        # Tarih parse
        created_at_str = tweet_obj.get("created_at", "")
        try:
            dt = datetime.strptime(created_at_str, TWITTER_DATE_FORMAT)
        except (ValueError, TypeError):
            skipped += 1
            continue

        # ID parse
        id_str = tweet_obj.get("id_str", "")
        try:
            id_int = int(id_str)
        except (ValueError, TypeError):
            skipped += 1
            continue

        tweets.append(
            {
                "text": text.strip(),
                "date_str": dt.strftime("%Y-%m-%d"),
                "datetime": dt,
                "id_int": id_int,
                "id_str": id_str,
            }
        )

    # id_str'nin int değerine göre kronolojik sırala
    # (string sıralama farklı basamak sayılarında yanlış sonuç verir)
    tweets.sort(key=lambda t: t["id_int"])

    print(f"[BİLGİ] {len(tweets):,} tweet parse edildi.")
    if retweets:
        print(f"[BİLGİ] {retweets:,} retweet atlandı.")
    if skipped:
        print(f"[UYARI] {skipped:,} kayıt parse edilemedi / metin bulunamadı, atlandı.")

    return tweets


# ============================================================================
# 2. BAĞLAMA GÖRE GRUPLAMA (CHUNKING)
# ============================================================================


def group_tweets(tweets: list[dict]) -> list[dict]:
    """
    Kronolojik sıralı tweet'leri zaman damgasına göre gruplar.

    Kural: Ardışık iki tweet'in created_at zaman damgaları arasındaki
    fark 10 dakikadan azsa aynı grupta kalır. Fark 10 dakika veya
    daha fazlaysa yeni bir grup başlatılır. Böylece thread'ler
    doğal olarak birleşir.

    ÖNEMLİ: Zaman farkı, parse edilmiş datetime değerlerinden hesaplanır.
    Twitter Snowflake ID formatında (timestamp << 22 | ...) ID farkı
    gerçek zaman farkıyla doğrudan orantılı olmadığından, ID farkına
    dayalı gruplama YANLIŞ sonuç verir.

    Args:
        tweets: parse_and_clean'den dönen sıralı tweet listesi.

    Returns:
        Grup dict'lerinin listesi. Her dict:
        {
            "text": gruptaki tweet'ler "\\n" ile birleşmiş,
            "source": "tweet YYYY-MM-DD (id: ...)",
            "date": "YYYY-MM-DD",
        }
    """
    if not tweets:
        return []

    gap = timedelta(minutes=GROUP_GAP_MINUTES)
    groups: list[list[dict]] = []
    current_group: list[dict] = [tweets[0]]

    for prev, curr in zip(tweets, tweets[1:]):
        # Zaman farkını datetime nesnelerinden hesapla (ID farkından DEĞİL)
        time_diff = curr["datetime"] - prev["datetime"]
        if abs(time_diff) < gap:
            current_group.append(curr)
        else:
            groups.append(current_group)
            current_group = [curr]

    # Son grubu ekle
    groups.append(current_group)

    # Grup dict'lerine dönüştür
    chunks: list[dict] = []
    for group in groups:
        combined_text = "\n".join(t["text"] for t in group)

        # 4000 karakter uyarısı
        if len(combined_text) > GROUP_CHAR_WARN:
            print(
                f"[UYARI] Grup ({group[0]['date_str']}, id: {group[0]['id_str']}) "
                f"{len(combined_text):,} karakter — {GROUP_CHAR_WARN} eşiğini aşıyor."
            )

        chunks.append(
            {
                "text": combined_text,
                "source": f"tweet {group[0]['date_str']} (id: {group[0]['id_str']})",
                "date": group[0]["date_str"],
            }
        )

    print(f"[BİLGİ] {len(chunks):,} gruba dönüştürüldü.")
    return chunks


# ============================================================================
# 3. EMBEDDING & VEKTÖR DEPOSU
# ============================================================================


def _get_file_signature(filepath: Path) -> dict:
    """
    Dosyanın boyut ve son değiştirme zamanını döner.

    İndeks geçerliliğini kontrol etmek için kullanılır:
    tweets.js değiştiyse indeks yeniden oluşturulmalı.

    Args:
        filepath: Kontrol edilecek dosyanın yolu.

    Returns:
        {"size": int, "mtime": float} dict'i.
    """
    stat = filepath.stat()
    return {"size": stat.st_size, "mtime": stat.st_mtime}


def _should_rebuild_index(chunks_meta: dict, tweets_file: Path) -> bool:
    """
    Mevcut indeksin yeniden oluşturulması gerekip gerekmediğini kontrol eder.

    Kontroller:
        1. İndeks dosyaları (.npy ve .json) mevcut mu?
        2. Kayıtlı embedding model adı hâlâ aynı mı?
        3. tweets.js dosyasının boyutu veya mtime'ı değişmiş mi?

    Args:
        chunks_meta: tweet_chunks.json'dan yüklenen metadata (veya None).
        tweets_file: tweets.js dosyasının yolu.

    Returns:
        True ise indeks yeniden oluşturulmalı.
    """
    # Dosyalar mevcut değilse yeniden oluştur
    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
        return True

    if chunks_meta is None:
        return True

    # Model adı kontrolü
    if chunks_meta.get("model") != EMBEDDING_MODEL:
        print("[BİLGİ] Embedding modeli değişmiş, indeks yenilenecek.")
        return True

    # tweets.js dosya imzası kontrolü
    saved_sig = chunks_meta.get("file_signature", {})
    current_sig = _get_file_signature(tweets_file)
    if (
        saved_sig.get("size") != current_sig["size"]
        or saved_sig.get("mtime") != current_sig["mtime"]
    ):
        print("[BİLGİ] tweets.js değişmiş, indeks yenilenecek.")
        return True

    return False


def _load_existing_index() -> tuple[np.ndarray | None, dict | None]:
    """
    Diskten mevcut indeks ve chunk metadata'sını yüklemeyi dener.

    Returns:
        (embeddings_array, chunks_meta) tuple'ı. Yüklenemezse (None, None).
    """
    try:
        embeddings = np.load(str(INDEX_FILE))
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            chunks_meta = json.load(f)
        return embeddings, chunks_meta
    except Exception as exc:
        print(f"[UYARI] Mevcut indeks yüklenemedi ({exc}), yeniden oluşturulacak.")
        return None, None


def build_or_load_index(
    chunks: list[dict], tweets_file: Path
) -> tuple[np.ndarray, list[dict]]:
    """
    Embedding vektör indeksini oluşturur veya diskten yükler.

    İlk çalıştırmada:
        - sentence-transformers modeli ile tüm chunk'ları batch'ler halinde
          encode eder (RAM dostu, batch_size=64).
        - Vektörleri normalize eder (normalize_embeddings=True).
        - tweet_index.npy ve tweet_chunks.json olarak diske kaydeder.

    Sonraki çalıştırmalarda:
        - Dosyalar mevcutsa VE model adı aynıysa VE tweets.js değişmemişse
          diskten yükler (hızlı başlatma).

    Args:
        chunks: group_tweets'den dönen chunk listesi.
        tweets_file: tweets.js dosyasının yolu (değişiklik tespiti için).

    Returns:
        (embeddings, chunks) tuple'ı.
        embeddings: (N, D) boyutunda normalize edilmiş np.ndarray.
        chunks: chunk dict'lerinin listesi.
    """
    # Mevcut indeksi yüklemeyi dene
    embeddings, chunks_meta = _load_existing_index()

    if not _should_rebuild_index(chunks_meta, tweets_file):
        loaded_chunks = chunks_meta.get("chunks", [])
        if embeddings is not None and len(loaded_chunks) == len(embeddings):
            print(
                f"[BİLGİ] Mevcut indeks yüklendi: "
                f"{len(loaded_chunks):,} chunk, {embeddings.shape[1]} boyut."
            )
            return embeddings, loaded_chunks

    # --- Yeni indeks oluştur ---
    print(f"[BİLGİ] Embedding modeli yükleniyor: {EMBEDDING_MODEL}")
    # sentence_transformers import'u burada — modül seviyesinde zorunlu
    # bağımlılık yaratmamak ve yükleme zamanını azaltmak için lazy import.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    all_embeddings: list[np.ndarray] = []

    print(f"[BİLGİ] {len(texts):,} chunk, {total_batches} batch ile encode ediliyor...")

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1

        emb = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.append(emb)
        print(f"  [{batch_num}/{total_batches}] batch işlendi", flush=True)

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    # Diske kaydet
    np.save(str(INDEX_FILE), embeddings)

    file_sig = _get_file_signature(tweets_file)
    meta = {
        "model": EMBEDDING_MODEL,
        "file_signature": file_sig,
        "chunks": chunks,
    }
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print(
        f"[BİLGİ] Yeni indeks oluşturuldu ve kaydedildi: "
        f"{len(chunks):,} chunk, {embeddings.shape[1]} boyut."
    )
    return embeddings, chunks


# ============================================================================
# 4. ARAMA (RETRIEVAL)
# ============================================================================


def search(
    query: str,
    embeddings: np.ndarray,
    chunks: list[dict],
    model: Any,
    top_k: int = DEFAULT_TOP_K,
) -> list[tuple[float, dict]]:
    """
    Verilen sorguyu embedding'e çevirip kosinüs benzerliği ile arar.

    Normalize edilmiş vektörlerde kosinüs benzerliği = nokta çarpımı.
    Bu sayede ayrıca cosine hesaplaması gerekmez.

    Args:
        query: Kullanıcının arama sorgusu.
        embeddings: (N, D) boyutunda normalize vektör matrisi.
        chunks: chunk dict'lerinin listesi.
        model: SentenceTransformer model nesnesi (query encode için).
        top_k: Döndürülecek en benzer sonuç sayısı.

    Returns:
        (skor, chunk_dict) tuple'larının listesi, skora göre azalan sırada.
    """
    # Sorguyu encode et (normalize)
    q_emb = model.encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    )
    q_emb = q_emb.astype(np.float32)

    # Kosinüs benzerliği = nokta çarpımı (vektörler normalize)
    scores = embeddings @ q_emb.T  # (N, 1)
    scores = scores.flatten()

    # En yüksek skorlu top_k indeksi
    # np.argpartition O(n), tam sort'tan hızlı
    k = min(top_k, len(scores))
    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    results = [(float(scores[i]), chunks[i]) for i in top_indices]
    return results


# ============================================================================
# 5. LLM ÇAĞRISI (Google Gemini — OpenAI-uyumlu uç nokta)
# ============================================================================


def _get_api_key() -> str:
    """
    GOOGLE_API_KEY ortam değişkenini döner.

    Anahtar tanımlanmamışsa anlaşılır hata mesajı verip çıkar.

    Returns:
        API anahtarı string'i.
    """
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        print(
            "[HATA] GOOGLE_API_KEY ortam değişkeni tanımlı değil.\n"
            "       PowerShell:  $env:GOOGLE_API_KEY=\"anahtarınız\"\n"
            "       Bash/Zsh:   export GOOGLE_API_KEY=\"anahtarınız\"\n"
            "       Ardından programı tekrar çalıştırın."
        )
        sys.exit(1)
    return key


def call_llm(context: str, question: str) -> str:
    """
    Google Gemini API'sine OpenAI-uyumlu uç noktadan istek atar.

    Geçici hatalarda (429, 500, 502, 503) basit retry/backoff uygular.
    Kalıcı hatalarda durum kodu ve yanıt gövdesini yazdırarak çıkar.

    Args:
        context: Retrieval sonucu birleştirilmiş tweet metinleri.
        question: Kullanıcının sorusu.

    Returns:
        LLM'in ürettiği cevap string'i.
    """
    api_key = _get_api_key()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Kullanıcı mesajı: bağlam + soru
    user_content = (
        f"=== TWEET METİNLERİ (kronolojik sırada) ===\n{context}\n"
        f"=== SORU ===\n{question}"
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }

    last_exc: Exception | None = None
    last_status: int | None = None
    last_body: str = ""

    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = requests.post(
                LLM_API_URL, headers=headers, json=payload, timeout=60
            )
            last_status = resp.status_code
            last_body = resp.text

            if resp.status_code == 200:
                data = resp.json()
                # OpenAI-uyumlu yanıt formatı
                return (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "[Boş yanıt]")
                )

            if resp.status_code in LLM_RETRYABLE_CODES:
                wait = LLM_RETRY_BACKOFF[min(attempt, len(LLM_RETRY_BACKOFF) - 1)]
                print(
                    f"[UYARI] LLM API hatası {resp.status_code}, "
                    f"{wait:.0f}s sonra tekrar denenecek ({attempt + 1}/{LLM_MAX_RETRIES})..."
                )
                time.sleep(wait)
                continue

            # Kalıcı hata (4xx vb.)
            break

        except requests.RequestException as exc:
            last_exc = exc
            wait = LLM_RETRY_BACKOFF[min(attempt, len(LLM_RETRY_BACKOFF) - 1)]
            print(
                f"[UYARI] Ağ hatası: {exc}. "
                f"{wait:.0f}s sonra tekrar denenecek ({attempt + 1}/{LLM_MAX_RETRIES})..."
            )
            time.sleep(wait)

    # Tüm denemeler başarısız
    error_detail = ""
    if last_status is not None:
        error_detail = f"HTTP {last_status}: {last_body[:500]}"
    elif last_exc is not None:
        error_detail = str(last_exc)

    print(
        f"[HATA] LLM API çağrısı başarısız oldu.\n"
        f"       {error_detail}\n"
        f"       Lütfen API anahtarınızı ve ağ bağlantınızı kontrol edin."
    )
    return "[LLM yanıt veremedi — yukarıdaki hata mesajına bakın.]"


# ============================================================================
# 6. ANA DÖNGÜ
# ============================================================================


def main() -> None:
    """
    Ana çalıştırma fonksiyonu.

    Akış:
        1. tweets.js dosyasını oku ve parse et.
        2. Tweet'leri zaman damgasına göre grupla (chunking).
        3. Embedding indeksini oluştur veya diskten yükle.
        4. Sonsuz sorgu döngüsü başlat.
           - Kullanıcı sorusu al.
           - Semantik arama yap, sonuçları skorlarıyla göster.
           - Bağlamı kronolojik sırala, LLM'e gönder, cevabı yazdır.
    """
    print("=" * 60)
    print("  Tweet Arşivi RAG Sistemi")
    print("=" * 60)
    print()

    # --- 1. Parse ---
    raw_items = load_tweets_js(TWEETS_FILE)
    tweets = parse_and_clean(raw_items)

    if not tweets:
        print("[HATA] Hiç tweet parse edilemedi. Dosya formatını kontrol edin.")
        sys.exit(1)

    # --- 2. Gruplama ---
    chunks = group_tweets(tweets)

    # --- 3. İndeks ---
    embeddings, chunks = build_or_load_index(chunks, TWEETS_FILE)

    # İndeks durumu özeti
    # build_or_load_index zaten ayrıntılı mesaj verdi; burada özet.
    print()
    print("-" * 60)
    print(
        f"  {len(tweets):,} tweet parse edildi, "
        f"{len(chunks):,} gruba dönüştürüldü."
    )
    print("-" * 60)
    print()

    # API anahtarı erken kontrol (döngüye girmeden)
    _get_api_key()

    # Sorgu modeli (lazy load — indeks diskten yüklendiyse model
    # henüz yüklenmemiş olabilir)
    print("[BİLGİ] Sorgu embedding modeli yükleniyor...")
    from sentence_transformers import SentenceTransformer

    query_model = SentenceTransformer(EMBEDDING_MODEL)
    print("[BİLGİ] Model hazır. Sorularınızı sorabilirsiniz.\n")

    # --- 4. Sorgu döngüsü ---
    while True:
        try:
            question = input("Soru: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÇıkılıyor...")
            break

        if not question:
            continue
        if question.lower() in ("q", "quit", "exit", "çık"):
            print("Çıkılıyor...")
            break

        # Retrieval
        results = search(question, embeddings, chunks, query_model, top_k=DEFAULT_TOP_K)

        # Sonuçları skorlarıyla göster
        print("\n--- Bulunan ilgili gruplar (skor sırasına göre) ---")
        for score, chunk in results:
            print(f"  [{score:.3f}] {chunk['source']}")

        # LLM'e gönderilecek bağlamı kronolojik sırala
        # (skor listesi ekranda kalır, sadece LLM'e giden context sırası değişir)
        sorted_by_date = sorted(results, key=lambda r: r[1]["date"])

        # Toplam bağlam uzunluğu kontrolü: 30000 karakteri aşarsa skora göre kes
        context_parts: list[tuple[float, dict]] = []
        total_len = 0
        for score, chunk in results:  # skor sırasına göre (yüksekten düşüğe)
            chunk_len = len(chunk["text"]) + len(chunk["source"]) + 20
            if total_len + chunk_len > MAX_CONTEXT_CHARS and context_parts:
                print(
                    f"[UYARI] Bağlam {MAX_CONTEXT_CHARS:,} karakter sınırına "
                    f"ulaştı, düşük skorlu gruplar kesildi."
                )
                break
            context_parts.append((score, chunk))
            total_len += chunk_len

        # Kalan chunk'ları kronolojik sırala
        context_parts_sorted = sorted(context_parts, key=lambda r: r[1]["date"])

        # Bağlam metni oluştur
        context_text = "\n\n".join(
            f"[Kaynak: {chunk['source']}]\n{chunk['text']}"
            for _, chunk in context_parts_sorted
        )

        # LLM çağrısı
        print("\n[BİLGİ] LLM'e soruluyor...\n")
        answer = call_llm(context_text, question)

        print("--- Cevap ---")
        print(answer)
        print()


if __name__ == "__main__":
    main()
