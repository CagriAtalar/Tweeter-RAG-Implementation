# Tweet Arşivi RAG Sistemi

Twitter/X arşiv dosyanızdaki (`tweets.js`) tweet'leriniz üzerinde anlamsal arama yapan ve Google Gemini ile doğal dilde cevap üreten tek dosyalık bir RAG sistemi.

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

1. [Google AI Studio](https://aistudio.google.com/apikey) üzerinden bir API anahtarı alın.

2. API anahtarını ortam değişkeni olarak tanımlayın:

   **PowerShell:**
   ```powershell
   $env:GOOGLE_API_KEY="anahtarınız"
   ```

   **Bash:**
   ```bash
   export GOOGLE_API_KEY="anahtarınız"
   ```

3. `tweets.js` dosyanızı `tweet_rag.py` ile aynı klasöre koyun.

4. Çalıştırın:
   ```bash
   python tweet_rag.py
   ```

İlk çalıştırmada embedding modeli indirilir ve tweet'ler vektörleştirilir (birkaç dakika sürebilir). Sonraki çalıştırmalarda indeks diskten yüklenir ve hızlıca başlar.

Çıkmak için `q` yazın.

## Nasıl Çalışır

- `tweets.js` parse edilir, retweet'ler atılır.
- Ardışık 10 dakika içindeki tweet'ler aynı gruba alınır (thread birleştirme).
- Her grup `paraphrase-multilingual-MiniLM-L12-v2` modeli ile vektöre çevrilir.
- Soru sorulduğunda kosinüs benzerliği ile en ilgili gruplar bulunur.
- Bulunan tweet'ler kronolojik sırayla Gemini'ye gönderilir; model yalnızca arşive dayanarak cevap verir.

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `tweet_rag.py` | Ana script |
| `requirements.txt` | Bağımlılıklar |
| `tweet_index.npy` | Vektör indeksi (otomatik oluşur) |
| `tweet_chunks.json` | Chunk metadatası (otomatik oluşur) |

## Gereksinimler

- Python 3.10+
- Google API anahtarı
- Twitter/X arşiv dosyası (`tweets.js`)
