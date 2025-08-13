# Yol Haritası – Locate AI Anomali Tespit Modülü

## 1. Veri Hazırlığı
**Amaç:** Modelin ve geofence kontrolünün çalışacağı veri formatını oluşturmak.

### Yapılacaklar
- **Veri formatı belirleme:**
  - Zorunlu alanlar: `timestamp` (ISO 8601 formatı), `lat` (float), `lon` (float), `hız` (m/s), `device_id` (string).
- **Veri temizleme:**
  - Eksik değerleri (NaN, boş) sil.
  - Fiziksel olarak imkânsız hız/mesafe değerlerini filtrele (ör. 300 km/s üstü hızlar).
- **Simülasyon veri seti oluşturma (gerekirse):**
  - **Normal rota:** sabit yön ve hızda ilerleyen GPS noktaları.
  - **Anomali senaryoları:**
    - Ani hız artışı/azalması (speed spike/drop).
    - Rota sapması (ani konum sıçraması).
    - Güvenli alan dışına çıkma (geofence violation).

### Çıktı
- `sim_train.jsonl`, `sim_test.jsonl` dosyaları.
- Daha sonra gerçek cihazdan log geldiğinde `real_test.jsonl`.

---

## 2. Geofence (Kural Tabanlı) Algoritması
**Amaç:** Belirlenen güvenli alanın dışına çıkıldığında uyarı üretmek.

### Yapılacaklar
- **Parametreler:**
  - Merkez koordinatları (`lat0`, `lon0`).
  - Yarıçap (`radius_m` metre cinsinden).
- **Kontrol algoritması:**
  - Her GPS noktası için, merkez ile nokta arasındaki mesafeyi hesapla (Haversine formülü).
  - Mesafe > yarıçap ise “alan dışı” olarak işaretle.
- **Debounce mantığı:**
  - GPS sinyalindeki anlık sapmalar yanlış alarm üretmemeli.
  - Kullanıcı **kesintisiz** `N` saniye (örn. 30) alan dışında kalırsa uyarı üret.

### Çıktı
- `GEOFENCE_EXIT` uyarısı (JSON alarm çıktısında `anomaly_reason` alanına yazılacak).

---

## 3. ML Model Geliştirme
**Amaç:** Geçmiş rotalara dayalı olarak olağandışı hareketleri tespit eden hafif model geliştirmek.

### Yapılacaklar
- **Özellik çıkarımı:**
  - Ardışık iki nokta arası mesafe (m).
  - Anlık hız (m/s).
  - Hız farkı (m/s²).
  - Yön değişimi (derece).
- **Model seçimi:**
  - Isolation Forest (IF) veya DBSCAN.
  - Tercihen IF (daha kolay threshold ayarı ve hızlı çalışma).
- **Eğitim:**
  - Normal örnekler ile modeli eğit.
  - Validation set ile `contamination` (IF) veya `eps/min_samples` (DBSCAN) parametrelerini ayarla.
- **Performans testi:**
  - Test setinde `recall ≥ %80` ve `false alarm ≤ %10` hedeflerini kontrol et.
  - Metrikleri (`recall`, `false alarm`, TP, FP, FN, TN) raporla.

### Çıktı
- Eğitilmiş model dosyası (`isoforest.joblib` veya benzeri).
- Test raporu.

---

## 4. Entegrasyon (API)
**Amaç:** Modelin API aracılığıyla çalışması.

### Yapılacaklar
- **Teknoloji:** Flask veya FastAPI (dokümanda ikisi de geçiyor).
- **Endpoint:**
  - `POST /detect`:
    - **Girdi:** JSON formatında GPS noktası (`device_id`, `timestamp`, `lat`, `lon`, `speed`).
    - **Çıktı:** JSON alarm (`device_id`, `timestamp`, `location`, `anomaly_reason`).
- **Çalışma mantığı:**
  - Gelen veriyi önce geofence kontrolünden geçir (debounce ile).
  - Sonra ML modelinde değerlendir.
  - Anomali tespit edilirse uygun `anomaly_reason` kodu ile JSON döndür.

### Çıktı
- Çalışır API servisi.
- Örnek JSON alarm çıktısı:
    {
      "device_id": "dev01",
      "timestamp": "2025-08-13T12:00:00Z",
      "location": {"lat": 36.8, "lon": 34.6},
      "anomaly_reason": "GEOFENCE_EXIT"
    }

---

## 5. Test ve Raporlama
**Amaç:** Modülün tüm senaryolarda çalıştığını göstermek.

### Yapılacaklar
- **Simülasyon testi:**
  - Tüm anomali senaryolarını (hız artışı/azalması, rota sapması, güvenli alan ihlali) çalıştır.
  - API’ye veri göndererek çıktıları doğrula.
- **Gerçek veri testi:**
  - Locate cihazından gelen GPS logu üzerinde aynı test prosedürünü uygula.
- **Rapor:**
  - Performans metrikleri tablosu (recall, false alarm, TP, FP, FN, TN).
  - Örnek JSON çıktıları.
  - Gereksinimlerin karşılandığını gösteren kısa açıklama.

### Çıktı
- `report.md` veya PDF formatında nihai rapor.

---

## Mantıklı Çalışma Sırası
1. **Veri Hazırlığı** → Simülasyon setlerini hazırla, temizlik adımlarını uygula.
2. **Geofence Algoritması** → Debounce’lu alan dışı tespiti uygula.
3. **ML Modeli** → Özellik çıkarımı, model eğitimi, performans testi.
4. **API Entegrasyonu** → `POST /detect` ile uçtan uca çalışır hale getir.
5. **Test & Raporlama** → Simülasyon + gerçek veri ile test, rapor üretimi.
