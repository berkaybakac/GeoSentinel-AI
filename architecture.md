# Locate – AI Anomali Tespit Modülü

## 0. Overview

Bu modül, GPS konum verilerini işleyerek **kural tabanlı geofence** kontrolü ve **hafif makine öğrenmesi** (Isolation Forest) ile anormallikleri tespit eder. Anomali bulunduğunda **JSON alarm çıktısı** üretir ve API üzerinden döndürür. Hedef performans: **recall ≥ %80** ve **false alarm ≤ %10**.

> **Veri Kaynağı Notu (Netlik Eklendi):** Bu projenin **resmî değerlendirme kaynağı DBRA24**’tür. Simülasyon, yalnızca geofence debounce/gürültü senaryolarını **tanısal** amaçla doğrulamak için **opsiyoneldir** ve **kabul metriklerine dahil edilmez**.

---

## 1. Veri Hazırlığı

**Amaç:** Modelin ve geofence kontrolünün çalışacağı veri formatını oluşturmak.

### Zorunlu Alanlar (schema)

- `timestamp` (ISO 8601, UTC; örn: `2025-08-14T10:15:00Z`)
- `lat` (float)
- `lon` (float)
- `speed` (float, m/s)
- `device_id` (string)

### Temizleme Kuralları

- Eksik (NaN/boş) kayıtları sil.
- **Opsiyonel hız filtresi (Netlik Eklendi):** Fiziksel olarak imkânsız hızlar için **opsiyonel** filtre bulunur. `config.filters.max_speed_kmh` bir sayı ise bu değerin üzerindeki hızlar atılır; `null` ise filtre devre dışıdır.
- Zaman sıralamasını kontrol et (aynı `device_id` için `timestamp` monoton artmalı).
- Gerekirse birim dönüşümü yap (örn. dataset `speed_kmh` ise **m/s**’e çevir: `mps = kmh / 3.6`).

### Simülasyon (opsiyonel)

- **Normal rota:** sabit/ılımlı hız ve yönle ilerleyen GPS noktaları.
- **Anomali senaryoları:**
  - Ani hız artışı/azalması (speed spike/drop)
  - Rota sapması / ani konum sıçraması (route jump)
  - Geofence dışına çıkma (geofence violation)

**Çıktılar:**

- `sim_train.jsonl`, `sim_test.jsonl` (**opsiyonel**, yalnızca tanısal)
- **Resmî test:** `dbra24_test.jsonl`

---

## 2. Geofence (Kural Tabanlı) Algoritması

**Amaç:** Belirlenen güvenli alanın dışına çıkıldığında uyarı üretmek.

### Çalışma Prensibi

- **Parametreler:** `lat0`, `lon0`, `radius_m`, `debounce_sec`
- **Mesafe hesabı:** **Haversine** formülü ile merkez–nokta arası mesafe.
- **Alan dışı kuralı:** `distance > radius_m` ise **dışarıda**.
- **Debounce:** GPS gürültüsünü azaltmak için **kesintisiz `debounce_sec`** (varsayılan **10 s**) süreyle alan dışında kalırsa alarm üret.

### Parametre Yönetimi (Netlik Eklendi)

- Tüm geofence parametreleri `config.json` dosyasından okunur. Parametre değişimi için API kodunu değiştirmeye gerek yoktur.
- `lat0/lon0/radius_m` değerleri **operasyonel gereksinimlerle** belirlenir (ör. tesis/rota merkezi ve güvenli yarıçap). **Otomatik türetme** (ör. medyan merkez + 95. persentil yarıçap) **kapsam dışı/opsiyonel** bir kolaylıktır; kullanılacaksa yine `config.json`’a yazılır.

### Alarm Üretimi (Geofence)

Alan dışı şartı **debounce** ile sağlandığında aşağıdaki JSON döndürülür:

```json
{
  "device_id": "dev01",
  "timestamp": "2025-08-14T12:00:00Z",
  "location": { "lat": 36.8, "lon": 34.6 },
  "anomaly_reason": "GEOFENCE_EXIT",
  "alarm": {
    "code": 1000,
    "label": "GEOFENCE_EXIT",
    "source": "GEOFENCE",
    "window_sec": 10
  }
}
```

**Notlar:**

- `debounce_sec` düşükse hızlı tepki fakat daha fazla false alarm; yüksekse tam tersi.
- Bu projede varsayılan **10 s** ve **parametrik**.

---

## 3. ML Model Geliştirme

**Amaç:** Geçmiş rotalara dayalı olarak olağandışı hareketleri tespit eden **hafif ve hızlı** bir model geliştirmek.

### Model Seçimi

- **Isolation Forest (IF)** kullanılacaktır (`sklearn.ensemble.IsolationForest`).
- Gerekçe:
  - Hafif ve hızlı (ağaç tabanlı; ~120k kayıt gibi hacimleri düşük işlem gücüyle işler).
  - Anomali oranı `contamination` ile ayarlanabilir.
  - Parametre ayarı görece basit (`n_estimators`, `max_samples`).
  - Çeşitli sürüş davranışlarında pratik ve ölçeklenebilir.
- **Alternatif DBSCAN** parametre optimizasyon zorluğu ve gerçek zamanlı dezavantajları nedeniyle **tercih edilmemiştir**.

### Özellikler (Feature Engineering)

- Ardışık iki nokta arası **mesafe** (Haversine, metre)
- **Hız** (m/s)
- **İvme** (acceleration, m/s²; ardışık hız farkı / zaman farkı)
- **Yön değişimi** (heading farkı, derece)
- **Direksiyon açısı** (`steering_angle`, keskin dönüş göstergesi)

#### Zamansal Bağlam (opsiyonel)

Gerçek zamanlı akışta kısa bir pencere üzerinden istatistik eklemek, IF performansını iyileştirebilir:

- Son `features.window_sec` içinde: hız/ivme/heading için `mean`, `std`, `max spike`, `turn-rate`, `jerk` gibi türevler.
- **Varsayılan:** devre dışı (0 sn). İhtiyaç görülürse etkinleştirilecektir.

### Pipeline

1. **Veri Yükleme:** **DBRA24** (simülasyon, geofence davranışını tanısal doğrulamak için **opsiyonel**).
2. **Ön İşleme:** Eksik/hatalı kayıt temizliği, filtreler, birim dönüşümleri.
3. **Özellik Çıkarımı:** Yukarıdaki feature set (mapping tablosu aracılığıyla).
4. **Eğitim:** Isolation Forest (tercihen yalnız **normal** davranış örnekleriyle).
5. **Model Kaydetme:** `joblib` ile `isoforest.joblib`.
6. **Değerlendirme:** Test setinde hedef metrikler (aşağıda).

- **Eğitim verisi seçimi:** IF denetimsiz olarak **normal** örneklerle eğitilir (DBRA24 için `route_anomaly==0 AND anomalous_event==0` kayıtlar). Etiketler sadece **değerlendirme** (4. bölüm) için kullanılır.

### Kullanılacak Kütüphaneler

- `scikit-learn` (IF)
- `joblib` (model persist)
- `numpy`, `pandas` (veri işleme)
- `math` (Haversine hesabı)

---

## 4. Model Testi ve Dataset Uyarlaması

**Amaç:** Dataset’in yapısına uyum sağlamak ve performansı doğrulamak.

### 4.1 Dataset Keşfi ve Uyum

- `pandas.info()`, `df.describe()`, `df.isna().sum()` ile tipler, dağılımlar, null oranları analiz edilir.
- Architecture’daki beklenen alanlar ↔ dataset kolonları **eşleştirilir** (mapping).
  - Örnek: `speed` ↔ `speed_kmh` → **m/s**’e dönüştür.
  - Örnek: `lat` ↔ `latitude`, `lon` ↔ `longitude`
- Özellik çıkarımı **mapping tablosu** üzerinden yapılır (dataset değişse de tek yerden güncelle).

### 4.2 Performans Testi (Netlik Eklendi)

- **Resmî test seti:** **DBRA24 test** bölümü.
- Model çıktıları ile **gerçek etiketler** karşılaştırılır.
- **Resmî metrik:** **Top-line micro-average** (tüm cihazlar/örnekler birleşik); kabul bu skorla yapılır.
- **Metrikler:**
  - Confusion Matrix: TP, FP, FN, TN
  - **Recall (≥ %80)**, **False Alarm Rate (≤ %10)**, Precision
- **Threshold/Eşik:** Dengesiz veri nedeniyle **ROC yerine PR** çerçevesi kullanılır; `contamination` grid taranır, **önce** **FAR ≤ 0.10** koşulunu sağlayan eşikler filtrelenir, **ardından** Recall maksimize eden eşik seçilir; **seçim ve metrikler** `out/eval/thresholds.json` + `out/eval/report.md`’de kayıt altına alınır.
- Sonuçlar `report.md` içinde raporlanır.
- Test çıktısı, model parametrelerinin (örn. `contamination`) **iyileştirilmesi** için tekrar kullanılır.

### 4.3 Ölçüm Kuralları (Net Tanım)

- **Pozitif sınıf:** “Anomali”
- **Değerlendirme birimi:** Nokta bazlı (her GPS örneği). Örnekleme hızı değişkense zaman senkronizasyonundan sonra nokta bazlı değerlendirme yapılır.
- **Confusion Matrix:**
  - **TP:** Anomali olan örneğin doğru yakalanması
  - **FP:** Normal örnek için yanlış alarm (false alarm)
  - **FN:** Anomali örneğin kaçırılması
  - **TN:** Normal örnekte alarm üretilmemesi
- **Metrikler (nokta bazlı):**
  - **Recall =** TP / (TP + FN)
  - **False Alarm Rate (FAR) =** FP / (FP + TN)
  - **Precision =** TP / (TP + FP)
- **Hedefler:** **Recall ≥ 0.80** ve **FAR ≤ 0.10**
  (**Resmî skor**: **DBRA24 test** üzerinde **micro-average**).
- **Etiket kaynağı:**
  - **DBRA24:** `geofencing_violation`, `route_anomaly`, `anomalous_event`.
  - **Simülasyon (opsiyonel):** yerleşik (ground-truth) anomali etiketleri; **resmî skora dahil değildir**.
- **Parametre taraması (IF):** `contamination ∈ [0.05, 0.12]` gibi aralıklar denenebilir; **kesin aralık** dataset/ortama göre belirlenir.

---

## 5. Entegrasyon (API)

**Amaç:** Modelin API aracılığıyla çalışması ve anomali durumunda JSON alarm üretmesi.

### Teknoloji

- **FastAPI** (tip desteği, performans ve otomatik dokümantasyon avantajları nedeniyle).

### Endpoint

- `POST /detect`

  - **Girdi (JSON):**
    ```json
    {
      "device_id": "dev01",
      "timestamp": "2025-08-14T12:00:00Z",
      "lat": 36.8005,
      "lon": 34.617,
      "speed": 15.0
    }
    ```
  - **İş Akışı:**
    1. Şema ve birim doğrulama.
    2. **Geofence** kontrolü (debounce ile).
    3. **ML modeli** ile anomali skoru ve karar.
    4. Anomali varsa **JSON alarm** döndür.
  - **Çıktı (alarm örneği):**
    > Not: Tam şema ve zorunlu alanlar için **5.2 Alarm JSON Şeması** bölümüne bakınız.

#### Normal (Alarm Yok) Cevap Şeması

Anomali tespit edilmezse API şu minimal cevabı döndürür:

```json
{ "device_id": "dev01", "timestamp": "2025-08-14T12:00:00Z", "anomaly": false }
```

### 5.1 Alarm Kodları (Sabit Sözlük)

Aşağıdaki sabit kodlar alarm nedenlerini standardize eder:

- `1000` → `GEOFENCE_EXIT` (debounce sonrası güvenli alan dışı)
- `1100` → `SPEED_ANOMALY` (ani hız artışı/düşüşü eşik üstü)
- `1101` → `ROUTE_JUMP` (rotada sıçrama / tutarsız konum; büyük haversine farkı)
- `1200` → `MODEL_ANOMALY` (ML model skoru eşik üstü; Isolation Forest)

**Çakışma kuralı (Netlik Eklendi):** Aynı anda birden fazla tetikleyici oluşursa **tek alarm** üretilir; öncelik
`GEOFENCE_EXIT > MODEL_ANOMALY > SPEED_ANOMALY > ROUTE_JUMP`. İkincil tetikleyiciler, **cevapta yer almaz** ancak **iç log**’a yazılabilir (opsiyonel).

### 5.2 Alarm JSON Şeması (Minimal ve İzlenebilir)

> **Uyumluluk notu:** Dokümanda geçen **top-level `anomaly_reason`** alanı **korunur** (değerlendirme sistemleri bunu bekleyebilir).
> Yeni `alarm` nesnesi ise izlenebilirlik için ek bilgileri taşır.

**GEOFENCE örneği:**

```json
{
  "device_id": "dev01",
  "timestamp": "2025-08-14T12:00:00Z",
  "location": { "lat": 36.8005, "lon": 34.617 },
  "anomaly_reason": "GEOFENCE_EXIT",
  "alarm": {
    "code": 1000,
    "label": "GEOFENCE_EXIT",
    "source": "GEOFENCE",
    "window_sec": 10
  }
}
```

**MODEL örneği:**

```json
{
  "device_id": "dev01",
  "timestamp": "2025-08-14T12:00:05Z",
  "location": { "lat": 36.8007, "lon": 34.618 },
  "anomaly_reason": "MODEL_ANOMALY",
  "alarm": {
    "code": 1200,
    "label": "MODEL_ANOMALY",
    "source": "MODEL",
    "score": 0.91,
    "threshold": 0.85,
    "window_sec": 5
  }
}
```

- **Zorunlu alanlar:** `device_id`, `timestamp`, `location`, `anomaly_reason`, `alarm.code`, `alarm.label`, `alarm.source`
- **Opsiyonel ama faydalı:** `alarm.score`, `alarm.threshold`, `alarm.window_sec`
  - `score/threshold` **sadece** `source="MODEL"` olduğunda set edilir (geofence’te yoktur).
  - `window_sec` geofence’te **debounce**, modelde ise **özellik penceresi** olarak yorumlanır.

**Skor/Threshold Ölçeği (Isolation Forest)**

- `alarm.score` ve `alarm.threshold` **0–1 aralığına normalize** edilir.
- Normalize yaklaşımı: `raw = -decision_function(x)`; `score = (raw - min) / (max - min)`.
  Burada `min/max`, eğitim dağılımından veya hareketli bir pencereden (rolling) alınır.
- `threshold` değeri, seçilen `contamination` yüzdesine karşılık gelen **persentil**dir (örn. 0.85).

### API Cevabı Modu (Netlik Eklendi)

- Varsayılan: **İzlenebilir Mod** (zorunlu alanlar + `alarm{code,label,source}`).
- `config.api.alarm_verbose = false` ise **Minimal Mod** (yalnız zorunlu alanlar döner).

### Durum Yönetimi (Debounce)

- Debounce için **cihaz bazlı kısa süreli bellek** (in-memory pencere) tutulur:
  - `device_id → son N saniyelik alan-dışı süresi`
- Kalıcı depolama gerekmiyor (gereksinim dışı).

---

## 6. Test ve Raporlama

**Amaç:** Tüm senaryolarda modülün doğru çalıştığını göstermek.

### Simülasyon Testi (çalıştırması zorunlu, üretimi opsiyonel)

- **Sim seti üretimi opsiyonel**, **çalıştırılması zorunludur**; geofence **debounce** ve **model tetikleyicileri** beklenen yerlerde alarm veriyor mu?
- Örnek komut:

```bash
python -m scripts.eval \
  --test data/sim/sim_test.jsonl \
  --mapping mapping_dbra24.json \
  --config config.json \
  --model isoforest.joblib \
  --out_dir out/sim_eval
```

- Çıktı: `out/sim_eval/report.md` (beklenen vs görülen alarm tablosu + kısa yorum).
- **Not:** Simülasyon sonuçları **resmî skora dahil edilmez**; tanısal ek tablo olarak raporlanabilir.

### Gerçek Veri Testi (Resmî)

- **DBRA24 test bölümü** aynı prosedür ile test edilir.
- **Resmî skor politikası (Netlik Eklendi):** Kabul kriteri, DBRA24 üzerinde hesaplanan **top-line micro-average** metriklerle doğrulanır.

### Rapor

- Metrik tablosu: TP, FP, FN, TN, Recall, Precision, False Alarm Rate
- Örnek JSON alarm çıktıları
- Kısa sonuç: **recall ≥ %80**, **false alarm ≤ %10** sağlandı/sağlanmadı

**Çıktı:**

- `report.md` (veya PDF)

---

## 7. Kabul Kriterleri (Doğrulama Listesi)

- [ ] GPS verilerini okuyup işleyen Python script(ler)i mevcut.
- [ ] Geofence tanımlama ve **debounce’lu** çıkış uyarısı tamam.
- [ ] ML modeli (Isolation Forest) **eğitildi** ve dosyaya kaydedildi (`isoforest.joblib`).
- [ ] Test setinde **recall ≥ %80**, **false alarm ≤ %10** hedefleri sağlandı (resmî test: **DBRA24**).
- [ ] API (`POST /detect`) veri aldığında **doğru JSON alarm** üretiyor.

---

## 8. Mantıklı Çalışma Sırası (Roadmap)

1. **Veri Hazırlığı** → Temizlik, birim dönüşümleri, **DBRA24 mapping**, (opsiyonel) simülasyon setleri.
2. **Geofence** → Haversine + debounce (10 s) ve parametreleri `config.json`.
3. **ML Modeli** → Feature’lar, IF eğitimi, `isoforest.joblib` kaydı.
4. **API Entegrasyonu** → `POST /detect`: geofence → model → alarm.
5. **Test & Raporlama** → **DBRA24** (resmî) + (varsa) simülasyon (tanısal), metrikler, `report.md`.

---

## 9. Çıktı Artefaktları (Dosyalar)

- **Veri:** `dbra24_test.jsonl` (resmî), `sim_train.jsonl`, `sim_test.jsonl` (opsiyonel)
- **Model:** `isoforest.joblib`
- **Konfigürasyon:** `config.json`, `mapping_dbra24.json`
- **Rapor:** `report.md`

---

> **Not:** Bu doküman, dokümantasyon kapsamına **sadık** kalınarak hazırlanmıştır. Bildirim servisleri (push/SMS/e-mail), kalıcı durum depolama, yetkilendirme vb. ek özellikler kapsam dışıdır ve ilerleyen fazlarda ele alınabilir.

---

## Appendix A — DBRA24 Dataset Uygunluk Özeti

**Kaynak:** Driver Behavior and Route Anomaly (DBRA24)

**Güçlü Yönler**

- GPS alanları mevcut: `timestamp`, `latitude/longitude` (→ `lat/lon`), `speed` (→ m/s’e çevrilebilir).
- Davranış özellikleri: `acceleration`, `steering_angle` vb. (feature engineering için uygun).
- (Sağlanan bilgiye göre) hazır etiketler: `geofencing_violation`, `route_anomaly`, `anomalous_event` → test/validasyon için pratik.
- Tek setle hem geofence hem ML tespiti senaryoları kurulabilir.

**Sınırlılıklar & Etki**

- Coğrafi çeşitlilik sınırlı (tek bölge) → genelleme riski. _Bu proje kapsamı için kritik değil, not edilmiştir._
- Geofence sınırları veri setinde sabit olmayabilir → proje konfigürasyonundan tanımlanacaktır.
- GPS gürültüsü/hataları false alarmı artırabilir → debounce (geofence) + temel filtreler (veri temizliği) uygulanır.

**Sonuç:** Bu proje için **uygun**. Gerekli eşlemeler mapping ile çözülecek (örn. `speed_kmh → speed_mps`, `latitude/longitude → lat/lon`).

> **Kapsam Notu:** Bu ek, yalnızca proje boyunca **DBRA24 dataset’i** kullanıldığında geçerlidir. Farklı dataset’lerde bu ek ya çıkarılır ya da ilgili dataset’e göre güncellenir.

---

### Appendix A.1 — DBRA24 → Proje Alan Eşlemesi (Mapping)

| Beklenen Alan (Projede) | DBRA24 Kolon Adı       | Dönüşüm / Not                                                                                    |
| ----------------------- | ---------------------- | ------------------------------------------------------------------------------------------------ |
| device_id               | `vehicle_id`           | string olarak kullan                                                                             |
| timestamp               | `timestamp`            | Naive ise UTC varsay; ISO 8601 formatına dönüştür                                                |
| lat                     | `latitude`             | float                                                                                            |
| lon                     | `longitude`            | float                                                                                            |
| speed (m/s)             | `speed`                | **km/h → m/s**: `speed_mps = speed / 3.6` (DBRA24 hız birimi km/h)                               |
| acceleration (m/s²)     | `acceleration`         | **Öneri:** Tutarlılık için hızdan **yeniden türet** (`Δspeed_mps / Δt`); kolon referans olabilir |
| heading (deg)           | `heading`              | 0–360°                                                                                           |
| steering_angle (deg)    | `steering_angle`       | ~ -45 – +45°                                                                                     |
| label_geofence          | `geofencing_violation` | bool                                                                                             |
| label_route             | `route_anomaly`        | bool                                                                                             |
| label_event             | `anomalous_event`      | bool                                                                                             |

> **Tek Nokta Kaynağı (Netlik Eklendi):** Kolon eşlemesi **tek bir mapping dosyası** üzerinden yönetilir (örn. `mapping_dbra24.json`). Hız dönüşümü (km/h→m/s) bu dosyada tanımlıdır; dataset değişirse yalnızca mapping güncellenir.

---

### Appendix A.2 — DBRA24 Test Etiket Politikası

- **ML (IF) için pozitif:** `route_anomaly OR anomalous_event`
- **Geofence için pozitif:** `geofencing_violation`
- **Bileşik rapor (opsiyonel):** ML ve geofence metrikleri **ayrı** verilir; istenirse **birleşik anomali = (ML ∪ geofence)** ayrıca raporlanabilir.

---

### Appendix A.3 — Geofence Doğrulama Notu (dürüstlük)

DBRA24’te `geofencing_violation` etiketi bulunsa da, bu etiketin **hangi merkez/yarıçap** ile üretildiği bilinmemektedir.
Bu projede geofence **bizim `config.json`** dosyamızdaki `lat0, lon0, radius_m` ile tanımlanır. Bu nedenle:

- Geofence modülünün **asıl doğrulaması**, architecture’da tanımlanan **simülasyon senaryoları** üzerinden yapılacaktır (opsiyonel).
- DBRA24’teki `geofencing_violation` etiketi **ek doğrulama** olarak raporlanır; **birebir eşleşme beklenmemelidir** (geometri farkı olabilir).

---

## Appendix B — Örnek Konfigürasyon (valid JSON)

Aşağıdaki JSON **örnektir**; dağıtımda değerler ortamınıza göre güncellenir.

```json
{
  "version": "1.0",
  "api": {
    "alarm_verbose": true
  },
  "geofence": {
    "lat0": 0.0,
    "lon0": 0.0,
    "radius_m": 500,
    "debounce_sec": 10
  },
  "filters": {
    "max_speed_kmh": null,
    "max_accel_mps2": null
  },
  "features": {
    "window_sec": 0,
    "scaler": "standard"
  },
  "model": {
    "type": "isolation_forest",
    "n_estimators": 256,
    "max_samples": 1024,
    "contamination": null,
    "random_state": 42
  },
  "dataset": {
    "mapping_file": "mapping_dbra24.json"
  }
}
```

**Notlar**

- `geofence.lat0/lon0/radius_m` dağıtımda **zorunlu** doldurulmalıdır.
- `contamination` grid aramasıyla bulunur; boş bırakılırsa deploy sırasında set edilir.
- `window_sec=0` → zamansal pencere devre dışı (basit modül varsayılanı).
- `api.alarm_verbose=false` ile **Minimal Mod** çıktısı alınabilir.

---

## 10. Kritik Yol ve DoD (Definition of Done)

Aşağıdaki sıra, projenin gerçek icra sırasıdır. Her adım, DoD sağlanınca tamam sayılır.

### 1) Veri Hazırlığı → **DoD**

- Girdi: DBRA24 CSV (ve/veya opsiyonel simülasyon).
- İş: Şema/temizlik, birim dönüşümü (km/h→m/s), sıralama, mapping doğrulama.
- Çıktı: `dbra24_test.jsonl` (resmî), `sim_train.jsonl`, `sim_test.jsonl` (opsiyonel).
- Doğrulama: Zorunlu alanlar (`timestamp, lat, lon, speed, device_id`) eksiksiz.

### 2) Geofence → **DoD**

- İş: Haversine + `radius_m`, `debounce_sec=10` (config’ten).
- Çıktı: Alan dışına çıkışta `GEOFENCE_EXIT` JSON alarmı.
- Doğrulama: (Opsiyonel) simülasyon senaryolarında beklenen yerde alarm **üretiliyor**.

### 3) ML Model (Isolation Forest) → **DoD**

- İş: Özellik çıkarımı (mesafe, hız, ivme, yön farkı, direksiyon açısı), eğitim.
- Çıktı: `isoforest.joblib` (joblib ile kaydedildi).
- Doğrulama: Test döngüsü çalışıyor; metrikler hesaplanıyor (eşik/fine-tune bir sonraki adımda).

### 4) API Entegrasyonu (FastAPI) → **DoD**

- İş: `POST /detect` akışı: şema → geofence → model → alarm.
- Çıktı: Anomali varsa dokümana uygun **JSON alarm** dönüyor (varsayılan: İzlenebilir Mod).
- Doğrulama: Tek cihaz akışında uçtan uca çağrı başarılı.

### 5) Test & Raporlama → **DoD**

- İş: **DBRA24 test** (resmî) + (varsa) simülasyon (tanısal); metrikler (TP, FP, FN, TN, Recall, Precision, FAR).
- Hedef: **Recall ≥ 0.80**, **FAR ≤ 0.10** (resmî skor: DBRA24 micro-average).
- Çıktı: `report.md` (hedefler sağlandı/sağlanmadı bilgisiyle).

#### Doğal İterasyon Döngüleri

- **[3 ↔ 5] Eşik/`contamination` ayarı:** FAR ≤ 0.10’u tutturarak Recall’ı maksimize et.
- **[2 ↔ 4] Entegrasyon köşeleri:** Debounce ve cihaz bazlı pencere mantığında küçük düzeltmeler gerekebilir.

> Not: Simülasyon üretimi (1) ile geofence geliştirme (2) kısmen paralel yürütülebilir; API iskeleti (4) önce geofence ile bağlanıp sonra modele genişletilebilir.
