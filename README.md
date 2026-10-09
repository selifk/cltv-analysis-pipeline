# Müşteri Yaşam Boyu Değeri (CLTV) Analiz Pipeline'ı

Bu proje, müşteri verileri üzerinden BG/NBD ve Gamma-Gamma modellerini kullanarak Müşteri Yaşam Boyu Değeri (CLTV) tahmini ve segmentasyonu gerçekleştiren Python tabanlı bir analiz pipeline'ıdır.

## Proje Dosyaları
* `cltv_senior_refactored.py`: Ana analiz ve modelleme scripti.
* `run_cltv_pipeline.bat`: Pipeline'ı otomatize eden betik.
* `requirements_cltv.txt`: Gerekli kütüphane bağımlılıkları.

## Proje Çıktıları ve Raporlar
* `VMG_CLV_RAPOR.docx`: Detaylı analiz ve iş zekası raporu.
* `VMG_SUNU.pptx`: Proje sunumu ve yönetici özeti.
* `cltv_results.csv`: Model tahmin sonuçları veriseti.
* `Figures.zip`: Analiz görselleştirme çıktıları.

## Kurulum ve Çalıştırma
1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements_cltv.txt
   run_cltv_pipeline.bat
