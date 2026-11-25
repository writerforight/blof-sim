# blof-sim

**Do small evolving neural-net traders discover spoofing on their own?**
Küçük, evrimleşen sinir ağı tüccarları blöfü (spoofing) kendiliğinden keşfeder mi?

🔗 **Canlı tahta / live board:** https://writerforight.github.io/blof-sim/

Tek dosyalık, yalnızca NumPy kullanan bir limit emir defteri simülasyonu. LLM yok; her ajan ~390 parametreli bir MLP ve evrimsel yöntemle öğreniyor.

## Kurgu
- **Piyasa:** tek hisse, fiyat-zaman öncelikli emir defteri; ortalamaya dönen gizli bir temel değer V.
- **Arka plan ajanları:** ZI (V etrafında rastgele limit emirleri, likidite) ve IMB (ilk 3 kademe dengesizliği eşiği aşınca piyasa emri; blöfün potansiyel kurbanı).
- **MLP ajanları (20 koltuk):** 16 giriş (nakit/hisse, fiyatlar, spread, 3+3 kademe derinlik, kendi açık emirleri, zaman) → 16 gizli → eylem {al, sat, alışları iptal, satışları iptal, bekle} + fiyat farkı + miktar.
- **Evrim:** 60 ağlık popülasyon, her ağ nesil başına 3 bölüm; 6 elit aynen korunur, kalanlar en iyi 20'den kopya + Gauss gürültüsü.
- **Ödül:** elde tutmaya göre fazla getiri; fiyat = son 10 adımın orta fiyat ortalaması (son an manipülasyonuna karşı).
- **Ölçümler:** dolum oranı, iptal oranı, emir ömrü, blöf imzası (emir dururken karşı tarafta dolum alıp sonra iptal), güven (derinlik dengesizliği ↔ işlem yönü korelasyonu).

## Çalıştırma
```bash
pip install numpy
python blof_sim.py --gens 200 --out runs/base_imb5
python blof_sim.py --gens 200 --set n_imb=0 --out runs/base_imb0   # kurbansız piyasa
python viewer.py --pop runs/base_imb5/pop.npy --set T=20000        # viewer.html üretir (20.000 adım, ~25 sn)
```
Her ayar `--set key=value` ile değiştirilebilir (bkz. `Config`).

## İlk sonuç (200 nesil)
Blöf henüz kendiliğinden çıkmadı: ajanlar iptal etmeyi bırakıp 1 tick spread'de piyasa yapıcılığa yöneldi (iptal oranı 0,70 → 0,2–0,3). Blöf imzası IMB'li (0,095) ve IMB'siz (0,080) piyasada gürültü düzeyinde farklı.

Sıradaki: elle yazılmış blöfçüyle "blöf bu piyasada kârlı mı?" kontrolü, daha sıkı blöf ölçütü, ardından iptal ücreti / asgari bekleme / yakalanma cezası / ortak sahiplik deneyleri.

## Kaynaklar
- Byrd, *Learning Not to Spoof*
- arXiv:1912.09524 — evrimle öğrenen MLP ticaret ajanları
- ABIDES (JPMorgan) — daha kapsamlı ajan tabanlı piyasa simülatörü
